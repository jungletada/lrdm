import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, group_ids: torch.Tensor, temperature: float = 0.1, normalize: bool = True):
    """
    z: [N, D]  同一batch内混合了 (晴, 多风格) 的多个样本；同一场景用同一 group_id
    group_ids: [N]  同ID为正样本；自身不计为正
    return: 标量 SupCon 损失（对称）
    """
    assert z.dim() == 2 and group_ids.dim() == 1 and z.size(0) == group_ids.size(0)
    if normalize:
        z = F.normalize(z, dim=1)
    N = z.size(0)
    logits = (z @ z.t()) / temperature                 # [N,N]
    # mask_self: 去除自身
    mask_self = torch.eye(N, dtype=torch.bool, device=z.device)
    # 正样本掩码（同 group 且非自身）
    pos_mask = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)) & (~mask_self)  # [N,N]
    # 对每个 i：分子=对所有正样本 j 的 exp(sim_ij) 之和；分母=对所有 k≠i 的 exp(sim_ik) 之和
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # 稳定
    exp_logits = torch.exp(logits) * (~mask_self)                 # 去掉自身
    pos_exp = (exp_logits * pos_mask).sum(dim=1) + 1e-12
    all_exp = exp_logits.sum(dim=1) + 1e-12
    # 对没有正样本的锚（极少见），掩蔽不计
    valid = pos_mask.any(dim=1)
    loss = -torch.log(pos_exp[valid] / all_exp[valid]).mean()
    return loss


class PatchGANMultiDomain(nn.Module):
    def __init__(self, in_channels: int, num_domains: int, ndf: int = 64, n_layers: int = 3, spectral: bool = False):
        super().__init__()
        kw, pad = 4, 1
        def block(ic, oc, s, norm=True):
            conv = nn.Conv2d(ic, oc, kw, s, pad, bias=not norm)
            if spectral: conv = nn.utils.spectral_norm(conv)
            layers = [conv]
            if norm: layers += [nn.InstanceNorm2d(oc, affine=True)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        seq = [block(in_channels, ndf, 2, norm=False)]
        nf = ndf
        for _ in range(1, n_layers):
            seq += [block(nf, min(nf*2, 512), 2)]
            nf = min(nf*2, 512)
        seq += [block(nf, nf, 1)]            # 感受野进一步扩大
        self.head = nn.Conv2d(nf, num_domains, kw, 1, pad)  # 多类域 logits
        if spectral: self.head = nn.utils.spectral_norm(self.head)
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        feat = self.net(x)
        return self.head(feat)  # [B, num_domains, h', w']


def domain_ce_loss(logits: torch.Tensor, domain_index: int):
    """
    logits: [B, num_domains, h', w']
    domain_index: int in [0, num_domains-1]
    """
    B, C, H, W = logits.shape
    target = torch.full((B, H, W), domain_index, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def coral_loss(feat_a: torch.Tensor, feat_b: torch.Tensor, eps: float = 1e-5):
    """
    支持 [B,D] 或 [B,D,H,W]；空间维会展平进样本维
    """
    def flatten(x):
        if x.dim() == 4:  # [B,D,H,W] -> [N,D]
            B, D, H, W = x.shape
            x = x.permute(0,2,3,1).reshape(B*H*W, D)
        elif x.dim() != 2:
            raise ValueError("feat must be [B,D] or [B,D,H,W]")
        return x

    Xa, Xb = flatten(feat_a), flatten(feat_b)
    Xa = Xa - Xa.mean(0, keepdim=True); Xb = Xb - Xb.mean(0, keepdim=True)
    na = max(1, Xa.size(0)-1); nb = max(1, Xb.size(0)-1)
    Ca = (Xa.t() @ Xa) / na; Cb = (Xb.t() @ Xb) / nb
    d = Ca.size(0)
    return ((Ca - Cb).pow(2).sum()) / (4.0 * d * d + eps)


def coral_multi_to_sunny(Tss, Tsw_list):
    """
    将每个坏天气输出与晴天输出对齐： Σ_j CORAL(Tsw[j], Tss)
    """
    return sum(coral_loss(Tss, Tsw) for Tsw in Tsw_list) / max(1, len(Tsw_list))


lam1, lam2, lam3, lam4, lam5, lam6 = 1.0, 2.0, 0.5, 0.1, 0.1, 0.1

# 一个小工具：把 [B, d, H, W] 聚合为 [B, d]
def gap(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(-2, -1)) if x.dim() == 4 else x

# CE 封装：PatchGAN 的 patch-wise 多类交叉熵
def ce_patch(logits: torch.Tensor, label: int) -> torch.Tensor:
    # logits: [B, C_dom, h', w'];  target: [B, h', w']
    B, C, H, W = logits.shape
    tgt = torch.full((B, H, W), int(label), device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, tgt)

# ============ 一个完整的 step ============

def training_step(batch):
    """
    batch 需解包为：
      Is: [B, ...], Ts: [B, ...],
      Iw_list: list(len=m) of [B, ...],
      Tw_list: list(len=m) of [B, ...],
      scene_ids: [B] (torch.long)，同一场景同一个ID
    """
    Is, Ts, Iw_list, Tw_list, scene_ids = batch
    assert isinstance(Iw_list, (list, tuple)) and isinstance(Tw_list, (list, tuple))
    m = len(Iw_list)                             # 多风格视图数
    assert m == len(Tw_list), "Iw_list/Tw_list 长度必须一致"
    B = Is.size(0)

    # 1) 前向
    Tss = f_theta(Is, Ts)                        # 晴 -> 晴（恒等）
    Tsw_list = [f_theta(Iw_list[j], Tw_list[j]) for j in range(m)]  # 多风格 -> 晴

    # 2) 恒等 & 恢复
    L_id  = (Tss - Ts).abs().mean()
    L_rec = torch.stack([(Tsw - Ts).abs().mean() for Tsw in Tsw_list]).mean()

    # 3) SupCon（多正样本；同一 scene_id 视为正样本集合）
    #   SupCon 参考：Khosla et al., NeurIPS 2020:contentReference[oaicite:3]{index=3}
    assert scene_ids.dtype == torch.long and scene_ids.dim() == 1 and scene_ids.size(0) == B
    Z_list = [gap(Tss)] + [gap(Tsw) for Tsw in Tsw_list]   # (m+1) 个 [B, D]
    Z = torch.cat(Z_list, dim=0)                           # [(m+1)*B, D]
    G = scene_ids.repeat(m + 1)                            # [(m+1)*B]
    L_supcon = supcon_loss(Z, G, temperature=0.1)

    # 4) 组内一致（方差，鼓励多风格输出坍缩到同一语义）
    stacked = torch.stack([gap(Tsw) for Tsw in Tsw_list], dim=0)  # [m, B, D]
    L_var = stacked.var(dim=0, unbiased=False).mean()             # 标量

    # 5) 多源 CORAL（坏天气→晴天，二阶统计对齐；Deep CORAL:contentReference[oaicite:4]{index=4}）
    L_coral = coral_multi_to_sunny(Tss, Tsw_list)

    # 6) PatchGAN 多类域对抗（K+1 域；PatchGAN 思想来自 pix2pix:contentReference[oaicite:5]{index=5}）
    #    若 D 内部接了 GRL，则生成端的对抗项就是下面的 CE；若未接 GRL，可用 L_adv = -L_D。
    logits_s = D(Tss)                             # 晴域 logits: [B, C_dom, h', w']
    logits_w = [D(Tsw) for Tsw in Tsw_list]       # 各坏天气域 logits
    # 约定 domain 索引：晴=0；每种坏天气依次 1..m
    sunny_id = 0
    domain_ids = list(range(1, m + 1))
    L_D = ce_patch(logits_s, sunny_id) + torch.stack([
        ce_patch(lw, domain_ids[j]) for j, lw in enumerate(logits_w)
    ]).mean()

    use_grl = True  # 如果 D 或连接到 D 的桥里实现了 GRL，就置 True
    L_adv = L_D if use_grl else (-L_D)

    # 7) 总损失
    L = lam1 * L_id + lam2 * L_rec + lam3 * L_supcon + lam4 * L_var + lam5 * L_coral + lam6 * L_adv

    # 你也可以返回一个 dict 便于 log
    logs = {
        "L": L, "L_id": L_id, "L_rec": L_rec, "L_supcon": L_supcon,
        "L_var": L_var, "L_coral": L_coral, "L_adv": L_adv, "L_D": L_D
    }
    return L, logs
