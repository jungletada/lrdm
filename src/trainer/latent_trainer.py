import os
import logging
from tqdm import tqdm
from typing import List, Union, Optional
from PIL import Image
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.seeding import generate_seed_sequence


def supcon_loss(
    z: torch.Tensor, 
    group_ids: torch.Tensor, 
    temperature: float = 0.1, 
    normalize: bool = True):
    """
    z: [N, D]  同一batch内混合了 (晴, 多风格) 的多个样本；同一场景用同一 group_id
    group_ids: [N]  同ID为正样本；自身不计为正
    return: 标量 SupCon 损失（对称）
    """
    assert z.dim() == 2 and group_ids.dim() == 1 and z.size(0) == group_ids.size(0)
    if normalize:
        z = F.normalize(z, dim=1)
    N = z.size(0)
    logits = (z @ z.t()) / temperature  # [N,N]
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


def coral_multi_to_sunny(sunny_pred, pred_weather_list):
    """
    将每个坏天气输出与晴天输出对齐： Σ_j CORAL(weather[j], sunny)
    """
    return sum(coral_loss(sunny_pred, pred_wtr) for pred_wtr in pred_weather_list) / max(1, len(pred_weather_list))


lam1, lam2, lam3, lam4, lam5, lam6 = 1.0, 2.0, 0.5, 0.1, 0.1, 0.1


def global_average_pooling(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(-2, -1)) if x.dim() == 4 else x

# CE 封装：PatchGAN 的 patch-wise 多类交叉熵
def ce_patch(logits: torch.Tensor, label: int) -> torch.Tensor:
    # logits: [B, C_dom, h', w'];  target: [B, h', w']
    B, C, H, W = logits.shape
    tgt = torch.full((B, H, W), int(label), device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, tgt)


class RAMiTLatentTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: nn.Module,
        train_dataloader: DataLoader,
        visualize_dataset,
        device,
        out_dir_ckpt,
        accumulation_steps: int,
    ):
        self.cfg: OmegaConf = cfg
        self.model: nn.Module = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.train_loader: DataLoader = train_dataloader
        self.visualize_dataset: DataLoader = visualize_dataset
        self.accumulation_steps: int = accumulation_steps
        
        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Building Loss
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.train_metrics = MetricTracker(*["loss"])
    
        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.smooth_l1_loss = nn.SmoothL1Loss()

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def train_process(self, t_end=None):
        logging.info("Start training")
        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>
                sunny, weather_list, scene_ids = batch['sunny'], batch['weather'], batch['index']
                assert isinstance(weather_list, (list, tuple))
                num_domains = len(weather_list)
                # 1) 前向
                pred_sunny = self.model(sunny[0], sunny[1])       # 晴 -> 晴（恒等）
                pred_weathers = [self.model(weather_list[j][0], weather_list[j][1]) for j in range(num_domains)]  # 多风格 -> 晴

                # 2) 恒等 & 恢复
                L_idendity  = self.smooth_l1_loss(pred_sunny, sunny[1])
                L_reconstruct = torch.stack([self.smooth_l1_loss(pred_wtr, sunny[1]) for pred_wtr in pred_weathers]).mean()

                # 3) SupCon（多正样本；同一 scene_id 视为正样本集合）
                #   SupCon 参考：Khosla et al., NeurIPS 2020:contentReference[oaicite:3]{index=3}
                assert scene_ids.dtype == torch.int64 and scene_ids.dim() == 1
                output_pooling_list = [global_average_pooling(pred_sunny)] + \
                         [global_average_pooling(pred_wtr) for pred_wtr in pred_weathers]  # (m+1) 个 [B, D]
                output_pool = torch.cat(output_pooling_list, dim=0)   # [(m+1)*B, D]
                group_ids = scene_ids.repeat(num_domains + 1)         # [(m+1)*B]
                L_supcon = supcon_loss(output_pool, group_ids, temperature=0.1)

                # 4) 组内一致（方差，鼓励多风格输出坍缩到同一语义）
                stacked = torch.stack([global_average_pooling(Tsw) for Tsw in pred_weathers], dim=0)  # [m, B, D]
                L_var = stacked.var(dim=0, unbiased=False).mean()             # 标量

                # 5) 多源 CORAL（坏天气→晴天，二阶统计对齐；Deep CORAL:contentReference[oaicite:4]{index=4}）
                L_coral = coral_multi_to_sunny(pred_sunny, pred_weathers)

                # 6) PatchGAN 多类域对抗（K+1 域；PatchGAN 思想来自 pix2pix:contentReference[oaicite:5]{index=5}）
                #    若 D 内部接了 GRL，则生成端的对抗项就是下面的 CE；若未接 GRL，可用 L_adv = -L_D。
                logits_s = self.discriminator(pred_sunny)   # 晴域 logits: [B, C_dom, h', w']
                logits_w = [self.discriminator(pred_wtr) for pred_wtr in pred_weathers]       # 各坏天气域 logits
                
                # 约定 domain 索引：晴=0；每种坏天气依次 1..m
                sunny_id = 0
                domain_ids = list(range(1, num_domains + 1))
                L_discrinative = ce_patch(logits_s, sunny_id) + torch.stack([
                    ce_patch(lw, domain_ids[j]) for j, lw in enumerate(logits_w)
                ]).mean()

                use_grl = True  # 如果 D 或连接到 D 的桥里实现了 GRL，就置 True
                L_adv = L_discrinative if use_grl else (-L_discrinative)

                # 7) 总损失
                L = lam1 * L_idendity + lam2 * L_reconstruct + lam3 * L_supcon + lam4 * L_var + lam5 * L_coral + lam6 * L_adv

        # 你也可以返回一个 dict 便于 log
        logs = {
            "L": L, "L_id": L_idendity, "L_rec": L_reconstruct, "L_supcon": L_supcon,
            "L_var": L_var, "L_coral": L_coral, "L_adv": L_adv, "L_D": L_discrinative
        }
        return L, logs

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Load data with "input_latent" and "target_latent"
                rgb_latent = batch["input_latent"].to(device)
                rgb_image = batch["rgb_norm"].to(device)
                target = batch["target_latent"].to(device)
                
                model_pred = self.model(rgb_image, rgb_latent)  # [B, 4, h, w]
                
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")
                    exit(0)

                loss = self.smooth_l1_loss(model_pred, target)
                # charbonnier_loss = self.charbonnier_loss(model_pred, target)
                # ssim_loss = self.ssim_loss(model_pred, target)
                # loss = 2 * smooth_l1_loss + charbonnier_loss + ssim_loss
                
                # self.train_metrics.update("smooth_l1_loss", smooth_l1_loss.item())
                # self.train_metrics.update("charbonnier_loss", charbonnier_loss.item())
                # self.train_metrics.update("ssim_loss", ssim_loss.item())
                self.train_metrics.update("loss", loss.item())
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                accumulated_step += 1
                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dict(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0