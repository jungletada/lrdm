# RAMiT (Reciprocal Attention Mixing Transformer)
# import os
# import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from einops import rearrange
from timm.models import register_notrace_function
from timm.layers import trunc_normal_, DropPath

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from util.etc_utils import denormalize # ramit_model.
from .common.mean_std import mean_std


def make_model(args, opts, pe):
    model = RAMiT(target_mode=args.target_mode,
                  img_norm=args.img_norm,
                  in_chans=opts['in_chans'],
                  dim = opts['dim'],
                  depths = opts['depths'],
                  num_heads = opts['num_heads'],
                  head_dim = opts['head_dim'],
                  chsa_head_ratio = opts['chsa_head_ratio'],
                  window_size = opts['window_size'],
                  hidden_ratio = opts['hidden_ratio'],
                  qkv_bias = opts['qkv_bias'],
                  mv_ver = opts['mv_ver'],
                  exp_factor=opts['exp_factor'],
                  expand_groups=opts['expand_groups'],
                  act_layer = opts['act_layer'],
                  norm_layer = opts['norm_layer'],
                  tail_mv = opts['tail_mv'],
                  attn_drop = opts['attn_drop'],
                  proj_drop = opts['proj_drop'],
                  drop_path = opts['drop_path'],
                  helper = opts['helper'],
                  mv_act = opts['mv_act']
                 )
    num_params = sum([p.numel() for _, p in model.named_parameters()])
    print(format(num_params, ','))
    return model


class ShallowModule(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1):
        super(ShallowModule, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, kernel_size//2)
        
    def forward(self, x):
        return self.conv(x)
    
    def flops(self, resolutions):
        return resolutions[0]*resolutions[1] * self.kernel_size*self.kernel_size * self.in_chans * self.out_chans


class QKVProjection(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=True):
        super(QKVProjection, self).__init__()
        self.dim = dim
        self.num_head = num_head
        
        self.qkv = nn.Conv2d(dim, 3*dim, 1, bias=qkv_bias)
        
    def forward(self, x):
        B, C, H, W = x.size()
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b (l c) h w -> b l c h w', l=self.num_head)
        return qkv
    
    def flops(self, resolutions):
        return resolutions[0]*resolutions[1] * 1*1 * self.dim * 3*self.dim


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww (xaxis matrix & yaxis matrix)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, window_size=8, shift=0, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(SpatialSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.window_size = window_size
        self.window_area = window_size**2
        self.shift = shift
        self.helper = helper
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)
        
        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))
        
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Conv2d(dim*num_head, dim*num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)
        
    def forward(self, qkv, ch=None):
        B, L, C, H, W = qkv.size()
        # window shift
        if self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2,-1))
        
        # window partition
        q,k,v = rearrange(qkv, 'b l c (h wh) (w ww) -> (b h w) l (wh ww) c', 
                          wh=self.window_size, ww=self.window_size).chunk(3, dim=-1) # [B_, L1, hw, C/L] respectively
        if ch is not None and self.helper: # [B, C, H, W]
            if self.shift > 0:
                ch = torch.roll(ch, shifts=(-self.shift, -self.shift), dims=(-2,-1))
            ch = rearrange(ch, 'b (l c) (h wh) (w ww) -> (b h w) l (wh ww) c',
                           l=self.total_head-self.num_head, wh=self.window_size, ww=self.window_size) # [B_, L1, hw, C/L]
            ch = torch.mean(ch, dim=1, keepdim=True) # head squeeze [B_, 1, hw, C/L]
            v = v*ch # [B_, L1, hw, C/L]
            
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2,-1) # [B_, L1, hw, hw]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        
        attn = attn + self._get_rel_pos_bias()
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        
        x = attn @ v # [B_, L1, hw, C/L]
        
        # window unpartition + head merge
        x = window_unpartition(x, (H,W), self.window_size) # [B, L1*C/L, H, W]
        x = self.proj_drop(self.proj(x))
        
        # window reverse shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(-2,-1))
        
        return x
    
    def flops(self, resolutions):
        H,W = resolutions
        num_wins = H//self.window_size * W//self.window_size
        flops = self.num_head * H*W * self.dim if self.helper else 0 # v = v*ch
        flops += num_wins * self.num_head * self.window_area * self.dim * self.window_area # attn = Q@K^T
        flops += num_wins * self.num_head * self.window_area * self.window_area * self.dim # attn@V
        flops += H*W * 1*1 * self.num_head*self.dim * self.num_head*self.dim # self.proj
        return flops
    
@register_notrace_function
def window_unpartition(x, resolutions, window_size):
    return rearrange(x, '(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)', 
                     h=resolutions[0]//window_size, w=resolutions[1]//window_size, wh=window_size)
    
    
class ChannelSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(ChannelSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.helper = helper
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Conv2d(dim*num_head, dim*num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, qkv, sp=None):
        B, L, C, H, W = qkv.size()
        
        q,k,v = rearrange(qkv, 'b l c h w -> b l c (h w)').chunk(3, dim=-2) # [B, L2, C/L, HW]
        if sp is not None and self.helper:
            sp = torch.mean(sp, dim=1, keepdim=True) # channel squeeze # [B, 1, H, W]
            sp = rearrange(sp, 'b (l c) h w -> b l c (h w)', l=1) # [B, 1, 1, HW]
            v = v*sp # [B, L2, C/L, HW]
        
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2,-1) # [B, L2, C/L, C/L]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v # [B, L2, C/L, HW]
        
        # head merge
        x = rearrange(x, 'b l c (h w) -> b (l c) h w', h=H) # [B, L2*C/L, H, W]
        x = self.proj_drop(self.proj(x)) # [B, L2*C/L, H, W]
        
        return x
    
    def flops(self, resolutions):
        H,W = resolutions
        flops = self.num_head * self.dim * H*W if self.helper else 0 # v = v*sp
        flops += self.num_head * self.dim * H*W * self.dim # attn = Q@K^T
        flops += self.num_head * self.dim * self.dim * H*W # attn@V
        flops += H*W * 1*1 * self.num_head*self.dim * self.num_head*self.dim # self.proj
        return flops
    
    
class ReshapeLayerNorm(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(ReshapeLayerNorm, self).__init__()
        
        self.dim = dim
        self.norm = norm_layer(dim)
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x
    
    def flops(self, resolutions):
        H,W = resolutions
        flops = 0
        flops += H*W * self.dim
        return flops
   
    
class MobiVari1(nn.Module): # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim
        
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size//2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()
        
    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x))+x))
        return out + x if self.dim==self.out_dim else out
    
    def flops(self, resolutions):
        H,W = resolutions
        flops = H*W * self.kernel_size*self.kernel_size * self.dim  +  H*W * 1*1 * self.dim * self.out_dim # self.dw_conv + self.pw_conv
        return flops


class MobiVari2(MobiVari1): # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim*exp_factor)
        expand_dim = expand_dim+(expand_groups-expand_dim%expand_groups)
        self.expand_dim = expand_dim
        
        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size//2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)
        
    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1)+x1))
        return out + x if self.dim==self.out_dim else out
    
    def flops(self, resolutions):
        H,W = resolutions
        flops = H*W * 1*1 * (self.dim//self.expand_groups) * self.expand_dim # self.exp_conv
        flops += H*W * self.kernel_size*self.kernel_size * self.expand_dim # self.dw_conv
        flops += H*W * 1*1 * self.expand_dim * self.out_dim # self.pw_conv
        return flops
 
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_ratio, act_layer=nn.GELU, bias=True, drop=0.0):
        super(FeedForward, self).__init__()
        
        self.dim = dim
        self.hidden_ratio = hidden_ratio
        
        self.hidden = nn.Conv2d(dim, int(dim*hidden_ratio), 1, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.out = nn.Conv2d(int(dim*hidden_ratio), dim, 1, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.act = act_layer()
        
    def forward(self, x):
        return self.drop2(self.out(self.drop1(self.act(self.hidden(x)))))
    
    def flops(self, resolutions):
        H,W = resolutions
        flops = 2 * H*W * 1*1 * self.dim * self.dim*self.hidden_ratio # self.hidden + self.out
        return flops
    
    
class NoLayer(nn.Identity):
    def __init__(self):
        super(NoLayer, self).__init__()
    def flops(self, resolutions):
        return 0
    def forward(self, x, **kwargs):
        return x.flatten(1,2)
    
    
class DRAMiTransformer(nn.Module): # Reciprocal Attention Transformer Block
    def __init__(self, dim, num_head, chsa_head_ratio, window_size=8, shift=0, head_dim=None, qkv_bias=True, mv_ver=1,
                 hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, helper=True,
                 mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(DRAMiTransformer, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.chsa_head = int(num_head*chsa_head_ratio)
        self.shift = shift
        self.helper = helper
        
        self.qkv_proj = QKVProjection(dim, num_head, qkv_bias=qkv_bias)
        self.sp_attn = SpatialSelfAttention(dim//num_head, num_head-self.chsa_head, num_head, 
                                            window_size, shift, attn_drop, proj_drop, helper) if num_head-self.chsa_head != 0 else NoLayer()
        self.ch_attn = ChannelSelfAttention(dim//num_head, self.chsa_head, num_head, attn_drop, proj_drop, helper) if self.chsa_head != 0 else NoLayer()
        if mv_ver==1:
            self.mobivari = MobiVari1(dim, 3, 1, act=mv_act)
        elif mv_ver==2:
            self.mobivari = MobiVari2(dim, 3, 1, act=mv_act, out_dim=None, exp_factor=exp_factor, expand_groups=expand_groups)
            
        self.norm1 = norm_layer(dim)
        
        self.ffn = FeedForward(dim, hidden_ratio, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, sp_=None, ch_=None):
        B, C, H, W = x.size()
        
        # QKV projection + head split
        qkv = self.qkv_proj(x) # [B, L, C, H, W]
        
        # SP-SA / CH-SA
        sp = self.sp_attn(qkv[:,:self.num_head-self.chsa_head], ch=ch_) # [B, L1*C/L, H, W]
        ch = self.ch_attn(qkv[:,self.num_head-self.chsa_head:], sp=sp_) # [B, L2*C/L, H, W]
        attn0 = self.mobivari(torch.cat([sp, ch], dim=1)) # merge [B, C, H, W]
        attn = self.drop_path(self.norm1(attn0)) + x # LN, skip connection [B, C, H, W]
        
        # FFN
        out = self.drop_path(self.norm2(self.ffn(attn))) + attn # FFN, LN, skip connection [B, C, H, W]
        
        return out, sp, ch, attn0
    
    def flops(self, resolutions):
        flops = self.qkv_proj.flops(resolutions)
        flops += self.sp_attn.flops(resolutions)
        flops += self.ch_attn.flops(resolutions)
        flops += self.mobivari.flops(resolutions)
        flops += self.norm1.flops(resolutions)
        flops += self.ffn.flops(resolutions)
        flops += self.norm2.flops(resolutions)
        params = sum([p.numel() for n,p in self.named_parameters()])
        return flops
   
    
class EncoderStage(nn.Module):
    def __init__(self, depth, dim, num_head, chsa_head_ratio, window_size=8, head_dim=None, 
                 qkv_bias=True, mv_ver=1, hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0, helper=True, mv_act=nn.LeakyReLU):
        super(EncoderStage, self).__init__()
        
        self.depth = depth
        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        shift = window_size//2
        
        self.blocks = nn.ModuleList()
        for d in range(depth):
            self.blocks.add_module(
                f'block{d}', 
                DRAMiTransformer(dim, num_head, chsa_head_ratio, window_size, 0 if d%2==0 else shift, 
                                 head_dim, qkv_bias, mv_ver, hidden_ratio, act_layer, norm_layer,
                                 attn_drop, proj_drop, drop_path, helper, mv_act))
            
    def forward(self, x):
        sp, ch = None, None
        for i, blk in enumerate(self.blocks):
            x, sp, ch, attn = blk(x, sp, ch)
            sp, ch = (None, None) if (sp.size(1)==0 or ch.size(1)==0) else (sp, ch) # pure SPSA or CHSA
        return x, attn
    
    def flops(self, resolutions):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops(resolutions)
        return flops


class Downsizing(nn.Module):
    """ Patch Merging Layer.

    Args:
        dim (int): Number of input dimension (channels).
        downsample_dim (int, optional): Number of output dimension (channels) (dim if not set).  Default: None
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, downsample_dim=None, norm_layer=ReshapeLayerNorm, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(Downsizing, self).__init__()
        self.dim = dim
        self.downsample_dim = downsample_dim or dim
        self.norm = norm_layer(4*dim)
        if mv_ver==1:
            self.reduction = MobiVari1(4*dim, 3, 1, act=mv_act, out_dim=self.downsample_dim)
        elif mv_ver==2:
            self.reduction = MobiVari2(4*dim, 3, 1, act=mv_act, out_dim=self.downsample_dim, exp_factor=exp_factor, expand_groups=expand_groups)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Concat 2x2
        x0 = x[:, :, 0::2, 0::2]  # [B, C, H/2, W/2], top-left
        x1 = x[:, :, 0::2, 1::2]  # [B, C, H/2, W/2], top-right
        x2 = x[:, :, 1::2, 0::2]  # [B, C, H/2, W/2], bottom-left
        x3 = x[:, :, 1::2, 1::2]  # [B, C, H/2, W/2], bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=1)  # [B, 4C, H/2, W/2]
        return self.reduction(self.norm(x)) # [B, C, H/2, W/2]
    
    def flops(self, resolutions):
        H,W = resolutions
        flops = self.norm.flops((H//2,W//2)) + self.reduction.flops((H//2,W//2))
        return flops  


class Bottleneck(nn.Module):
    def __init__(self, dim, num_stages, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm, 
                 mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(Bottleneck, self).__init__()
        self.dim = dim
        self.cat_dim = dim * num_stages
        if mv_ver==1:
            self.mobivari = MobiVari1(self.cat_dim, 3, 1, act=mv_act, out_dim=dim)
        elif mv_ver==2:
            self.mobivari = MobiVari2(self.cat_dim, 3, 1, act=mv_act, out_dim=dim, exp_factor=exp_factor, expand_groups=expand_groups)
        self.act = act_layer()
        self.norm = norm_layer(dim)
                
    def forward(self, x_list):
        xs = x_list[0]
        new_x = []
        for i in range(len(x_list[1:])):
            x_ = x_list[i+1] + F.leaky_relu(xs)
            new_x.append(x_)
        new_x = self.norm(self.mobivari(torch.cat(new_x, dim=1)))
        return new_x
    
    def flops(self, resolutions):
        H,W = resolutions
        flops = 0
        # self.shallow_down (iterative max-pool)
        flops += (H//2)*(W//2) * 2*2 * self.dim # shallow-down into stage2 output (iter1)
        flops += (H//2)*(W//2) * 2*2 * self.dim # shallow-down into stage3 output (iter1)
        flops += (H//4)*(W//4) * 2*2 * self.dim # shallow-down into stage3 output (iter2)
        flops += self.mobivari.flops((H,W))
        flops += self.norm.flops((H,W))
        return flops
  
    
class HRAMi(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(HRAMi, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        if mv_ver==1:
            self.mobivari = MobiVari1(dim * 4, kernel_size, stride, act=mv_act, out_dim=dim)
        elif mv_ver==2:
            self.mobivari = MobiVari2(dim * 4, kernel_size, stride, act=mv_act, out_dim=dim, exp_factor=exp_factor, expand_groups=expand_groups)
            
    def forward(self, attn_list):
        x = torch.cat(attn_list, dim=1)
        x = self.mobivari(x)
        return x
    
    def flops(self, resolutions):
        return self.mobivari.flops(resolutions)

    
class Reconstruction(nn.Module):
    def __init__(self, out_chans, dim, kernel_size=3, stride=1, num_mv=2, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(Reconstruction, self).__init__()
        
        self.mobivari = nn.ModuleList()
        for i in range(num_mv):
            if mv_ver==1:
                self.mobivari.add_module(f'mobivari{i}', MobiVari1(dim, kernel_size, stride, mv_act))
            elif mv_ver==2:
                self.mobivari.add_module(f'mobivari{i}', MobiVari2(dim, kernel_size, stride, mv_act, None, exp_factor, expand_groups))
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size//2)
        self.final_conv = nn.Conv2d(dim, out_chans, kernel_size, stride, kernel_size//2)
        
    def forward(self, x):
        for mobivari in self.mobivari:
            x = mobivari(x)
        return self.final_conv(self.conv(x))
            
    def flops(self, resolutions):
        H,W = resolutions
        flops = 0
        for mobivari in self.mobivari: # self.mobivari
            flops += mobivari.flops((H,W))
        flops += H*W * self.kernel_size*self.kernel_size * self.dim * self.out_chans*(self.upscale**2) # self.conv
        flops += H*W * self.kernel_size*self.kernel_size * self.out_chans * self.out_chans # self.final_conv
        return flops
  

class RAMiTModule(nn.Module):
    def __init__(self, 
                 in_chans=4, 
                 dim=4, 
                 depths=(6,4,4,6), 
                 num_heads=(4,4,4,4), 
                 head_dim=None, 
                 chsa_head_ratio=0.25,
                 window_size=4, 
                 hidden_ratio=2.0, 
                 qkv_bias=True, 
                 mv_ver=1, 
                 exp_factor=1.2, 
                 expand_groups=4,
                 act_layer=nn.GELU, 
                 norm_layer=ReshapeLayerNorm, 
                 tail_mv=2, 
                 target_mode='light_dr', 
                 img_norm=True,
                 attn_drop=0.0, 
                 proj_drop=0.0, 
                 drop_path=0.0, 
                 helper=True, 
                 mv_act=nn.LeakyReLU):
        super(RAMiTModule, self).__init__()
        
        self.unit = 2 ** (len(depths)-2) * window_size
        self.in_chans = in_chans
        self.dim = dim
        self.depths = depths
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.hidden_ratio = hidden_ratio
        self.qkv_bias = qkv_bias
        self.act_layer = act_layer
        norm_layer = ReshapeLayerNorm if norm_layer == 'ReshapeLayerNorm' else norm_layer
        self.norm_layer = norm_layer = ReshapeLayerNorm if norm_layer == 'ReshapeLayerNorm' else norm_layer
        self.tail_mv = tail_mv
        
        self.scale = 1
        self.mean, self.std = mean_std(self.scale, target_mode)
        self.target_mode = target_mode
        self.img_norm = img_norm
        self.shallow = ShallowModule(in_chans, dim, 3, 1)
        self.stage1 = EncoderStage(depths[0], dim, num_heads[0], chsa_head_ratio, window_size, head_dim, qkv_bias, mv_ver, 
                                   hidden_ratio, act_layer, norm_layer, attn_drop, proj_drop, drop_path, helper, mv_act)
        self.stage2 = EncoderStage(depths[1], dim, num_heads[1], chsa_head_ratio, window_size, head_dim, qkv_bias, mv_ver, 
                                   hidden_ratio, act_layer, norm_layer, attn_drop, proj_drop, drop_path, helper, mv_act)
        self.stage3 = EncoderStage(depths[2], dim, num_heads[2], chsa_head_ratio, window_size, head_dim, qkv_bias, mv_ver, 
                                   hidden_ratio, act_layer, norm_layer, attn_drop, proj_drop, drop_path, helper, mv_act)
        self.bottleneck = Bottleneck(dim, len(depths)-1, act_layer, norm_layer, mv_ver, mv_act)
        self.stage4 = EncoderStage(depths[3], dim, num_heads[3], chsa_head_ratio, window_size, head_dim, qkv_bias, mv_ver, 
                                   hidden_ratio, act_layer, norm_layer, attn_drop, proj_drop, drop_path, helper, mv_act)
        self.attn_mix = HRAMi(dim, 3, 1, mv_ver, mv_act)
        self.to_target = Reconstruction(in_chans, dim, 3, 1, tail_mv, mv_ver, mv_act, exp_factor, expand_groups)
        
        self.apply(self._init_weights)

    def forward(self, rgb_latent):
        """
        Forward pass of the fusion module.
        Inputs:
          rgb_latent: tensor of shape [B, 4, H, W]
          split as (rgb_latent, depth_latent): tensors of shape [B, 4, H, W].
        Output:
          out: tensor of shape [B, 320, H, W] combining all.
        """
        shallow = self.shallow(rgb_latent)
        o1, attn1 = self.stage1(shallow) # [B, C, H//8, W//8]
        o2, attn2 = self.stage2(o1)         # [B, C, H//8, W//8]
        o3, attn3 = self.stage3(o2)         # [B, C, H//8, W//8]
        
        ob = self.bottleneck([shallow, o1, o2, o3])   # [B, C, H//8, W//8]
        
        o4, attn4 = self.stage4(ob)             # [B, C, H//8, W//8]
        mix = self.attn_mix([attn1, attn2, attn3, attn4]) # [B, C, H//8, W//8]
        o4 = o4 * mix   # [B, C, H, W]
        
        ra_latent = self.to_target(o4 + shallow)    # global skip connection
        ra_latent = ra_latent + rgb_latent
        
        return ra_latent
    
    def _init_weights(self, m):
        # Swin V2 manner
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        # Additionally, if this is the to_target layer, initialize its weights and bias to zero
        if hasattr(self, 'to_target') and m is self.to_target:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd
   
    
class RAMiT(nn.Module):
    def __init__(self, **kwargs):
        super(RAMiT, self).__init__()
        self.ramit_module = RAMiTModule(**kwargs)
        self.new_conv_in = nn.Conv2d(12, 320, kernel_size=3, padding=1)

    def forward(self, sample):
        """
        Forward pass of the fusion module.
        Inputs:
          sample: tensor of shape [B, 8, H, W]
          split as (rgb_latent, depth_latent): tensors of shape [B, 4, H, W].
        Output:
          out: tensor of shape [B, 320, H, W] combining all.
        """
        rgb_latent, depth_latent = torch.split(sample, [4, 4], dim=1)
        ra_latent = self.ramit_module(rgb_latent)
        
        out = torch.cat((rgb_latent, ra_latent, depth_latent), dim=1) # [B, 12, H, W]
        out = self.new_conv_in(out)     # [B, 320, H, W]
        
        return out
    
    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd


if __name__ == '__main__':
    H = 352 // 8
    W = 1216 // 8
    x = torch.randn(4, 4, H, W)
    
    model = RAMiT()
    output = model(x)