import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class CharbonnierLoss(nn.Module):
    """
    Charbonnier (a.k.a. pseudo-Huber) loss:
      L(x, y) = mean( sqrt( (x - y)^2 + eps^2 ) )
    Args:
      eps: smoothing term (typical 1e-3 ~ 1e-6)
      reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, eps: float = 1e-5, reduction: Literal['mean','sum','none']='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape = N,C,H,W


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.reshape(1, 1, 1, -1)
    window_2d = (window_1d.transpose(-1, -2) @ window_1d).squeeze(0).squeeze(0)
    window_2d = window_2d / window_2d.sum()
    return window_2d


def _ssim_conv(x: torch.Tensor, window: torch.Tensor, C: int):
    """
    Depthwise conv2d with 'same' padding using a shared KxK Gaussian window.
    x: [N, C, H, W]
    window: [1, 1, K, K] (Gaussian kernel)
    """
    kH, kW = window.shape[-2], window.shape[-1]       # <-- use spatial dims
    weight = window.expand(C, 1, kH, kW)              # [C, 1, K, K]
    return F.conv2d(x, weight, padding=(kH // 2, kW // 2), groups=C)


class SSIMLoss(nn.Module):
    """
    SSIM loss = 1 - SSIM(x, y), averaged over batch.
    Works on NCHW, expects inputs in [0, 1].
    Args:
      window_size: odd int, e.g. 11
      sigma: Gaussian sigma, e.g. 1.5
      C1, C2: stability constants; defaults assume L=1 (images in [0,1])
      channel_average: average SSIM over channels (True) or keep per-channel
    """
    def __init__(self, 
                window_size: int = 11, 
                sigma: float = 1.5,
                C1: float = (0.01 ** 2), 
                C2: float = (0.03 ** 2),
                channel_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2
        self.channel_average = channel_average

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape and x.dim() == 4, "Inputs must be NCHW and have the same shape."
        N, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        window = _gaussian_window(self.window_size, self.sigma, device, dtype)
        window = window.unsqueeze(0).unsqueeze(0)  # 1x1xKxK

        mu_x = _ssim_conv(x, window, C)
        mu_y = _ssim_conv(y, window, C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = _ssim_conv(x * x, window, C) - mu_x2
        sigma_y2 = _ssim_conv(y * y, window, C) - mu_y2
        sigma_xy = _ssim_conv(x * y, window, C) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2))

        if self.channel_average:
            ssim_val = ssim_map.mean()  # average over N,C,H,W
        else:
            ssim_val = ssim_map.flatten(2).mean(-1).mean(0)  # per-channel average over N,H,W

        return 1.0 - ssim_val  # SSIM loss

