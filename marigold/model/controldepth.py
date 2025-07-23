import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


# 1x1 Conv with zero initialization
class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ControlMarigold(UNet2DConditionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Copy block channel numbers from base UNet config
        block_out_channels = self.config.block_out_channels
        # Build ZeroConvs for down blocks and mid block
        self.control_zero_convs = nn.ModuleList()
        for ch in block_out_channels:
            self.control_zero_convs.append(ZeroConv2d(ch, ch))
        self.control_zero_convs.append(ZeroConv2d(block_out_channels[-1], block_out_channels[-1]))

    def forward(
        self,
        **kwargs,
    ):
        super.forward(**kwargs)
    # def forward_control(self, cond_latent, timestep, encoder_hidden_states, **kwargs):
    #     """
    #     Forward only the control branch and output the control feature maps (after zero conv).
    #     cond_latent: conditioning input (e.g. image latent)
    #     """
    #     control_feats = []
    #     sample = self.conv_in(cond_latent)
    #     # Down blocks (encoder)
    #     for i, down_block in enumerate(self.down_blocks):
    #         if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
    #             sample, _ = down_block(sample, timestep, encoder_hidden_states)
    #         else:
    #             sample, _ = down_block(sample, timestep)
    #         # After each block, apply zero conv (collect for later addition)
    #         control_feats.append(self.control_zero_convs[i](sample))
    #     # Mid block
    #     if hasattr(self, "mid_block"):
    #         if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
    #             sample = self.mid_block(sample, timestep, encoder_hidden_states)
    #         else:
    #             sample = self.mid_block(sample, timestep)
    #     control_feats.append(self.control_zero_convs[-1](sample))
    #     return control_feats

   