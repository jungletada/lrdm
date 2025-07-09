from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)


class ControlDepth(UNet2DConditionModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    