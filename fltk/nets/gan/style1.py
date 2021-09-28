from importlib import reload
from math import log

import torch

from .base import InferenceGenerator, import_from


class Style1Generator(InferenceGenerator):
    def __init__(self, size):
        with import_from("style-based-gan-pytorch"):
            import model

            reload(model)
            from model import EqualConv2d, StyledConvBlock, StyledGenerator
        super().__init__(StyledGenerator, 512)
        self.step = int(log(size, 2)) - 2
        if size == 512:
            del self.module.generator.progression, self.module.generator.to_rgb
            setattr(
                self.module.generator,
                "progression",
                torch.nn.ModuleList(
                    [
                        StyledConvBlock(512, 512, initial=True, fused=True),
                        StyledConvBlock(512, 512, upsample=True, fused=True),
                        StyledConvBlock(512, 512, upsample=True, fused=True),
                        StyledConvBlock(512, 256, upsample=True, fused=True),
                        StyledConvBlock(256, 128, upsample=True, fused=True),
                        StyledConvBlock(128, 64, upsample=True, fused=True),
                        StyledConvBlock(64, 32, upsample=True, fused=True),
                        StyledConvBlock(32, 16, upsample=True, fused=True),
                    ]
                ),
            )
            setattr(
                self.module.generator,
                "to_rgb",
                torch.nn.ModuleList(
                    [
                        EqualConv2d(512, 3, 1),
                        EqualConv2d(512, 3, 1),
                        EqualConv2d(512, 3, 1),
                        EqualConv2d(256, 3, 1),
                        EqualConv2d(128, 3, 1),
                        EqualConv2d(64, 3, 1),
                        EqualConv2d(32, 3, 1),
                        EqualConv2d(16, 3, 1),
                    ]
                ),
            )
        elif size == 256:
            del self.module.generator.progression, self.module.generator.to_rgb
            setattr(
                self.module.generator,
                "progression",
                torch.nn.ModuleList(
                    [
                        StyledConvBlock(512, 512, initial=True, fused=True),
                        StyledConvBlock(512, 512, upsample=True, fused=True),
                        StyledConvBlock(512, 256, upsample=True, fused=True),
                        StyledConvBlock(256, 128, upsample=True, fused=True),
                        StyledConvBlock(128, 64, upsample=True, fused=True),
                        StyledConvBlock(64, 32, upsample=True, fused=True),
                        StyledConvBlock(32, 16, upsample=True, fused=True),
                    ]
                ),
            )
            setattr(
                self.module.generator,
                "to_rgb",
                torch.nn.ModuleList(
                    [
                        EqualConv2d(512, 3, 1),
                        EqualConv2d(512, 3, 1),
                        EqualConv2d(256, 3, 1),
                        EqualConv2d(128, 3, 1),
                        EqualConv2d(64, 3, 1),
                        EqualConv2d(32, 3, 1),
                        EqualConv2d(16, 3, 1),
                    ]
                ),
            )

    def forward(self, latent, noise):
        return self.module(latent, noise=noise, step=self.step)
