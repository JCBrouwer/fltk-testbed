print("stylemap")

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "StyleMapGAN"))

from training.model import Generator

sys.path = sys.path[:-1]

from .base import InferenceGenerator

from functools import partial


class StyleMapGenerator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(
            partial(
                Generator,
                mapping_layer_num=8,
                style_dim=64,
                latent_spatial_size=8,
                lr_mul=0.01,
                channel_multiplier=2,
                normalize_mode="LayerNorm",
            ),
            size,
        )

    def forward(self, latent, noise):
        # StyleMapGAN doesn't use noise!
        return self.module(latent)