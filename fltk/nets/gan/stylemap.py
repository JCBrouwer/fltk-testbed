from functools import partial

from .base import InferenceGenerator, import_from

with import_from("StyleMapGAN"):
    from training.model import Generator


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
