from functools import partial
from importlib import reload

from .base import InferenceGenerator, import_from


class StyleMapGenerator(InferenceGenerator):
    def __init__(self, size):
        with import_from("StyleMapGAN"):
            import training

            reload(training)
            from training.model import Generator

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
        return self.module(latent[:, :64])[0]
