from importlib import reload

from .base import InferenceGenerator, import_from


class Style2ADAGenerator(InferenceGenerator):
    def __init__(self, size):
        with import_from("stylegan2-ada-pytorch"):
            import training

            reload(training)
            from training.networks import Generator

        super().__init__(
            lambda size: Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=size, img_channels=3), size
        )

    def forward(self, latent, noise):
        return self.module(latent, c=None)
