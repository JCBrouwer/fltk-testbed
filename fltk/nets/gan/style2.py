from functools import partial
from importlib import reload

from .base import InferenceGenerator, import_from


class Style2Generator(InferenceGenerator):
    def __init__(self, size):
        with import_from("stylegan2-pytorch"):
            import model

            reload(model)
            from model import Generator

        super().__init__(partial(Generator, style_dim=512, n_mlp=8), size)

    def forward(self, latent, noise):
        return self.module([latent], noise=noise)[0]
