from .base import InferenceGenerator, import_from


class AnyCostGenerator(InferenceGenerator):
    def __init__(self, size):
        with import_from("anycost-gan"):
            from models.anycost_gan import Generator
        super().__init__(Generator, size)

    def forward(self, latent, noise):
        return self.module(latent[:, None], noise=noise)[0]
