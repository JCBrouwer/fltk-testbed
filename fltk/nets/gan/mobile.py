import torch

from .base import InferenceGenerator, import_from


class MobileStyleGenerator(InferenceGenerator):
    def __init__(self, size):
        with import_from("MobileStyleGAN.pytorch"):
            from core.models.mapping_network import MappingNetwork
            from core.models.mobile_synthesis_network import MobileSynthesisNetwork

        channels_dict = {
            1024: [512, 512, 512, 512, 512, 256, 128, 64],
            512: [512, 512, 512, 512, 256, 128, 64],
            256: [512, 512, 512, 256, 128, 64],
        }
        super().__init__(lambda _: None, 0)
        self.mapping = MappingNetwork(512, 8)
        self.synthesis = MobileSynthesisNetwork(512, channels=channels_dict[size])

    def forward(self, latent, noise):
        style = self.mapping(latent).unsqueeze(1).repeat(1, self.synthesis.wsize(), 1)
        return self.synthesis(style, noise=noise)["img"]
