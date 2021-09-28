from functools import partial
from importlib import reload

from .base import InferenceGenerator, import_from

with import_from("MobileStyleGAN.pytorch"):
    import core

    reload(core)
    from core.models.mobile_synthesis_network import MobileSynthesisNetwork


class MobileStyleGenerator(InferenceGenerator):
    def __init__(self, size):
        channels_dict = {
            1024: [512, 512, 512, 512, 512, 256, 128, 64],
            512: [512, 512, 512, 512, 256, 128, 64],
            256: [512, 512, 512, 256, 128, 64],
        }
        super().__init__(partial(MobileSynthesisNetwork, channels=channels_dict[size]), 512)
