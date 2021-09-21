print("mobile")

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MobileStyleGAN.pytorch"))

from core.models.mobile_synthesis_network import MobileSynthesisNetwork

sys.path = sys.path[:-1]

from functools import partial

from .base import InferenceGenerator


class MobileStyleGenerator(InferenceGenerator):
    def __init__(self, size):
        channels_dict = {
            1024: [512, 512, 512, 512, 512, 256, 128, 64],
            512: [512, 512, 512, 512, 256, 128, 64],
            256: [512, 512, 512, 256, 128, 64],
        }
        super().__init__(partial(MobileSynthesisNetwork, channels=channels_dict[size]), 512)
