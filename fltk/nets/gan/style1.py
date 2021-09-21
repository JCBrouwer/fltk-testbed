print("style1")

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style-based-gan-pytorch"))

from model import Generator

sys.path = sys.path[:-1]

from .base import InferenceGenerator


class Style1Generator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(Generator, 512)
        if size == 512:
            setattr(self.module, "progression", self.module.progression[1:])
            setattr(self.module, "to_rgb", self.module.to_rgb[1:])
        elif size == 256:
            setattr(self.module, "progression", self.module.progression[2:])
            setattr(self.module, "to_rgb", self.module.to_rgb[2:])
