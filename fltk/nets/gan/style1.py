from functools import partial

from .base import InferenceGenerator, import_from

with import_from("style-based-gan-pytorch"):
    from model import Generator


class Style1Generator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(Generator, 512)
        if size == 512:
            setattr(self.module, "progression", self.module.progression[1:])
            setattr(self.module, "to_rgb", self.module.to_rgb[1:])
        elif size == 256:
            setattr(self.module, "progression", self.module.progression[2:])
            setattr(self.module, "to_rgb", self.module.to_rgb[2:])
