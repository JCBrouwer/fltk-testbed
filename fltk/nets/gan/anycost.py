from .base import InferenceGenerator, import_from
from importlib import reload

with import_from("anycost-gan"):
    import models

    reload(models)

    from models.anycost_gan import Generator


class AnyCostGenerator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(Generator, size)
