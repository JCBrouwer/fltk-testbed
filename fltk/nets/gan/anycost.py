from .base import InferenceGenerator, import_from

with import_from("anycost-gan"):
    from models.anycost_gan import Generator


class AnyCostGenerator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(Generator, size)
