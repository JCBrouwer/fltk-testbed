from functools import partial
from importlib import reload

from .base import InferenceGenerator, import_from

with import_from("stylegan2-pytorch"):
    import model

    reload(model)
    from model import Generator


class Style2Generator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(partial(Generator, style_dim=512, n_mlp=8), size)
