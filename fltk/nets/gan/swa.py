print("swa")

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stylegan2-pytorch"))

from swagan import Generator

sys.path = sys.path[:-1]

from .base import InferenceGenerator

from functools import partial


class SWAGenerator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(partial(Generator, style_dim=512, n_mlp=8), size)