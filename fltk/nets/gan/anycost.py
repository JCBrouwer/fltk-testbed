import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "anycost-gan"))

from models.anycost_gan import Generator

sys.path = sys.path[:-1]

from .base import InferenceGenerator


class AnyCostGenerator(InferenceGenerator):
    def __init__(self, size):
        super().__init__(Generator, size)