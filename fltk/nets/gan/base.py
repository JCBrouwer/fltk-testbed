import importlib
import os
import pkgutil
import sys

import torch


class import_from:
    def __init__(self, submodule_name):
        self.submodule = os.path.join(os.path.dirname(os.path.abspath(__file__)), submodule_name)
        sys.path.append(self.submodule)

        # reload all modules/packages to avoid importing wrong files that happen to have the same name in previous paths
        # for _, name, _ in pkgutil.walk_packages(path=[self.submodule]):
        #     importlib.reload(importlib.import_module(name))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        sys.path = [path for path in sys.path if path != self.submodule]
        return False


class InferenceGenerator(torch.nn.Module):
    def __init__(self, module, size):
        super().__init__()
        self.module = module(size)

    def forward(self, latent, noise):
        return self.module(latent, noise=noise)