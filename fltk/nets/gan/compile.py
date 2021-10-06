from time import time

import torch

from . import *

for generator in [
    AnyCostGenerator,
    MobileStyleGenerator,
    Style1Generator,
    Style2Generator,
    StyleMapGenerator,
    SWAGenerator,
    Style2ADAGenerator,
]:
    print(f"compiling {generator.__name__} CUDA ops...".ljust(50), end="")
    t = time()
    exception = None
    try:
        generator(256).forward(torch.randn(1, 512, device="cpu"), None)
        generator(256).cuda().forward(torch.randn(1, 512, device="cuda"), None)
    except RuntimeError as e:
        exception = e
    print(f"took {time() - t:.2f} sec")
    if exception is not None:
        print()
        print(exception)
        print()
    torch.cuda.empty_cache()