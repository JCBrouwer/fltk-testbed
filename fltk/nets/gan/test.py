from time import time

import torch

from . import *

torch.backends.cudnn.benchmark = True

with torch.inference_mode():
    for generator in [
        AnyCostGenerator,
        Style1Generator,
        Style2Generator,
        StyleMapGenerator,
        MobileStyleGenerator,
        # SWAGenerator,
    ]:
        for size in [256, 512, 1024]:
            output, _ = generator(size).cuda().forward(torch.randn(8, 1, 512), None)
            assert tuple(output.shape) == (8, 3, size, size)

            t = time()
            for _ in range(20):
                output, _ = generator(size).forward(torch.randn(8, 1, 512), None)
            print(generator, size, (time() - t) / 10)
