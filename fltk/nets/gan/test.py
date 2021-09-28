from time import time

import torch

from . import *

torch.backends.cudnn.benchmark = True

batch_size = 4
n_trials = 20
print(
    "model".ljust(30),
    "size".rjust(10),
    "exec time (ms)".rjust(30),
)
with torch.no_grad(), torch.inference_mode():
    for generator in [
        AnyCostGenerator,
        Style1Generator,
        Style2Generator,
        SWAGenerator,
        StyleMapGenerator,
        MobileStyleGenerator,
    ]:
        for size in [256, 512, 1024]:
            # print("\n\n\n")
            # print(size)
            for _ in range(5):
                output = generator(size).cuda().forward(torch.randn(batch_size, 512, device="cuda"), None)
            assert tuple(output.shape) == (batch_size, 3, size, size)

            t = time()
            for _ in range(n_trials):
                # t2 = time()
                output = generator(size).cuda().forward(torch.randn(batch_size, 512, device="cuda"), None)
                output = output.cpu()
                # print("it:", time() - t2)
            print(
                generator.__name__.ljust(30),
                f"{size}".rjust(10),
                f"{1000 * (time() - t) / batch_size / n_trials:.3f}".rjust(30),
            )
