import gc
from time import time

import numpy as np
import torch

from . import *

torch.backends.cudnn.benchmark = True

print(
    "device".ljust(30),
    "model".ljust(30),
    "size".rjust(5),
    "batch_size".rjust(20),
    "ms / img (sync)".rjust(25),
    "std dev (sync)".rjust(25),
    "ms / img (no sync)".rjust(25),
    "std dev (no sync)".rjust(25),
)
n_trials = 10
with torch.no_grad(), torch.inference_mode():
    for device in [0, 1]:
        torch.cuda.set_device(device)
        for generator in [
            AnyCostGenerator,
            Style1Generator,
            Style2Generator,
            Style2ADAGenerator,
            SWAGenerator,
            StyleMapGenerator,
            MobileStyleGenerator,
        ]:
            for batch_size in [1, 2, 4, 8] + ([16, 24, 32] if device == 0 else [10, 12, 14]):
                for size in [256, 512, 1024]:

                    G = generator(size).cuda()

                    try:
                        torch.cuda.synchronize()
                        output = G.forward(torch.randn(batch_size, 512, device="cuda"), None)
                        output = output.cpu()
                        assert tuple(output.shape) == (batch_size, 3, size, size)

                        torch.cuda.synchronize()
                        times = []
                        for _ in range(n_trials):
                            t = time()
                            output = G.forward(torch.randn(batch_size, 512, device="cuda"), None)
                            output = output.cpu()
                            torch.cuda.synchronize()
                            times.append(time() - t)
                        time_sync = f"{1000 * np.mean(times) / batch_size:.3f}"
                        std_sync = f"{1000 * np.std(times) / batch_size:.3f}"

                        torch.cuda.synchronize()
                        times = []
                        for _ in range(n_trials):
                            t = time()
                            output = G.forward(torch.randn(batch_size, 512, device="cuda"), None)
                            times.append(time() - t)
                        time_nosync = f"{1000 * np.mean(times) / batch_size:.3f}"
                        std_nosync = f"{1000 * np.std(times) / batch_size:.3f}"

                        print(
                            ("RTX 3090" if device == 0 else "GTX 1080 Ti").ljust(30),
                            generator.__name__.ljust(30),
                            f"{size}".rjust(5),
                            f"{batch_size}".rjust(20),
                            time_sync.rjust(25),
                            std_sync.rjust(25),
                            time_nosync.rjust(25),
                            std_nosync.rjust(25),
                        )

                    except Exception as e:
                        print(
                            ("RTX 3090" if device == 0 else "GTX 1080 Ti").ljust(30),
                            generator.__name__.ljust(30),
                            f"{size}".rjust(5),
                            f"{batch_size}".rjust(20),
                            "CUDA out of memory".rjust(103),
                        )

                    del G
                    gc.collect()
                    torch.cuda.empty_cache()
