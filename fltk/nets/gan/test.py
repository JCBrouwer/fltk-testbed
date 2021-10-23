import gc
import tracemalloc
from time import time

import cpuinfo
import numpy as np
import torch

from . import *

torch.backends.cudnn.benchmark = True

n_trials = 10

tracemalloc.start()
print("device", "model", "size", "batch_size", "ms / img", "std dev", "MB VRAM allocated", "MB RAM allocated", sep=",")
with torch.no_grad(), torch.inference_mode():
    for gs in [
        #        [
        #            AnyCostGenerator,
        #            Style1Generator,
        #            Style2Generator,
        #            Style2ADAGenerator,
        #            SWAGenerator,
        #            StyleMapGenerator,
        #        ],
        [MobileStyleGenerator],
    ]:
        for device in ["cpu"] + list(range(torch.cuda.device_count())):
            for size in [256, 512, 1024]:
                for generator in gs:
                    G = generator(size).to(device)

                    for batch_size in [1, 2, 4, 8] + (
                        [16, 24, 32] if device == 0 else ([10, 12, 14, 16] if device == 1 else [])
                    ):

                        # try:
                        try:
                            torch.cuda.synchronize()
                            output = G.forward(torch.randn(batch_size, 512, device=device), None)
                            output = output.cpu()
                            assert tuple(output.shape) == (batch_size, 3, size, size)

                            torch.cuda.reset_peak_memory_stats()
                            tracemalloc.reset_peak()

                            torch.cuda.synchronize()
                            times, outputs = [], []
                            for _ in range(n_trials):
                                t = time()
                                output = G.forward(torch.randn(batch_size, 512, device=device), None)
                                outputs.append(output.cpu())
                                times.append(time() - t)
                            time_ms = f"{1000 * np.mean(times) / batch_size:.3f}"
                            std_ms = f"{1000 * np.std(times) / batch_size:.3f}"

                        except Exception as e:
                            time_ms = -1
                            std_ms = -1

                        _, peak_cpu = tracemalloc.get_traced_memory()
                        peak_gpu = torch.cuda.max_memory_allocated(device) if not device == "cpu" else 0

                        print(
                            torch.cuda.get_device_name(device)
                            if device != "cpu"
                            else cpuinfo.get_cpu_info()["brand_raw"],
                            generator.__name__,
                            size,
                            batch_size,
                            time_ms,
                            std_ms,
                            round(peak_gpu / 1024 / 1024),
                            round(peak_cpu / 1024 / 1024),
                            sep=",",
                        )

                        gc.collect()
                        torch.cuda.empty_cache()

                        # except Exception as e:
                        #     print()
                        #     print("ERROR")
                        #     print(e)
                        #     print()

                    del G
                    gc.collect()
                    torch.cuda.empty_cache()
