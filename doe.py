#%%
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyDOE2 import gsd as generalized_subset_design
from tqdm import tqdm

from fltk.nets.gan import *
from fltk.nets.gan.util import gaussian_filter

torch.backends.cudnn.benchmark = True
#%%
repetitions = 10
reduction = 4

batch_sizes = [1, 2, 4, 8, 16]
models = [
    SWAGenerator,
    AnyCostGenerator,
    Style1Generator,
    Style2Generator,
    # MobileStyleGenerator,
    Style2ADAGenerator,
]
image_sizes = [256, 512, 1024]
job_types = ["random", "interpolation"]
num_imgs = [100, 200, 400, 800]
gpus = ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce GTX 1080 Ti"]

# %%
levels = [
    len(batch_sizes),
    len(models),
    len(image_sizes),
    len(job_types),
    len(num_imgs),
]
expected_mean = 10
print(f"Expected time: ~{np.prod(levels) * repetitions * expected_mean / 60 / 60 / reduction} hours")
#%%
configs = generalized_subset_design(levels, reduction=reduction)
#%%
# times = pd.read_csv("allresults.csv")
# def run_experiment(batch_size, model, image_size, job_type, num_img, gpu):
#     time = float(
#         times.loc[
#             (times.device == gpu)
#             & (times.model == model)
#             & (times["size"] == image_size)
#             & (times.batch_size == batch_size),
#             "ms / img",
#         ]
#     )
#     std = float(
#         times.loc[
#             (times.device == gpu)
#             & (times.model == model)
#             & (times["size"] == image_size)
#             & (times.batch_size == batch_size),
#             "std dev",
#         ]
#     )
#     time = np.clip(std * np.random.randn() + time, times["ms / img"].min(), times["ms / img"].max())
#     time *= num_img * (1.5 if job_type == "interpolation" else 1)
#     return time / 1000
#%%
def run_experiment(batch_size, model, image_size, job_type, num_img, gpu):
    device = torch.device("cuda:0" if "3090" in gpu else "cuda:1")

    G = model(image_size).to(device)

    for _ in range(3):  # warm up cudnn benchmark
        G.forward(torch.randn((batch_size, 512), device=device), None)

    t = time()

    if job_type == "random":
        latents = torch.randn((num_img, 512), device=device)
    elif job_type == "interpolation":
        latents = torch.randn((num_img, 512), device=device)
        latents = gaussian_filter(latents, 20)

    for idx in range(0, num_img, batch_size):
        output = G.forward(latents[idx : idx + batch_size], None)
        output = output.cpu()

    elapsed = time() - t

    del G, latents

    return elapsed


# %%
results = []
with torch.inference_mode(), tqdm(total=len(gpus) * repetitions * len(configs), smoothing=0) as pbar:
    for b, m, i, j, n in configs:
        for gpu in gpus:
            try:
                for reps in range(repetitions):
                    exec_time = run_experiment(
                        batch_sizes[b], models[m], image_sizes[i], job_types[j], num_imgs[n], gpu
                    )
                    results.append(
                        [batch_sizes[b], models[m].__name__, image_sizes[i], job_types[j], num_imgs[n], gpu, exec_time]
                    )
                    pbar.update(1)
            except Exception as e:
                print()
                print("ERROR!")
                print(e)
                print()
results = pd.DataFrame(results, columns=["batch_size", "model", "image_size", "job_type", "num_img", "gpu", "time"])
#%%
results.to_csv("measured_times.csv")
#%%
results = pd.read_csv("measured_times.csv")
print("\nMeasured times")
results
#%%
print(
    f"min {results.time.min():.2f}s    median {np.median(results.time):.2f}s    mean {results.time.mean():.2f}s    max {results.time.max():.2f}s"
)
#%%
plt.subplots(1, 1, figsize=(8, 6))
plt.hist(results.time, bins=100)
plt.tight_layout()
plt.savefig("time-hist.pdf")
#%%
time_lm = sm.formula.ols(
    "time ~ C(model) + batch_size + image_size*image_size + C(job_type) + num_img + C(gpu)", data=results
).fit()
aov = sm.stats.anova_lm(time_lm, typ=2)
print("\nANOVA")
print(aov)
aov.to_csv("anova.csv")
#%%
print("\nLinear model parameters")
print(time_lm.params)
#%%
print("\nEffect sizes")
total = aov["sum_sq"].sum()
for var, (sum_sq, df, f, p) in list(aov.iterrows()):
    print(
        var.replace("C(", "").replace(")", "").ljust(12),
        f"{sum_sq / total * 100:.2f}%",
    )
#%%