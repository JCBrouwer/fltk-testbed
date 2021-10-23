# %%
import pandas as pd
import numpy as np

# %%
data = pd.read_csv("cpu_results.csv")
data = data.dropna()
print("benchmark")
print(data.groupby(["device", "model", "size", "batch_size"]).agg("ms / img").aggregate("mean").to_string())
#%%

data = pd.read_csv("lambda2_2.csv")
print(data.groupby(["device"]).size())
data = data[data.response_time != -1]
data.response_time = data.response_time / 1000
data.groupby(["device"]).size()

#%%
print("experiment")
table = (
    data.groupby(["device", "image_size", "num_imgs"])
    .agg("response_time")
    .aggregate(["min", "median", "mean", "max", "size"])
)
print(table)