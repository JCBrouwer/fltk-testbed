#%%
import pandas as pd
import numpy as np

data = pd.read_csv("lambda2_2.csv")
print(data.groupby(["device"]).size())
data = data[data.response_time != -1]
data.response_time = data.response_time / 1000
data.groupby(["device"]).size()
#%%
table = (
    data.groupby(["device", "image_size", "num_imgs"])
    .agg("response_time")
    .aggregate(["min", "median", "mean", "max", "size"])
)
print(table)
# %%
data = pd.read_csv("results.csv")
data = data.dropna()
print(data)
print(data.groupby(["device", "size"]).agg("ms / img (no sync)").mean())
# print(data.groupby(["device", "batch_size", "size"]).agg("ms / img (no sync)").std())