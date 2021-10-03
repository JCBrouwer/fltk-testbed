import pandas as pd

data = pd.read_csv("results.csv")
data = data.dropna()
print(data)

print(data.groupby(["device", "size"]).agg("ms / img (no sync)").mean())
# print(data.groupby(["device", "batch_size", "size"]).agg("ms / img (no sync)").std())
