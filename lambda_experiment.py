#%%
from glob import glob
from os import error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#%%
# %%capture output
# %%bash
# . ~/.zshrc
# {
#     echo schedule,lambda,stat,num
#     find lambda_results/* | \
#         grep -v .log | grep -v .csv | \
#         while read f; do
#             echo $(echo ${f##*/} | cut -d_ -f1),$(echo $f | cut -d_ -f3),${f##*.},$(cat $f)
#     done
# } > lambda_results.csv
#%%
data = pd.read_csv("lambda_results.csv")
#%%
data.groupby(["schedule", "lambda", "stat"]).num.agg(["mean", "std"])
#%%

results = []
for csv in glob("lambda_results/*.csv"):
    run_data = pd.read_csv(csv)
    run_num_imgs = run_data.num_imgs.sum()
    run_num_pix = (run_data.num_imgs * run_data.image_size.pow(2)).sum()
    schedule, lambd, seed = csv.split("/")[-1].split("_")

    run_stats = {}
    for ext in ["completed", "restarts", "evicted"]:
        with open(csv.replace("csv", ext), "r") as file:
            run_stats[ext] = int(file.readline().strip())

    results.append(
        {
            "arrival": int(lambd),
            "schedule": schedule,
            "total_num_imgs": run_num_imgs,
            "total_num_pix": run_num_pix,
            "response_times": run_data.response_time.values,
            "num_imgs": run_data.num_imgs.values,
            "image_sizes": run_data.image_size.values,
            "network": run_data.network.values,
            "job_type": run_data.job_type.values,
            "device": run_data.device.values,
            "batch_size": run_data.batch_size.values,
            "restarts": run_stats["restarts"],
            "completed": run_stats["completed"],
            "evicted": run_stats["evicted"],
        }
    )
# %%
columns = [
    "schedule",
    "lambda",
    "trials",
    "failed",
    "std failed",
    "restarts",
    "std restarts",
    "completed",
    "std completed",
    "evicted",
    "std evicted",
    "time",
    "std time",
    "time / img",
    "std / img",
    "time / megapixel",
    "std / megapixel",
]
widths = [len(col) + 3 for col in columns]
print("".join([col.rjust(width) for col, width in zip(columns, widths)]))
df = []
for lambd in [15, 10, 5, 4, 3]:
    for sched in ["random", "vram-aware"]:
        times, perimg, perpix, runs, restarts, completed, evicted, failed = [], [], [], 0, [], [], [], []
        for res in results:
            if res["arrival"] == lambd and res["schedule"] == sched:
                times.extend(list(res["response_times"]))
                perimg.extend(list(res["response_times"] / res["num_imgs"]))
                perpix.extend(list(1_000_000 * res["response_times"] / (res["num_imgs"] * res["image_sizes"] ** 2)))
                runs += 1
                restarts.append(res["restarts"])
                completed.append(res["completed"])
                evicted.append(res["evicted"])
                failed.append((res["response_times"] == -1).sum())
        times, perimg, perpix = np.array(times), np.array(perimg), np.array(perpix)
        if len(times) == 0:
            continue
        times = times[times > 0] / 1000
        perimg = perimg[perimg > 0] / 1000
        perpix = perpix[perpix > 0] / 1000

        # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        # ax[0].hist(perpix, bins=100)
        # ax[1].hist(perimg, bins=100)
        # ax[1].set_title(f"{sched} {lambd}")
        # ax[2].hist(times, bins=100)
        # plt.tight_layout()

        vals = [
            sched,
            lambd,
            runs,
            np.median(failed),
            np.std(failed),
            np.median(restarts),
            np.std(restarts),
            np.median(completed),
            np.std(completed),
            np.median(evicted),
            np.std(evicted),
            np.median(times),
            np.std(times),
            np.median(perimg),
            np.std(perimg),
            np.median(perpix),
            np.std(perpix),
        ]
        df.append(vals)
        # print(
        #     "".join(
        #         [
        #             (f"{val}" if not isinstance(val, float) else f"{val:.1f}").rjust(width)
        #             for val, width in zip(vals, widths)
        #         ]
        #     )
        # )
df = pd.DataFrame(df, columns=columns)
df.to_csv("lambda_response_times.csv")
df
#%%
colors = ["tab:blue", "tab:orange"]
x = list(reversed(np.sort(np.unique(df["lambda"])).astype(str)))
for title, col, std in [
    ("jobs completed", "completed", "std completed"),
    ("jobs restarted", "restarts", "std restarts"),
    ("jobs failed", "failed", "std failed"),
    ("response time (sec)", "time", "std time"),
    # ("response time per image (sec)", "time / img", "std / img"),
    # ("response time per megapixel (sec)", "time / megapixel", "std / megapixel"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for s, sched in enumerate(["random", "vram-aware"]):
        dat = df[df["schedule"] == sched]

        y = dat[col]
        if len(y) < len(x):
            y = np.concatenate((y, [np.nan]))

        ax.plot(x, y, label=sched, color=colors[s])
        ax.plot(x, y, "o", color=colors[s])

        if std is not None:
            err = dat[std]
            if len(err) < len(y):
                err = np.concatenate((err, [np.nan]))
            lower = y - err
            upper = y + err
            ax.plot(x, lower, color=colors[s], alpha=0.1)
            ax.plot(x, upper, color=colors[s], alpha=0.1)
            ax.fill_between(x, lower, upper, alpha=0.2)

    ax.set_xlabel("arrival statistic")
    ax.set_ylabel(col)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin), ymax)
    ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.pdf")
# %%
