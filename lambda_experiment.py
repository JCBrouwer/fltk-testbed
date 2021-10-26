#%%
import re
from glob import glob
from os import error

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import Affine2D

# #%%
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
# } > lambda_results_improved.csv
# #%%
# data = pd.read_csv("lambda_results_improved.csv")
# #%%
# data.groupby(["schedule", "lambda", "stat"]).num.agg(["mean", "std"])
#%%

results = []
for csv in glob("lambda_results/*.csv") + glob("lambda_results/first_runs/*.csv"):
    run_data = pd.read_csv(csv)
    run_num_imgs = np.nansum(run_data.num_imgs)
    run_num_pix = np.nansum(run_data.num_imgs * run_data.image_size.pow(2))
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
    "unfinished",
    "std unfinished",
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
# print("".join([col.rjust(width) for col, width in zip(columns, widths)]))
df = []
for lambd in [10, 5, 4, 3, 2, 1]:
    for sched in ["random", "vram-aware", "improved"]:
        times, perimg, perpix, runs, restarts, completed, evicted, unfinished = [], [], [], 0, [], [], [], []
        for res in results:
            if res["arrival"] == lambd and res["schedule"] == sched:
                times.extend(list(res["response_times"]))
                perimg.extend(list(res["response_times"] / res["num_imgs"]))
                perpix.extend(list(1_000_000 * res["response_times"] / (res["num_imgs"] * res["image_sizes"] ** 2)))
                runs += 1
                restarts.append(res["restarts"])
                completed.append(res["completed"])
                evicted.append(res["evicted"])
                unfinished.append((res["response_times"] == -1).sum())
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
            np.median(unfinished),
            np.std(unfinished),
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
colors = ["tab:blue", "tab:orange", "tab:green"]
x = np.flip(np.sort(np.unique(df["lambda"])))
for title, ylabel, col, std in [
    ("jobs completed", "number of jobs completed", "completed", "std completed"),
    ("jobs restarted", "number of jobs restarted", "restarts", "std restarts"),
    ("jobs unfinished", "number of jobs unfinished", "unfinished", "std unfinished"),
    ("response time", "response time (sec)", "time", "std time"),
    # ("response time per image (sec)", "time / img", "std / img"),
    # ("response time per megapixel (sec)", "time / megapixel", "std / megapixel"),
]:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for s, sched in enumerate(["random", "vram-aware", "improved"]):
        dat = df[df["schedule"] == sched]

        y = dat[col]
        if sched == "random":
            y = np.concatenate((y, [np.nan, np.nan, np.nan]))
        elif sched == "vram-aware":
            y = np.concatenate((y, [np.nan, np.nan]))
        elif sched == "improved" and 15 in x:
            y = np.concatenate(([np.nan], y))

        ax.plot(x, y, label=sched, color=colors[s])
        ax.plot(x, y, "o", color=colors[s])

        err = dat[std]
        if sched == "random":
            err = np.concatenate((err, [np.nan, np.nan, np.nan]))
        elif sched == "vram-aware":
            err = np.concatenate((err, [np.nan, np.nan]))
        elif sched == "improved" and 15 in x:
            err = np.concatenate(([np.nan], err))
        lower = y - err
        upper = y + err
        ax.plot(x, lower, color=colors[s], alpha=0.1)
        ax.plot(x, upper, color=colors[s], alpha=0.1)
        ax.fill_between(x, lower, upper, alpha=0.2)

    ax.set_xlabel("arrival statistic")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin), ymax)
    ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    ax.invert_xaxis()
    ax.set_xticks(x)
    plt.savefig(f"{title}.pdf")
# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
scheds = ["random", "vram-aware", "improved"]
scheds = ["random", "vram-aware", "improved"]
lambdas = [10, 5, 4, 3, 2, 1]

gpu_results = {}
for csv in sorted(glob("lambda_results/*.csv"), key=lambda x: -int(x.split("/")[-1].split("_")[1])):
    run_data = pd.read_csv(csv)
    gpu_data = pd.read_csv(csv.replace(".csv", ".gpustats"))
    gpu_data.columns = [re.sub("\[.*\]", "", col).replace(" ", "") for col in gpu_data.columns]
    gpu_data["utilization.gpu"] = gpu_data["utilization.gpu"].apply(lambda x: int(x.replace(" %", "")))
    gpu_data["utilization.memory"] = gpu_data["utilization.memory"].apply(lambda x: int(x.replace(" %", "")))
    gpu_data["name"] = gpu_data["name"].apply(lambda x: x.replace(" NVIDIA GeForce ", ""))

    sched, lambd, seed = csv.split("/")[-1].split("_")

    if not (sched, lambd) in gpu_results:
        gpu_results[(sched, lambd)] = [gpu_data[gpu_data.name == "RTX 3090"], gpu_data[gpu_data.name == "GTX 1080 Ti"]]
    gpu_results[(sched, lambd)][0] = pd.concat((gpu_results[(sched, lambd)][0], gpu_data[gpu_data.name == "RTX 3090"]))
    gpu_results[(sched, lambd)][1] = pd.concat(
        (gpu_results[(sched, lambd)][1], gpu_data[gpu_data.name == "GTX 1080 Ti"])
    )

for (sched, lambd), (rtx3090, gtx1080ti) in gpu_results.items():
    sched_idx = scheds.index(sched)
    tform0o = Affine2D().translate(0.15 * sched_idx - 0.15, 0)
    tform0x = Affine2D().translate(0.15 * sched_idx - 0.075, 0)
    ax[0].errorbar(
        [lambd],
        [rtx3090["utilization.gpu"].mean()],
        yerr=[rtx3090["utilization.gpu"].std()],
        label=sched,
        color=colors[sched_idx],
        alpha=0.3,
        marker="",
        transform=tform0o + ax[0].transData,
    )
    ax[0].plot(
        [lambd],
        [rtx3090["utilization.gpu"].mean()],
        label=sched,
        color=colors[sched_idx],
        marker="o",
        transform=tform0o + ax[0].transData,
    )
    ax[0].errorbar(
        [lambd],
        [gtx1080ti["utilization.gpu"].mean()],
        yerr=[gtx1080ti["utilization.gpu"].std()],
        label=sched,
        color=colors[sched_idx],
        alpha=0.3,
        marker="",
        transform=tform0x + ax[0].transData,
    )
    ax[0].plot(
        [lambd],
        [gtx1080ti["utilization.gpu"].mean()],
        label=sched,
        color=colors[sched_idx],
        marker="x",
        transform=tform0x + ax[0].transData,
    )
    ax[0].set_ylim(0, 100)
    ax[0].set_xlabel("Arrival Statistic")
    ax[0].set_ylabel("GPU Utilization (%)")

    ax[1].errorbar(
        [lambd],
        [rtx3090["utilization.memory"].mean()],
        yerr=[rtx3090["utilization.memory"].std()],
        label=sched,
        color=colors[sched_idx],
        alpha=0.3,
        marker="",
        transform=tform0o + ax[1].transData,
    )
    ax[1].plot(
        [lambd],
        [rtx3090["utilization.memory"].mean()],
        label=sched,
        color=colors[sched_idx],
        marker="o",
        transform=tform0o + ax[1].transData,
    )
    ax[1].errorbar(
        [lambd],
        [gtx1080ti["utilization.memory"].mean()],
        yerr=[gtx1080ti["utilization.memory"].std()],
        label=sched,
        color=colors[sched_idx],
        alpha=0.3,
        marker="",
        transform=tform0x + ax[1].transData,
    )
    ax[1].plot(
        [lambd],
        [gtx1080ti["utilization.memory"].mean()],
        label=sched,
        color=colors[sched_idx],
        marker="x",
        transform=tform0x + ax[1].transData,
    )
    ax[1].set_ylim(0, 100)
    ax[1].set_xlabel("Arrival Statistic")
    ax[1].set_ylabel("VRAM Utilization (%)")
plt.legend(
    handles=[matplotlib.patches.Patch(color=colors[scheds.index(sched)], label=sched) for sched in scheds]
    + [
        matplotlib.lines.Line2D(
            [], [], color="black", marker=["o", "x"][c], linestyle="None", markersize=10, label=classifier
        )
        for c, classifier in enumerate(["RTX 3090", "GTX 1080 Ti"])
    ]
)
plt.suptitle("GPU/VRAM utilization under different schedules")
plt.tight_layout()
plt.savefig("gputil.pdf")
# %%
