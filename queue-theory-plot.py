#%%
import pandas as pd
import matplotlib.pyplot as plt

sizes = [256, 512, 1024]
st_gtx_1080 = [16, 25, 40]
st_rtx_3090 = [9, 17, 27]
probs = [0.25, 0.5, 0.25]

# Compute mean service times
mean_st_1080 = sum([rt * prob for rt, prob in zip(st_gtx_1080, probs)])
mean_st_3090 = sum([rt * prob for rt, prob in zip(st_rtx_3090, probs)])

mean_st_1080 = 10
mean_st_3090 = 6.6

mu_1080 = 1 / mean_st_1080
mu_3090 = 1 / mean_st_3090

# Compute optimal split
p = mu_1080 / (mu_1080 + mu_3090)
print(mu_1080)
print(mu_3090)
print(p)

import numpy as np

betas = 10 ** np.linspace(1, 0.566, 100)
print(betas)
lambdas = [1 / b for b in betas]
print(lambdas)
rtp = [p * 1 / (mu_1080 - p * a) + (1 - p) * 1 / (mu_3090 - (1 - p) * a) for a in lambdas]

p = 0.5
rt50 = np.array([p * 1 / (mu_1080 - p * a) + (1 - p) * 1 / (mu_3090 - (1 - p) * a) for a in lambdas])
rt50[90:] = -1

# plotting
df = pd.DataFrame(
    {
        "lambda": lambdas,
        "beta": betas,
        "rho": [l / (mu_1080 + mu_3090) for l in lambdas],
        "equal split": rt50,
        "optimal split": rtp,
    }
)
print(df)

#%%
plt.plot(df["beta"], df["equal split"], color="tab:purple", label="equal split")
plt.plot(df["beta"], df["optimal split"], color="tab:blue", label="optimal split")
plt.ylim(0, 150)
plt.ylabel("theoretical response time")
plt.xticks([10, 9, 8, 7, 6, 5, 4, 3])
plt.gca().invert_xaxis()
plt.legend()
plt.savefig("queueing-theory-asymptote.pdf")
# %%