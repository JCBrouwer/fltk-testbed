#%%
import pandas as pd
#%%
%%capture output
%%bash
. ~/.zshrc
{ echo schedule,lambda,stat,num
find lambda_results/* | \
    grep -v .log | \
    while read f; do
        echo $(echo ${f##*/} | cut -d_ -f1),$(echo $f | cut -d_ -f3),${f##*.},$(cat $f)
done } > lambda_results.csv
#%%
data = pd.read_csv("lambda_results.csv")
#%%
data.groupby(["schedule", "lambda", "stat"]).num.agg(["mean","std"])