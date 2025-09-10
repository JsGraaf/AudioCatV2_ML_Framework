import glob
import os
import pprint

import pandas as pd

pp = pprint.PrettyPrinter(indent=4)

results = {}
for f in glob.glob("Experiments/*/Training_Results.csv"):
    exp_name = f.split("/")[1]

    # Load the DF
    df = pd.read_csv(f)
    results[exp_name] = df["loss"].mean()

sorted_scores = dict(sorted(results.items(), key=lambda item: item[1], reverse=False))
print(sorted_scores)
