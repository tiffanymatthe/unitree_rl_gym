paths = [
    "logs/behavior_cloning/distill",
    "logs/behavior_cloning/dagger",
    "logs/behavior_cloning/dagger_10",
    "logs/behavior_cloning/dagger_100"
]

import pandas
import matplotlib.pyplot as plt

dfs = []
for path in paths:
    f = open(f"{path}/bc_results.csv", "r")
    df = pandas.read_csv(f)
    dfs.append(df)

for path, df in zip(paths, dfs):
    plt.plot(df["Epoch"], df["Action Loss"], label=path)

plt.xlabel("Epoch")
plt.ylabel("Value Loss")
plt.title("Value Loss Comparison")
plt.legend()
plt.show()