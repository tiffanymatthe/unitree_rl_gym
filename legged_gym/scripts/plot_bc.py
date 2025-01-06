PATH = "logs/behavior_cloning/distill"
PATH1 = "logs/behavior_cloning/dagger"

import pandas
import matplotlib.pyplot as plt

f = open(f"{PATH}/bc_results.csv", "r")
df = pandas.read_csv(f)

f1 = open(f"{PATH1}/bc_results.csv", "r")
df1 = pandas.read_csv(f1)

plt.plot(df["Epoch"], df["Action Loss"], label=PATH)
plt.plot(df1["Epoch"], df1["Action Loss"], label=PATH1)

plt.xlabel("Epoch")
plt.ylabel("Value Loss")
plt.title("Value Loss Comparison")
plt.legend()
plt.show()