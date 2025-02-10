paths = [
    # "logs/behavior_cloning/distill",
    "logs/behavior_cloning/dagger",
    "logs/behavior_cloning/walking_dagger_1_teacher"
    # "logs/behavior_cloning/dagger_10",
    # "logs/behavior_cloning/dagger_100"
]

import matplotlib.pyplot as plt

def read_csv(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    headers = lines[0].strip().split(",")
    data = {header: [] for header in headers}
    
    for line in lines[1:]:
        values = line.strip().split(",")
        for header, value in zip(headers, values):
            data[header].append(float(value))
    
    return data

dfs = [read_csv(f"{path}/bc_results.csv") for path in paths]

for path, df in zip(paths, dfs):
    plt.plot(df["Epoch"], df["Action Loss"], label=path)

plt.xlabel("Epoch")
plt.ylabel("Action Loss")
plt.title("Action Loss Comparison")
plt.legend()
plt.show()
