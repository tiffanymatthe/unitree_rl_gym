paths = [
    # "logs/behavior_cloning/distill",
    # order: function for stochastic action for teacher epochs, batch size
    # "logs/behavior_cloning/walking_dagger_1_teacher_100k_batch", # act(), 100k
    "logs/behavior_cloning/walking_dagger_multi_task",
    # "logs/behavior_cloning/walking_dagger_1_teacher_inference_50k_batch", # act_inference(), 50k
    # "logs/behavior_cloning/walking_dagger_1_teacher_inference", # act_inference(), 20k
    # "logs/behavior_cloning/walking_dagger_1_teacher_2", # act(), 50k
    # "logs/behavior_cloning/walking_dagger_1_teacher" # act(), 20k
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
    plt.plot(df["Epoch"], df["Action Loss"], label=path + " action loss")
    plt.plot(df["Epoch"], df["Action Loss Lin Vel"], label=path + " action loss lin vel")

plt.xlabel("Epoch")
plt.ylabel("Action Loss")
plt.title("Action Loss Comparison")
plt.legend()
plt.show()
