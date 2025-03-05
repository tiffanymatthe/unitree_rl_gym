paths = [
    "logs/simple_bc/teacher_1_epochs_x_-1.0_1.0_y_-1.0_1.0_yaw_-1_1_heading_-3.14_3.14"
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

cmd_dfs = [read_csv(f"{path}/cmd_vel_data.csv") for path in paths]

fig, axs = plt.subplots(4,4, figsize=(20,20))
axs = axs.flatten()

for i in range(12):
    ax = axs[i]
    for path, df in zip(paths, dfs):
        ax.plot(df["Epoch"], df[f"Dof {i} Action Loss"], label=path)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Dof {i} Action Loss")
    ax.set_title(f"Dof {i} Action Loss Comparison")
    ax.legend()

for path, df in zip(paths, dfs):
    axs[-1].plot(df["Epoch"], df["Action Loss"], label=path)

# Ensure all plots have the same scales
all_y_values = [df[f"Dof {i} Action Loss"] for df in dfs for i in range(12)]
all_y_values = [df["Action Loss"] for df in dfs] + all_y_values
all_y_values_flat = [item for sublist in all_y_values for item in sublist]
y_min, y_max = min(all_y_values_flat), max(all_y_values_flat)

for ax in axs:
    ax.set_ylim(y_min, y_max)

fig1, axs1 = plt.subplots(4,1, figsize=(12,8))
axs1 = axs1.flatten()

headers = ["cmd_vel_x","cmd_vel_y","cmd_vel_yaw","cmd_heading"]
for i, header in enumerate(headers):
    for path, df in zip(paths, cmd_dfs):
        axs1[i].hist(df[header], label=path)
        axs1[i].set_title(header)

plt.show()
