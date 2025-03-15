num_data = 1000
import numpy as np

action_timesteps = []
decimations = []
decimation_timestep = 1 / 200

for _ in range(num_data):
    lammy = np.random.uniform(125, 1000)
    beta = 1 / lammy
    action_timestep = 0.02 + np.random.exponential(scale=beta)
    action_timesteps.append(action_timestep)
    decimations.append(round(action_timestep / decimation_timestep))

import matplotlib.pyplot as plt

# plt.hist(action_timesteps, bins=100)
# plt.title("Action Timesteps Distribution")
# plt.xlabel("Action Timestep (s)")
plt.hist(decimations, bins=100)
plt.title("Num Decimations Distribution")
plt.xlabel("Number of decimations, each decimation is 5ms")
plt.show()