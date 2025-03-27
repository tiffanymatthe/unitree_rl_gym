import pickle
import matplotlib.pyplot as plt

[axs, axs1, axs2] = pickle.load(open("plot.pickle", "rb"))
plt.show()