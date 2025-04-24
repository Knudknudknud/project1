import matplotlib.pyplot as plt
import numpy as np
import Algorithms_for_presentation as alg


dataset = alg.load_data()
dataset = [(x, y, int(label)) for x, y, label in dataset]
dataset = np.array(dataset)

plt.clf()
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:,2], label='Data Points')
plt.title("Chosen dataset")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(loc='best', fontsize='small', markerscale=0.5)
plt.savefig("chosen_plot.png")