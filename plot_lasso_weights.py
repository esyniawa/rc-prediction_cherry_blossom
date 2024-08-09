import numpy as np
import matplotlib.pyplot as plt

from utils import load_lasso_weights

n_runs = 50
mean, ws = load_lasso_weights(save_path='lasso_data', n_runs=n_runs)

ws[ws == 0] = np.nan
mean[mean == 0.] = np.nan

# plot average weights with error bars
fig = plt.figure(figsize=(20, 10))
plt.plot(mean, label='Mean', linestyle=None, marker='o', markersize=5)
plt.hlines(0, 0, len(mean), color='gray', alpha=0.6)
for run in range(n_runs):
    plt.plot(ws[run], linestyle=None, marker='.', markersize=2, alpha=0.4, color='gray')

plt.xlabel('[#] Output Neuron')
plt.ylabel('Weight')

plt.show()

