import numpy as np
import matplotlib.pyplot as plt

n_runs = 20

ws = []

for run in range(n_runs):
    save_path = f'lasso_data/run_{run}/lasso_weights.npy'
    w = np.load(save_path)
    ws.append(w)

ws = np.array(ws)
mean = ws.mean(axis=0)
ws[ws == 0] = np.nan
mean[mean == 0] = np.nan

# plot average weights with error bars
fig = plt.figure(figsize=(20, 10))
plt.plot(mean, label='Mean', linestyle=None, marker='o', markersize=5)
plt.hlines(0, 0, len(mean), color='gray', alpha=0.6)
for run in range(n_runs):
    plt.plot(ws[run], linestyle=None, marker='.', markersize=2, alpha=0.4, color='gray')

plt.xlabel('[#] Output Neuron')
plt.ylabel('Weight')

plt.show()

