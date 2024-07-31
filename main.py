# Main part for sakura blossom date prediction with ESN and ANNarchy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from input_reservoir_data import load_input_data
from network.reservoir import N

n_runs = 50
do_plot = True

# If figures need to be stored
if do_plot and not os.path.exists('figures'):
    os.mkdir('figures')
    print('New directory for the figures created!')


results_mse = []

###################################################### Data ##############################################################
# load training data
df = load_input_data()

# Scaling output data
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
Y = scaler.fit_transform(df['Output'].values.reshape(-1, 1))
X = np.array(df['Input'].tolist())

for run in range(n_runs):
    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

    ######################################################## Training part ###########################################
    # Train output weights using Lasso regression
    lasso = linear_model.Lasso(alpha=0.001, max_iter=100_000)
    lasso.fit(X_train, Y_train)

    # Make predictions
    Y_pred = lasso.predict(X_test)

    # Evaluate predictions
    mse_scaled = mean_squared_error(Y_test, Y_pred)
    # print(f'Mean Squared Error: {mse}')

    pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
    test = scaler.inverse_transform(Y_test.reshape(-1, 1))
    mse_unscaled = mean_squared_error(test, pred)
    # print(f'Mean Squared Error Converted: {mse}')

    results_mse.append((mse_scaled, mse_unscaled))

    if do_plot:
        fig = plt.figure()
        plt.plot(test, label='True', color='b')
        plt.plot(pred, label='Predicted', alpha=0.8, marker=".", markersize=2,
                 linestyle='None', color='r')
        plt.plot(pred-test, label='Predicted', alpha=0.8, marker="x", markersize=2,
                 linestyle='None', color='k')

        plt.legend()
        plt.title(f'Sakura Blossom Date Prediction N={N}')
        plt.savefig(f'figures/{run}', bbox_inches='tight')
        plt.close(fig)

np.save('results.npy', np.array(results_mse))

print(f'MSE = {np.mean(np.array(results_mse), axis=0)} +/- {np.std(np.array(results_mse), axis=0)}')