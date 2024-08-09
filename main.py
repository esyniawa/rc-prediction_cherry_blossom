# Main part for sakura blossom date prediction with ESN and ANNarchy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from input_dynamic_data import load_input_data
from network.reservoir import N

from utils import safe_save, find_largest_factors

n_runs = 20
do_plot = True

###################################################### Data ##############################################################
# load training data
df = load_input_data()

# Scaling output data
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
Y = scaler.fit_transform(df['Output'].values.reshape(-1, 1))
X = np.array(df['Input'].tolist())

mse = []

for run in range(n_runs):
    save_path = f'lasso_data/run_{run}/'

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

    # save
    safe_save(save_path + 'lasso_weights.npy', lasso.coef_)
    np.savetxt(save_path + 'lasso_mse.csv', np.array([mse_unscaled, mse_scaled]), delimiter=',')
    mse.append([mse_unscaled, mse_scaled])

    if do_plot:

        # Plot predictions and true values
        fig = plt.figure()
        plt.plot(test, label='True', color='b')
        plt.plot(pred, label='Predicted', alpha=0.8, marker=".", markersize=2,
                 linestyle='None', color='r')
        plt.plot(pred-test, label='Predicted', alpha=0.8, marker="x", markersize=2,
                 linestyle='None', color='k')

        plt.legend()
        plt.title(f'Sakura Blossom Date Prediction N={N}')
        plt.savefig(save_path + 'prediction.pdf', bbox_inches='tight')
        plt.close(fig)

        # Plot output weights
        w = lasso.coef_
        fig = plt.figure()
        plt.plot(w, linestyle=None, marker=".", markersize=2)
        plt.title(f'Output Weights Lasso N={N}')
        plt.savefig(save_path + 'weight_matrix.pdf', bbox_inches='tight')
        plt.close(fig)

print(f'MSE_unscaled = {np.mean(np.array(mse), axis=0)[0]:.3f} +/- {np.std(np.array(mse), axis=0)[0]:.3f}')
print(f'MSE_scaled = {np.mean(np.array(mse), axis=0)[1]:.3f} +/- {np.std(np.array(mse), axis=0)[1]:.3f}')