# Main part for sakura blossom date prediction with ESN and ANNarchy

import numpy as np
import pandas as pd

np.random.RandomState(1)

import matplotlib.pyplot as plt
from sakura_data import load_sakura_data, create_sakura_data
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import os

from network.reservoir import *
from monitoring import PopMonitor

n_runs = 20
do_plot = True
save_reservoir_states = True

# If figures need to be stored
if do_plot and not os.path.exists('figures'):
    os.mkdir('figures')
    print('New directory for the figures created!')

ann.compile(clean=True)

# Create a monitor to record the evolution of firing rates during simulation
monitor = ann.Monitor(reservoir_pop, 'r', period=1.0)

results_mse = []

###################################################### Data ##############################################################
# Data from sakura dataset
df = load_sakura_data().dropna()

input_df, output_df = df[['Temps', 'Lat', 'Lng']], df['Blossom']

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scaler.fit_transform(output_df.values.reshape(-1, 1))

for run in range(n_runs):
    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(input_df, output_df, test_size=0.3, shuffle=True)

    ######################################################## Training part ###########################################
    # Function to run the reservoir and collect states
    def run_reservoir(inputs: pd.DataFrame,
                      mean_pop: int | None = 30,
                      period: int = 1):
        states = []
        for i, row in inputs.iterrows():

            len_temps = len(row['Temps'])

            @ann.every(period=period)
            def set_inputs(n):
                # Set inputs to the network
                inp[0].baseline = row['Temps'][n]  # Temps as dynamic input
                inp[1].baseline = row['Lat']  # Lat
                inp[2].baseline = row['Lng']  # Lng

            # simulate
            try:
                ann.simulate(period * len_temps, callbacks=True)
            except:
                print(row['City'], row['Session'])

            if mean_pop is not None:
                state = monitor.get(variables='r', keep=False)[-mean_pop:]
                state = np.mean(state, axis=0)
            else:
                state = monitor.get(variables='r', keep=False)[-1]

            states.append(state)
            ann.reset()

        return np.array(states)


    # Run the reservoir on the training data
    print(f'Training Run {run}.')
    reservoir_states_train = run_reservoir(X_train)

    if save_reservoir_states:
        np.save(f'training_states_run{run}.npy', reservoir_states_train)

    # Train output weights using Lasso regression
    lasso = linear_model.Lasso(alpha=0.001, max_iter=100_000)
    lasso.fit(reservoir_states_train, Y_train)

    # Run the reservoir on the testing data
    print(f'Testing Run {run}.')
    reservoir_states_test = run_reservoir(X_test)

    # Make predictions
    Y_pred = lasso.predict(reservoir_states_test)

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
        plt.legend()
        plt.title(f'Sakura Blossom Date Prediction N={N}')
        plt.savefig(f'figures/{run}', bbox_inches='tight')
        plt.close(fig)


np.save('results.npy', np.array(results_mse))

