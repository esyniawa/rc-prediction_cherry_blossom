import sys

import ANNarchy as ann
import pandas as pd
import numpy as np
import os

from network.reservoir import *
from sakura_data import load_sakura_data


def create_input_data(file_path: str = 'data/lasso_data.parquet',
                      mean_pop: int | None = None,
                      period: int = 1,
                      tau: float = 10.,
                      g: float = 1.) -> pd.DataFrame:

    # Data from sakura dataset
    df = load_sakura_data().dropna()

    # Create the dataframe
    lasso_df = pd.DataFrame()
    lasso_df['Output'] = df['Blossom']
    lasso_df['Input'] = None

    # Create a monitor to record the evolution of firing rates during simulation
    monitor = ann.Monitor(reservoir_pop, 'r', period=period)

    def run_reservoir(inputs: pd.DataFrame,
                      mean_pop=mean_pop,
                      period=period):

        for i, row in inputs.iterrows():
            len_temps = row['Temps'].size

            @ann.every(period=period)
            def set_inputs(n):
                # Set inputs to the network
                input_pop[0].baseline = row['Temps'][n]  # Temps as dynamic input
                input_pop[1].baseline = row['Lat']  # Lat
                input_pop[2].baseline = row['Lng']  # Lng

            # simulate
            try:
                ann.simulate(period * len_temps, callbacks=True)
            except:
                print(i, row['Temps'].size, row)

            if mean_pop is not None:
                state = monitor.get(variables='r', keep=False)[-mean_pop:]
                state = np.mean(state, axis=0)
            else:
                state = monitor.get(variables='r', keep=False)[-1]

            lasso_df.at[i, 'Input'] = state
            ann.reset(monitors=True, populations=True)

    # Compile network
    ann.compile(f'annarchy/mean_pop{mean_pop}_period{period}_g{g}_tau{tau}/', clean=True)

    # set parameters
    reservoir_pop.g = g
    reservoir_pop.tau = tau

    # Create input data
    run_reservoir(df)

    # Save the dataframe
    lasso_df.to_parquet(file_path, index=False)

    return lasso_df


def load_input_data(file_path: str = 'data/lasso_data.parquet') -> pd.DataFrame:
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
    else:
        df = create_input_data(file_path=file_path)
    return df


if __name__ == '__main__':

    period, mean_pop = int(sys.argv[1]), int(sys.argv[2])

    create_input_data(file_path=f'data/lasso_data_period{period}_mean{mean_pop}.parquet',
                      mean_pop=mean_pop,
                      period=period)
