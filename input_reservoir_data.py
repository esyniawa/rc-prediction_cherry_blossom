import ANNarchy as ann
import pandas as pd
import numpy as np
import os


def create_input_data(file_path: str = 'data/lasso_data.parquet') -> pd.DataFrame:
    from network.reservoir import *
    from sakura_data import load_sakura_data

    # Data from sakura dataset
    df = load_sakura_data().dropna()

    # Create the dataframe
    lasso_df = pd.DataFrame()
    lasso_df['Output'] = df['Blossom']
    lasso_df['Input'] = np.nan

    # Create a monitor to record the evolution of firing rates during simulation
    monitor = ann.Monitor(reservoir_pop, 'r', period=1.0)

    def run_reservoir(inputs: pd.DataFrame,
                      mean_pop: int | None = 30,
                      period: int = 1):

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
    ann.compile(clean=True)

    # Create input data
    run_reservoir(df, mean_pop=20, period=1)

    # Save the dataframe
    lasso_df.to_parquet(file_path, index=False)

    return lasso_df


def load_input_data(file_path: str = 'data/lasso_data.parquet') -> pd.DataFrame:
    if os.path.isfile(file_path):
        df = pd.read_parquet(file_path)
    else:
        df = create_input_data(file_path=file_path)
    return df
