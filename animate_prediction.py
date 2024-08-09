import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from monitoring import PopMonitor
from network.reservoir import *
from utils import load_lasso_weights


n_trials = 4
ann.compile('annarchy/test_runs/', clean=False)

def set_output_weights(weights: np.ndarray):
    for d in Wo.dendrites:
        d.w = weights

monitors = PopMonitor([input_pop, reservoir_pop, output_pop], sampling_rate=1.0, auto_start=True)
sakura_data = pd.read_parquet('data/training_data.parquet')
output_pop_weights, _ = load_lasso_weights(save_path='lasso_data', n_runs=20)

set_output_weights(output_pop_weights)

def run_reservoir(inputs: pd.DataFrame, period: int = 5):

    for i, row in inputs.iterrows():
        len_temps = row['Temps'].size

        @ann.every(period=period)
        def set_inputs(n):
            # Set inputs to the network
            input_pop[0].baseline = row['Temps'][n]  # Temps as dynamic input
            input_pop[1].baseline = row['Lat']  # Lat
            input_pop[2].baseline = row['Lng']  # Lng

        # simulate
        ann.simulate(period * len_temps, callbacks=True)
        ann.reset(monitors=False, populations=True)


# pick n_trials random samples
sample = sakura_data.sample(n=n_trials)
run_reservoir(sample)
monitors.animate_current_monitors(
    plot_types=['Bar', 'Matrix', 'Line'],
)
