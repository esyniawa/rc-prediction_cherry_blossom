import torch
import numpy as np
import pandas as pd
import os
import json
from network.reservoir_torch import Reservoir, ForceTrainer
from sakura_data import load_sakura_data
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
from tqdm import tqdm
import datetime


class SakuraReservoir:
    def __init__(self,
                 reservoir_size: int = 1000,
                 tau: float = 10.0,
                 chaos_factor: float = 1.5,
                 train_percentage: float = 0.8,
                 probability_recurrent_connection: float = 0.1,
                 initial_noise: float = 0.025,  # renamed from noise_scaling
                 alpha_FORCE: float = 1.0,
                 seed: Optional[int] = None,
                 load_pretrained_model: Optional[str] = None,
                 sim_id: int = 0,
                 training_end_year: Optional[int] = None,
                 test_year: Optional[int] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize the Sakura Reservoir"""
        tqdm.write(f"Simulation ID: {sim_id} | PyTorch version: {torch.__version__} | Using device: {device}")

        self.device = device
        self.train_percentage = train_percentage
        self.seed = seed
        self.tqdm_bar_position = sim_id
        self.initial_noise = initial_noise

        # Load and process data (assumes the resulting DataFrame has a 'year' column among others)
        self.df, self.scalers = load_sakura_data()

        # New: store the training and test year info (if provided)
        self.training_end_year = training_end_year
        self.test_year = test_year

        # Initialize reservoir
        self.reservoir = Reservoir(
            dim_reservoir=reservoir_size,
            dim_input=4,  # lat, lng, temp_full, humidity_full
            dim_output=2,  # countdown_first, countdown_full
            tau=tau,
            probability_recurrent_connection=probability_recurrent_connection,
            noise_scaling=initial_noise,
            chaos_factor=chaos_factor,
            device=device
        )

        if load_pretrained_model is not None:
            self.reservoir.load(load_pretrained_model)

        # Initialize trainer
        self.trainer = ForceTrainer(self.reservoir, alpha=alpha_FORCE)

        # Split data:
        # If training_end_year and test_year are provided, use custom splitting.
        if self.training_end_year is not None and self.test_year is not None:
            self.train_indices, self.test_indices = self._split_data_custom()
        else:
            self.train_indices, self.test_indices = self._split_data_default()

    def _calculate_noise_schedule(self, sequence_length: int, epoch: int, total_epochs: int) -> torch.Tensor:
        """
        Calculate noise schedule for a sequence, implementing noise annealing.
        The noise decreases linearly from initial_noise to 0 over the sequence length.
        For later epochs, the starting noise is also reduced.
        """
        epoch_factor = 1 - (epoch / total_epochs)
        start_noise = self.initial_noise * epoch_factor
        noise_schedule = torch.linspace(start_noise, 0, sequence_length)
        return noise_schedule.to(self.device)

    def train(self, n_epochs: int = 5, dt: float = 0.1):
        """
        Train the reservoir on the sakura dataset for multiple epochs.
        (The training set is determined either by a default random split or by year if provided.)
        """
        for epoch in range(n_epochs):
            tqdm.write(f"\nEpoch {epoch + 1}/{n_epochs}")

            # Shuffle training indices at the start of each epoch
            train_indices = torch.tensor(self.train_indices)
            train_indices = train_indices[torch.randperm(len(train_indices))].tolist()

            for train_idx in tqdm(train_indices,
                                  desc=f"Training {self.tqdm_bar_position} Epoch {epoch + 1}",
                                  position=self.tqdm_bar_position):
                inputs, targets, seq_length = self._prepare_sequence(train_idx)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Calculate noise schedule for this sequence
                noise_schedule = self._calculate_noise_schedule(seq_length, epoch, n_epochs)

                # Reset reservoir state
                self.reservoir.reset_state()

                # Train over the sequence
                for t in range(seq_length):
                    self.reservoir.noise_scaling = noise_schedule[t].item()
                    _ = self.trainer.train_step(
                        inputs[t],
                        targets[t],
                        dt=dt
                    )

    def _calculate_training_error(self, dt: float):
        """Calculate and report average training error across all training sequences."""
        total_error = 0.0
        total_steps = 0

        self.reservoir.noise_scaling = 0.0  # Turn off noise for error calculation

        for train_idx in self.train_indices:
            inputs, targets, seq_length = self._prepare_sequence(train_idx)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.reservoir.reset_state()

            sequence_error = 0.0
            for t in range(seq_length):
                output = self.reservoir.forward(inputs[t], dt=dt)
                error = torch.mean((output - targets[t]) ** 2)
                sequence_error += error.item()

            total_error += sequence_error
            total_steps += seq_length

        avg_error = total_error / total_steps
        tqdm.write(f"Average training error: {avg_error:.6f}")

        self.reservoir.noise_scaling = self.initial_noise

    def _split_data_default(self) -> Tuple[List[int], List[int]]:
        """Fallback: split data into training and testing sets randomly."""
        all_indices = np.arange(len(self.df))
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=self.train_percentage,
            random_state=self.seed
        )
        return train_idx.tolist(), test_idx.tolist()

    def _split_data_custom(self) -> Tuple[List[int], List[int]]:
        """
        Split data into training and testing sets based on years.
        Training: rows where 'year' <= training_end_year.
        Testing: rows where 'year' == test_year.
        """
        train_indices = self.df.index[self.df['year'] <= self.training_end_year].tolist()
        test_indices = self.df.index[self.df['year'] == self.test_year].tolist()
        return train_indices, test_indices

    def update_years(self, training_end_year: int, test_year: int):
        """
        Update the reservoir's training and test split according to new years.
        This lets you expand the training set without re-instantiating the entire object.
        """
        self.training_end_year = training_end_year
        self.test_year = test_year
        self.train_indices, self.test_indices = self._split_data_custom()

    def reset_output_weights(self):
        """
        Reset the reservoir's output (readout) weights by reinitializing the FORCE trainer.
        (This effectively sets the output weights to zero.)
        """
        self.trainer = ForceTrainer(self.reservoir, alpha=self.trainer.alpha)

    def _prepare_sequence(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input and target sequences for a single example."""
        row = self.df.iloc[idx]

        # Static inputs (lat, lng)
        static_input = torch.tensor([row['lat'], row['lng']], dtype=torch.float32)

        # Dynamic inputs (temperature and humidity sequences)
        temps = torch.tensor(row['temps_to_full'], dtype=torch.float32)
        humidity = torch.tensor(row['humidity_to_full'], dtype=torch.float32)

        # Targets (countdown sequences)
        target_first = torch.tensor(row['countdown_to_first'], dtype=torch.float32)
        target_full = torch.tensor(row['countdown_to_full'], dtype=torch.float32)

        # Determine minimum sequence length for alignment
        seq_length = min(len(temps), len(humidity), len(target_first), len(target_full))
        temps = temps[:seq_length]
        humidity = humidity[:seq_length]
        target_first = target_first[:seq_length]
        target_full = target_full[:seq_length]

        inputs = torch.zeros((seq_length, 4))  # [lat, lng, temp_full, humid_full]
        targets = torch.zeros((seq_length, 2))  # [countdown_first, countdown_full]

        inputs[:, 0] = static_input[0]
        inputs[:, 1] = static_input[1]
        inputs[:, 2] = temps
        inputs[:, 3] = humidity

        targets[:seq_length, 0] = target_first
        targets[:seq_length, 1] = target_full

        return inputs, targets, torch.tensor(seq_length)

    @staticmethod
    def _inverse_transform_predictions(scaled_data: np.ndarray, scaler, is_sequence: bool = True) -> np.ndarray:
        """Inverse transform scaled predictions back to the original scale."""
        if is_sequence:
            original_shape = scaled_data.shape
            reshaped_data = scaled_data.reshape(-1, 1)
            unscaled_data = scaler.inverse_transform(reshaped_data).reshape(original_shape)
        else:
            unscaled_data = scaler.inverse_transform(scaled_data)
        return unscaled_data

    def test(self, dt: float = 0.1):
        """
        Test the reservoir on the test set.
        Instead of using a fractional cutoff, we use a fixed simulation window:
          - The test sequence is assumed to start on August 1 (of test_year)
          - It is run until February 28 (of test_year+1), so the cutoff is computed accordingly.
        The model is expected to predict blossom dates based on data up to February 28.
        """
        # Turn off noise during testing
        self.reservoir.noise_scaling = 0.0

        tqdm.write(f"\nTesting on {len(self.test_indices)} sequences for test year {self.test_year}...")

        predictions = []
        # For each test example:
        for test_idx in tqdm(self.test_indices, desc=f"Testing {self.tqdm_bar_position}", position=self.tqdm_bar_position):
            row = self.df.iloc[test_idx]
            inputs, targets, seq_length = self._prepare_sequence(test_idx)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Compute cutoff based on a fixed test window:
            # Test window always starts at August 1 of test_year and ends at February 28 of test_year+1.
            test_start_date = datetime.datetime(self.test_year, 8, 1)
            test_cutoff_date = datetime.datetime(self.test_year + 1, 2, 28)
            cutoff_days = (test_cutoff_date - test_start_date).days + 1
            cutoff = min(cutoff_days, seq_length)

            self.reservoir.reset_state()

            seq_predictions = []
            for t in range(cutoff):
                output = self.reservoir.forward(inputs[t], dt=dt)
                seq_predictions.append(output.cpu().detach().numpy())

            pred_tensor = np.array(seq_predictions)

            unscaled_pred_first = self._inverse_transform_predictions(
                pred_tensor[:, 0],
                self.scalers['countdown_to_first']
            )
            unscaled_pred_full = self._inverse_transform_predictions(
                pred_tensor[:, 1],
                self.scalers['countdown_to_full']
            )

            unscaled_true_first = self._inverse_transform_predictions(
                targets[:cutoff, 0].cpu().numpy(),
                self.scalers['countdown_to_first']
            )
            unscaled_true_full = self._inverse_transform_predictions(
                targets[:cutoff, 1].cpu().numpy(),
                self.scalers['countdown_to_full']
            )

            mae_first = np.mean(np.abs(unscaled_pred_first - unscaled_true_first))
            mae_full = np.mean(np.abs(unscaled_pred_full - unscaled_true_full))

            predictions.append({
                'site_name': row['site_name'],
                'year': row['year'],
                'test_start_date': test_start_date.strftime("%Y-%m-%d"),
                'date_first': row['first_bloom'],
                'date_full': row['full_bloom'],
                'true_first_sequence': unscaled_true_first.tolist(),
                'true_full_sequence': unscaled_true_full.tolist(),
                'pred_first_sequence': unscaled_pred_first.tolist(),
                'pred_full_sequence': unscaled_pred_full.tolist(),
                'cutoff': cutoff,
                'cutoff_date': (test_start_date + datetime.timedelta(days=cutoff)).strftime("%Y-%m-%d"),
                'pred_first_bloom_date': (test_start_date + datetime.timedelta(days=cutoff) + 
                                          datetime.timedelta(days=float(unscaled_pred_first[-1]))).strftime("%Y-%m-%d"),
                'pred_full_bloom_date': (test_start_date + datetime.timedelta(days=cutoff) + 
                                         datetime.timedelta(days=float(unscaled_pred_full[-1]))).strftime("%Y-%m-%d"),
                'full_length': int(seq_length.item()),
                'mae_first': mae_first,
                'mae_full': mae_full
            })

        predictions_df = pd.DataFrame(predictions)

        avg_mae_first = predictions_df['mae_first'].mean()
        avg_mae_full = predictions_df['mae_full'].mean()

        tqdm.write(f"\nMAE (days):")
        tqdm.write(f"  First bloom: {avg_mae_first:.2f}")
        tqdm.write(f"  Full bloom: {avg_mae_full:.2f}")
        tqdm.write(f"  Average: {(avg_mae_first + avg_mae_full) / 2:.2f}")

        metrics = {
            'mae_first': float(avg_mae_first),
            'mae_full': float(avg_mae_full)
        }

        return predictions_df, metrics

    def save_model(self, save_path: str):
        self.reservoir.save(path=save_path)

    def load_model(self, load_path: str):
        self.reservoir.load(load_path)

    def dump_parameters(self, save_path: Optional[str] = None):
        """Get network parameters as a dictionary, ensuring all values are JSON serializable"""
        params = {
            'dim_reservoir': self.reservoir.dim_reservoir,
            'dim_input': self.reservoir.dim_input,
            'dim_output': self.reservoir.dim_output,
            'tau': self.reservoir.tau,
            'chaos_factor': self.reservoir.chaos_factor,
            'probability_recurrent_connection': self.reservoir.probability_recurrent_connection,
            'device': str(self.reservoir.device),
            'noise_scaling': self.reservoir.noise_scaling
        }

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
                json.dump(params, f, indent=4)

        return params


def main(save_data_path: str,
         dim_reservoir: int,
         num_epochs: int,
         dt: float = 0.1,
         chaos_factor: float = 1.5,
         alpha: float = 1.0,
         probability_recurrent_connection: float = 0.2,
         noise_scaling: float = 0.025,
         save_model_path: Optional[str] = None,
         do_plot: bool = True,
         seed: Optional[int] = None,
         tqdm_bar_position: int = 0,
         device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    This main function now runs an iterative experiment.
    For each test year (e.g. from 2001 to 2020), it:
      - Sets training data from 1950 up to test_year-1,
      - Trains the reservoir,
      - Tests using a fixed window starting August 1 and cutting off at February 28,
      - Saves the model and results,
      - Resets the output weights before the next iteration.
    """
    # Create the save folder if it doesn't exist.
    if save_data_path[-1] != '/':
        save_data_path += '/'
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    # Set random seed for reproducibility.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize the reservoir.
    # (We create one instance and then update its training/test years in the loop.)
    sakura_rc = SakuraReservoir(
        reservoir_size=dim_reservoir,
        tau=10.0,
        chaos_factor=chaos_factor,
        train_percentage=0.8,  # not used when using custom split
        initial_noise=noise_scaling,
        alpha_FORCE=alpha,
        probability_recurrent_connection=probability_recurrent_connection,
        seed=seed,
        device=device,
        sim_id=tqdm_bar_position
    )

    # Loop over test years (for example, from 2001 to 2020).
    for test_year in range(2001, 2021):
        training_end_year = test_year - 1  # training from 1950 up to previous year
        tqdm.write(f"\n===== Training with data up to {training_end_year} and testing on {test_year} =====")

        # Update the training/test split based on year.
        sakura_rc.update_years(training_end_year, test_year)

        # Train the reservoir.
        sakura_rc.train(dt=dt, n_epochs=num_epochs)

        # Save the model state for this iteration.
        if save_model_path is not None:
            iter_model_path = f"{save_model_path}_test_year_{test_year}.pt"
            sakura_rc.save_model(save_path=iter_model_path)

        # Dump parameters.
        sakura_rc.dump_parameters(save_path=save_data_path)

        # Test the reservoir (the test window uses August 1 to Feb 28).
        predictions_df, metrics = sakura_rc.test(dt=dt)

        # Save predictions and metrics.
        folder = os.path.join(save_data_path, f'test_year_{test_year}/')
        if not os.path.exists(folder):
            os.makedirs(folder)
        predictions_df.to_parquet(os.path.join(folder, 'predictions.parquet'))
        with open(os.path.join(folder, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)

        # Optionally, plot MAE results.
        if do_plot:
            from utils import plot_mae_results
            plot_mae_results(predictions_df=predictions_df, save_path=os.path.join(folder, 'mae'))

        # Reset the reservoir's output (readout) weights before next iteration.
        sakura_rc.reset_output_weights()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_id', type=int, default=0)
    parser.add_argument('--dim_reservoir', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--prop_recurrent', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--chaos_factor', type=float, default=1.5)
    parser.add_argument('--noise_scaling', type=float, default=0.02)
    parser.add_argument('--training_set_size', type=float, default=0.8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    save_data_path = f'src_test/reservoir_size_{args.dim_reservoir}/sim_id_{args.sim_id}/'
    # Note: test_cutoff is no longer used since the test window is fixed.
    
    main(save_data_path=save_data_path,
         save_model_path=save_data_path + 'reservoir_model',
         num_epochs=args.num_epochs,
         dim_reservoir=args.dim_reservoir,
         dt=0.1,
         chaos_factor=args.chaos_factor,
         alpha=args.alpha,
         probability_recurrent_connection=args.prop_recurrent,
         noise_scaling=args.noise_scaling,
         seed=args.seed,
         do_plot=True,
         device=torch.device(args.device),
         tqdm_bar_position=args.sim_id)
