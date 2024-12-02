import torch
import numpy as np
import pandas as pd
from network.reservoir_torch import Reservoir, ForceTrainer
from sakura_data import load_sakura_data
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import os
import json


class SakuraReservoir:
    def __init__(self,
                 reservoir_size: int = 1000,
                 tau: float = 10.0,
                 chaos_factor: float = 1.5,
                 train_percentage: float = 0.8,
                 probability_recurrent_connection: float = 0.1,
                 noise_scaling: float = 0.025,
                 alpha_FORCE: float = 1.0,
                 seed: Optional[int] = None,
                 load_pretrained_model: Optional[str] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize the Sakura Reservoir Computer"""
        self.device = device
        self.train_percentage = train_percentage
        self.seed = seed

        # Load and process data
        self.df, self.scalers = load_sakura_data()

        # Initialize reservoir
        self.reservoir = Reservoir(
            dim_reservoir=reservoir_size,
            dim_input=3,  # lat, lng, temp_full
            dim_output=2,  # countdown_first, countdown_full
            tau=tau,
            probability_recurrent_connection=probability_recurrent_connection,
            noise_scaling=noise_scaling,
            chaos_factor=chaos_factor,
            device=device
        )

        if load_pretrained_model is not None:
            self.reservoir.load(load_pretrained_model)

        # Initialize trainer
        self.trainer = ForceTrainer(self.reservoir, alpha=alpha_FORCE)

        # Split data
        self.train_indices, self.test_indices = self._split_data()

    def _split_data(self) -> Tuple[List[int], List[int]]:
        """Split data into training and testing sets"""
        all_indices = np.arange(len(self.df))
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=self.train_percentage,
            random_state=self.seed
        )
        return train_idx.tolist(), test_idx.tolist()

    def _prepare_sequence(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input and target sequences for a single example"""
        row = self.df.iloc[idx]

        # Static inputs (lat, lng)
        static_input = torch.tensor([row['lat'], row['lng']], dtype=torch.float32)

        # Dynamic input (temperature)
        temps = torch.tensor(row['temps_to_full'], dtype=torch.float32)

        # Targets (countdown sequences)
        target_first = torch.tensor(row['countdown_to_first'], dtype=torch.float32)
        target_full = torch.tensor(row['countdown_to_full'], dtype=torch.float32)

        # Combine inputs
        seq_length = len(temps)
        inputs = torch.zeros((seq_length, 3))  # [lat, lng, temp_full]
        targets = torch.zeros((seq_length, 2))  # [countdown_first, countdown_full]

        # Fill sequences
        inputs[:, 0] = static_input[0]  # lat
        inputs[:, 1] = static_input[1]  # lng
        inputs[:, 2] = temps  # temperature sequence

        targets[:len(target_first), 0] = target_first
        targets[:len(target_full), 1] = target_full

        return inputs, targets, torch.tensor(seq_length)

    def _inverse_transform_predictions(self, scaled_data: np.ndarray, scaler, is_sequence: bool = True) -> np.ndarray:
        """Inverse transform scaled predictions back to original scale"""
        if is_sequence:
            original_shape = scaled_data.shape
            reshaped_data = scaled_data.reshape(-1, 1)
            unscaled_data = scaler.inverse_transform(reshaped_data).reshape(original_shape)
        else:
            unscaled_data = scaler.inverse_transform(scaled_data)

        return unscaled_data

    def train(self, dt: float = 0.1):
        """Train the reservoir on the sakura dataset"""
        print(f"Training on {len(self.train_indices)} sequences...")

        for train_idx in self.train_indices:
            inputs, targets, seq_length = self._prepare_sequence(train_idx)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Reset reservoir state
            self.reservoir.reset_state()

            # Train sequence
            for t in range(seq_length):
                error_minus, error_plus = self.trainer.train_step(
                    inputs[t],
                    targets[t],
                    dt=dt
                )

            if train_idx % 10 == 0:
                mse = torch.mean(error_minus ** 2).item()
                print(f"Sequence {train_idx}, MSE: {mse:.6e}")

    def test(self, dt: float = 0.1, sequence_offset: float = 1.0):
        """
        Test the prediction of the reservoir on the test set

        :param dt: Time step for the reservoir
        :param sequence_offset: Fraction of the sequence to simulate (0.0 to 1.0)
                                e.g., 0.5 means simulate only first half of sequence
        :return: predictions_df: DataFrame containing predictions and metadata
                 metrics: Dictionary containing overall error metrics
        """
        assert 0.0 <= sequence_offset <= 1.0, "sequence offset is a fraction between 0.0 and 1.0!"

        print(f"\nTesting on {len(self.test_indices)} sequences...")

        total_mse = 0.0
        predictions = []

        for test_idx in self.test_indices:
            # Get metadata for this sequence
            row = self.df.iloc[test_idx]
            site_name = row['site_name']
            year = row['year']

            inputs, targets, seq_length = self._prepare_sequence(test_idx)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Calculate sequence cutoff point
            cutoff = int(seq_length * sequence_offset)

            # Reset reservoir state
            self.reservoir.reset_state()

            # Test sequence up to cutoff
            seq_predictions = []
            for t in range(cutoff):
                output = self.reservoir.forward(inputs[t], dt=dt)
                seq_predictions.append(output.cpu().detach().numpy())

            # Convert predictions to numpy array
            pred_tensor = torch.tensor(seq_predictions)

            # Unscale predictions and targets
            unscaled_pred_first = self._inverse_transform_predictions(
                pred_tensor[:, 0].numpy(),
                self.scalers['countdown_to_first']
            )
            unscaled_pred_full = self._inverse_transform_predictions(
                pred_tensor[:, 1].numpy(),
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

            # Calculate errors
            mse_first = np.mean((unscaled_pred_first - unscaled_true_first) ** 2)
            mse_full = np.mean((unscaled_pred_full - unscaled_true_full) ** 2)
            mae_first = np.mean(np.abs(unscaled_pred_first - unscaled_true_first))
            mae_full = np.mean(np.abs(unscaled_pred_full - unscaled_true_full))

            mse = (mse_first + mse_full) / 2
            mae = (mae_first + mae_full) / 2
            total_mse += mse

            # Store results with metadata
            predictions.append({
                'site_name': site_name,
                'year': year,
                'true_first': unscaled_true_first.tolist(),
                'true_full': unscaled_true_full.tolist(),
                'pred_first': unscaled_pred_first.tolist(),
                'pred_full': unscaled_pred_full.tolist(),
                'cutoff': cutoff,
                'full_length': seq_length.item(),
                'mse_first': mse_first,
                'mse_full': mse_full,
                'mae_first': mae_first,
                'mae_full': mae_full
            })

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Calculate overall metrics
        avg_mse = total_mse / len(self.test_indices)
        avg_mae_first = predictions_df['mae_first'].mean()
        avg_mae_full = predictions_df['mae_full'].mean()

        # Print summary statistics
        print(f"Average test MSE (unscaled): {avg_mse:.6e}")
        print(f"RMSE (days): {np.sqrt(avg_mse):.2f}")
        print(f"MAE (days):")
        print(f"  First bloom: {avg_mae_first:.2f}")
        print(f"  Full bloom: {avg_mae_full:.2f}")
        print(f"  Average: {(avg_mae_first + avg_mae_full) / 2:.2f}")

        metrics = {
            'mse': avg_mse,
            'mae_first': avg_mae_first,
            'mae_full': avg_mae_full
        }

        return predictions_df, metrics

    def save_model(self, save_path: str):
        self.reservoir.save(path=save_path)


def main(save_data_path: str,
         test_cutoff: list[float],
         dim_reservoir: int,
         dt: float = 0.1,
         chaos_factor: float = 1.5,
         alpha: float = 1.0,
         probability_recurrent_connection: float = 0.2,
         save_model_path: Optional[str] = None,
         do_plot: bool = True,
         seed: Optional[int] = None):

    if do_plot:
        from utils import plot_mae_results

    if save_data_path[-1] != '/':
        save_data_path += '/'

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize reservoir
    sakura_rc = SakuraReservoir(
        reservoir_size=dim_reservoir,
        tau=10.0,
        chaos_factor=chaos_factor,
        train_percentage=0.8,
        alpha_FORCE=alpha,
        probability_recurrent_connection=probability_recurrent_connection,
        seed=seed
    )

    # Train
    sakura_rc.train(dt=dt)

    # Save model
    if save_model_path is not None:
        sakura_rc.save_model(save_path=save_model_path)

    # Test
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    for cutoff in test_cutoff:
        predictions_df, metrics = sakura_rc.test(dt=dt, sequence_offset=cutoff)

        # save predictions
        folder = save_data_path + f'test_cutoff_{cutoff}/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        predictions_df.to_parquet(folder + 'predictions.parquet')

        # save metrics
        with open(folder + 'metrics.json', 'w') as f:
            json.dump(metrics, f)

        if do_plot:
            plot_mae_results(predictions_df=predictions_df, save_path=folder + 'mae')


if __name__ == "__main__":
    cutoffs = [0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925]
    main(save_data_path='src_test',
         test_cutoff=cutoffs,
         dim_reservoir=2000,
         seed=42)
