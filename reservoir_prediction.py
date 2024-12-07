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
                 noise_scaling: float = 0.025,
                 alpha_FORCE: float = 1.0,
                 seed: Optional[int] = None,
                 load_pretrained_model: Optional[str] = None,
                 sim_id: int = 0,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize the Sakura Reservoir"""
        print(f"Simulation ID: {sim_id} | PyTorch version: {torch.__version__} | Using device: {device}")

        self.device = device
        self.train_percentage = train_percentage
        self.seed = seed
        self.tqdm_bar_position = sim_id

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
        # save noise scaling to switch it on and off between training and testing
        self.noise_scaling = noise_scaling

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

    @staticmethod
    def _inverse_transform_predictions(scaled_data: np.ndarray, scaler, is_sequence: bool = True) -> np.ndarray:
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
        # set noise scaling for training
        if not self.reservoir.noise_scaling:
            self.reservoir.noise_scaling = self.noise_scaling

        for train_idx in tqdm(self.train_indices, desc=f"Training {self.tqdm_bar_position}", position=self.tqdm_bar_position):
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

    def test(self, dt: float = 0.1, sequence_offset: float = 1.0):
        """
        Test the prediction of the reservoir on the test set

        :param dt: Time step for the reservoir
        :param sequence_offset: Fraction of the sequence to simulate (0.0 to 1.0)
                                e.g., 0.5 means simulate only first half of sequence
        :return: predictions_df: DataFrame containing predictions and metadata
                 metrics: Dictionary containing overall error metrics
        """
        # set noise scaling to 0 for testing
        if self.reservoir.noise_scaling:
            self.reservoir.noise_scaling = 0.0

        print(f"\nTesting on {len(self.test_indices)} sequences...")

        predictions = []
        for test_idx in tqdm(self.test_indices, desc=f"Testing {self.tqdm_bar_position}", position=self.tqdm_bar_position):
            # Get metadata for this sequence
            row = self.df.iloc[test_idx]

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
            pred_tensor = np.array(seq_predictions)

            # Unscale predictions and targets
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

            # Calculate errors
            mae_first = np.mean(np.abs(unscaled_pred_first - unscaled_true_first))
            mae_full = np.mean(np.abs(unscaled_pred_full - unscaled_true_full))

            # Store results with metadata
            predictions.append({
                'site_name': row['site_name'],
                'year': row['year'],
                'start_date': row['data_start_date'],
                'date_first': row['first_bloom'],
                'date_full': row['full_bloom'],
                'true_first_sequence': unscaled_true_first.tolist(),
                'true_full_sequence': unscaled_true_full.tolist(),
                'pred_first_sequence': unscaled_pred_first.tolist(),
                'pred_full_sequence': unscaled_pred_full.tolist(),
                'cutoff': cutoff,
                'cutoff_date': row['data_start_date'] + datetime.timedelta(days=cutoff),
                'pred_first_bloom_date': row['data_start_date'] + datetime.timedelta(days=cutoff) + datetime.timedelta(days=int(unscaled_pred_first[-1])),
                'pred_full_bloom_date': row['data_start_date'] + datetime.timedelta(days=cutoff) + datetime.timedelta(days=int(unscaled_pred_full[-1])),
                'full_length': seq_length.item(),
                'mae_first': mae_first,
                'mae_full': mae_full
            })

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Calculate overall metrics
        avg_mae_first = predictions_df['mae_first'].mean()
        avg_mae_full = predictions_df['mae_full'].mean()

        # Print summary statistics
        print(f"\nMAE (days):")
        print(f"  First bloom: {avg_mae_first:.2f}")
        print(f"  Full bloom: {avg_mae_full:.2f}")
        print(f"  Average: {(avg_mae_first + avg_mae_full) / 2:.2f}")

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
        import json
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
            # Save parameters to JSON file
            with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
                json.dump(params, f, indent=4)

        return params


def main(save_data_path: str,
         test_cutoff: list[float],
         dim_reservoir: int,
         training_set_size: float = 0.8,
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

    assert 0 < training_set_size < 1, "Training set size must be between 0 and 1"

    if do_plot:
        from utils import plot_mae_results

    # create save path
    if save_data_path[-1] != '/':
        save_data_path += '/'
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize reservoir
    sakura_rc = SakuraReservoir(
        reservoir_size=dim_reservoir,
        tau=10.0,
        chaos_factor=chaos_factor,
        train_percentage=training_set_size,
        noise_scaling=noise_scaling,
        alpha_FORCE=alpha,
        probability_recurrent_connection=probability_recurrent_connection,
        seed=seed,
        device=device,
        sim_id=tqdm_bar_position
    )

    # Train
    sakura_rc.train(dt=dt)

    # Save model
    if save_model_path is not None:
        sakura_rc.save_model(save_path=save_model_path)
    # save parameters
    sakura_rc.dump_parameters(save_path=save_data_path)

    # Test
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_id', type=int, default=0)
    parser.add_argument('--dim_reservoir', type=int, default=2_000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--prop_recurrent', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--chaos_factor', type=float, default=1.5)
    parser.add_argument('--noise_scaling', type=float, default=0.02)
    parser.add_argument('--training_set_size', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # folder
    save_data_path = f'src_test/reservoir_size_{args.dim_reservoir}/sim_id_{args.sim_id}/'
    cutoffs = [0.500, 0.600, 0.700, 0.750, 0.800, 0.825, 0.850, 0.875, 0.900, 0.920, 0.940, 0.950, 0.960, 0.970, 0.980, 0.990,]

    # run model
    main(save_data_path=save_data_path,
         save_model_path=save_data_path + '/reservoir_model.pt',
         test_cutoff=cutoffs,
         dim_reservoir=args.dim_reservoir,
         training_set_size=args.training_set_size,
         dt=0.1,
         chaos_factor=args.chaos_factor,
         alpha=args.alpha,
         probability_recurrent_connection=args.prop_recurrent,
         noise_scaling=args.noise_scaling,
         seed=args.seed,
         do_plot=True,
         device=torch.device(args.device),
         tqdm_bar_position=args.sim_id)
