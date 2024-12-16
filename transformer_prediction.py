import torch
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TransformerModel
from darts.metrics import mae
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
from tqdm import tqdm
import datetime
import os
import json


class SakuraTransformer:
    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 train_percentage: float = 0.8,
                 batch_size: int = 32,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 num_epochs: int = 5,
                 seed: Optional[int] = None,
                 load_pretrained_model: Optional[str] = None,
                 sim_id: int = 0,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize the Sakura Transformer"""
        tqdm.write(f"Simulation ID: {sim_id} | PyTorch version: {torch.__version__} | Using device: {device}")

        self.device = device
        self.train_percentage = train_percentage
        self.seed = seed
        self.tqdm_bar_position = sim_id
        self.batch_size = batch_size

        # Load and process data
        self.df, self.scalers = self._load_sakura_data()

        # Initialize transformer model
        self.model = TransformerModel(
            input_chunk_length=30,
            output_chunk_length=1,
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_model * 4,  # Standard transformer uses 4x hidden size
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=num_epochs,
            random_state=seed,
            force_reset=True,
            pl_trainer_kwargs={
                "accelerator": "gpu" if str(device) == "cuda" else "cpu",
                "devices": 1
            }
        )

        if load_pretrained_model is not None:
            self.model.load(load_pretrained_model)

        # Split data
        self.train_indices, self.test_indices = self._split_data()

    def _load_sakura_data(self) -> Tuple[pd.DataFrame, dict]:
        """Load the pre-processed and scaled sakura data"""
        from sakura_data import load_sakura_data
        return load_sakura_data()

    def _split_data(self) -> Tuple[List[int], List[int]]:
        """Split data into training and testing sets"""
        all_indices = np.arange(len(self.df))
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=self.train_percentage,
            random_state=self.seed
        )
        return train_idx.tolist(), test_idx.tolist()

    def _prepare_sequence(self, idx: int) -> Tuple[TimeSeries, TimeSeries]:
        """
        Prepare input and target sequences for a single example
        :returns: - features_ts: TimeSeries containing [temperature, lat, lng]
                  - targets_ts: TimeSeries containing [countdown_first, countdown_full]
        """
        row = self.df.iloc[idx]

        # Create time index
        start_date = row['data_start_date']
        dates = pd.date_range(start=start_date, periods=len(row['temps_to_full']), freq='D')

        # Features: temperature, lat, lng
        features = pd.DataFrame({
            'temperature': row['temps_to_full'],
            'lat': [row['lat']] * len(dates),
            'lng': [row['lng']] * len(dates)
        }, index=dates)

        # Targets: countdowns
        targets = pd.DataFrame({
            'countdown_first': row['countdown_to_first'],
            'countdown_full': row['countdown_to_full']
        }, index=dates)

        # Convert to TimeSeries
        features_ts = TimeSeries.from_dataframe(features)
        targets_ts = TimeSeries.from_dataframe(targets)

        return features_ts, targets_ts

    def train(self):
        """Train the transformer"""
        train_features = []
        train_targets = []

        for train_idx in tqdm(self.train_indices, desc="Preparing training data"):
            features, targets = self._prepare_sequence(train_idx)
            train_features.append(features)
            train_targets.append(targets)

        self.model.fit(
            series=train_targets,
            past_covariates=train_features,
            verbose=True
        )

    def test(self, sequence_offset: float = 1.0):
        """Test the model"""
        predictions = []

        for test_idx in tqdm(self.test_indices):
            # Get metadata
            row = self.df.iloc[test_idx]

            # Prepare sequences
            features, targets = self._prepare_sequence(test_idx)

            # Calculate cutoff
            cutoff = int(len(features) * sequence_offset)

            # Use data up to cutoff (exclusive) to predict cutoff point
            pred_features = features[:cutoff]
            true_targets = targets[:cutoff]

            # Generate predictions for the cutoff point
            pred_targets = self.model.predict(
                n=1,
                series=true_targets[:-1],  # Use all target values except the last one
                past_covariates=pred_features[:-1]  # Use all feature values except the last one
            )

            # The prediction is for the cutoff point
            mae_first = np.abs(
                targets['countdown_first'].values()[cutoff - 1] -
                pred_targets['countdown_first'].last_value()
            )
            mae_full = np.abs(
                targets['countdown_full'].values()[cutoff - 1] -
                pred_targets['countdown_full'].last_value()
            )

            # Store results with proper sequences up to cutoff
            predictions.append({
                'site_name': row['site_name'],
                'year': row['year'],
                'start_date': row['data_start_date'],
                'date_first': row['first_bloom'],
                'date_full': row['full_bloom'],
                'true_first_sequence': targets['countdown_first'].values()[:cutoff].flatten().tolist(),
                'true_full_sequence': targets['countdown_full'].values()[:cutoff].flatten().tolist(),
                'pred_first_sequence': targets['countdown_first'].values()[:cutoff - 1].tolist() + [
                    pred_targets['countdown_first'].last_value()],
                'pred_full_sequence': targets['countdown_full'].values()[:cutoff - 1].tolist() + [
                    pred_targets['countdown_full'].last_value()],
                'cutoff': cutoff,
                'cutoff_date': row['data_start_date'] + datetime.timedelta(days=cutoff - 1),
                'pred_first_bloom_date': row['data_start_date'] + datetime.timedelta(
                    days=cutoff - 1) + datetime.timedelta(days=float(pred_targets['countdown_first'].last_value())),
                'pred_full_bloom_date': row['data_start_date'] + datetime.timedelta(
                    days=cutoff - 1) + datetime.timedelta(days=float(pred_targets['countdown_full'].last_value())),
                'full_length': len(features),
                'mae_first': mae_first,
                'mae_full': mae_full
            })

    def save_model(self, save_path: str):
        """Save the transformer model"""
        self.model.save(save_path)

    def load_model(self, load_path: str):
        """Load a saved transformer model"""
        self.model.load(load_path)

    def dump_parameters(self, save_path: Optional[str] = None) -> dict:
        """Get model parameters as a dictionary"""
        params = {
            'hidden_size': self.model.d_model,
            'num_attention_heads': self.model.nhead,
            'dropout': self.model.dropout,
            'num_encoder_layers': self.model.num_encoder_layers,
            'num_decoder_layers': self.model.num_decoder_layers,
            'batch_size': self.batch_size,
            'device': str(self.device)
        }

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
                json.dump(params, f, indent=4)

        return params


def main(save_data_path: str,
         test_cutoff: list[float],
         d_model: int,
         num_epochs: int,
         training_set_size: float = 0.8,
         num_attention_heads: int = 4,
         dropout: float = 0.1,
         batch_size: int = 32,
         num_encoder_layers: int = 3,
         num_decoder_layers: int = 3,
         save_model_path: Optional[str] = None,
         do_plot: bool = True,
         seed: Optional[int] = None,
         tqdm_bar_position: int = 0,
         device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    assert 0 < training_set_size < 1, "Training set size must be between 0 and 1"

    if do_plot:
        from utils import plot_mae_results

    # Create save path
    if save_data_path[-1] != '/':
        save_data_path += '/'
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize transformer
    sakura_transformer = SakuraTransformer(
        d_model=d_model,
        n_heads=num_attention_heads,
        dropout=dropout,
        train_percentage=training_set_size,
        batch_size=batch_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_epochs=num_epochs,
        seed=seed,
        device=device,
        sim_id=tqdm_bar_position
    )

    # Train
    sakura_transformer.train()

    # Save model
    if save_model_path is not None:
        sakura_transformer.save_model(save_model_path)

    # Save parameters
    sakura_transformer.dump_parameters(save_path=save_data_path)

    # Test
    for cutoff in test_cutoff:
        predictions_df, metrics = sakura_transformer.test(sequence_offset=cutoff)

        # Save predictions
        folder = save_data_path + f'test_cutoff_{cutoff}/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        predictions_df.to_parquet(folder + 'predictions.parquet')

        # Save metrics
        with open(folder + 'metrics.json', 'w') as f:
            json.dump(metrics, f)

        if do_plot:
            plot_mae_results(predictions_df=predictions_df, save_path=folder + 'mae')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_id', type=int, default=0)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--training_set_size', type=float, default=0.8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Folder
    save_data_path = f'src_test/transformer_d{args.d_model}_h{args.num_heads}/sim_id_{args.sim_id}/'
    cutoffs = [0.500, 0.600, 0.700, 0.750, 0.800, 0.825, 0.850, 0.875, 0.900, 0.920, 0.940, 0.950, 0.960, 0.970, 0.980,
               0.990]

    # Run model
    main(save_data_path=save_data_path,
         save_model_path=save_data_path + '/transformer_model.pt',
         test_cutoff=cutoffs,
         num_epochs=args.num_epochs,
         d_model=args.d_model,
         training_set_size=args.training_set_size,
         num_attention_heads=args.num_heads,
         dropout=args.dropout,
         batch_size=args.batch_size,
         num_encoder_layers=args.num_encoder_layers,
         num_decoder_layers=args.num_decoder_layers,
         seed=args.seed,
         do_plot=True,
         device=torch.device(args.device),
         tqdm_bar_position=args.sim_id)
