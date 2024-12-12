import torch
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from sklearn.model_selection import train_test_split
from sakura_data import load_sakura_data


class SakuraTFTPredictor:
    def __init__(
            self,
            train_percentage: float = 0.8,
            max_prediction_length: int = 30,
            max_encoder_length: int = 90,
            seed: Optional[int] = None,
            learning_rate: float = 0.001,
            hidden_size: int = 64,
            attention_head_size: int = 4,
            dropout: float = 0.1,
            hidden_continuous_size: int = 32,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the Sakura TFT Predictor"""
        print(f"PyTorch version: {torch.__version__} | Using device: {device}")

        self.device = device
        self.train_percentage = train_percentage
        self.seed = seed
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length

        # Model hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size

        # Load data and scalers
        self.df, self.scalers = load_sakura_data()

        # Split data into train and test sets
        self.train_indices, self.test_indices = self._split_data()

        # Process data for TFT format
        self._prepare_data()

    def _split_data(self) -> Tuple[list, list]:
        """Split data into training and testing sets"""
        all_indices = np.arange(len(self.df))
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=self.train_percentage,
            random_state=self.seed
        )
        return train_idx.tolist(), test_idx.tolist()

    def _prepare_data(self):
        """Transform data into format suitable for TFT"""
        processed_data = []

        for idx, row in self.df.iterrows():
            # Get base dates and sequences
            start_date = pd.to_datetime(row['data_start_date'])
            temps = row['temps_to_full']
            countdown_first = row['countdown_to_first']
            countdown_full = row['countdown_to_full']

            # Create time series for each day
            for i in range(len(temps)):
                current_date = start_date + timedelta(days=i)

                processed_data.append({
                    'time_idx': i,
                    'site_name': row['site_name'],
                    'year': row['year'],
                    'date': current_date,
                    'lat': row['lat'],
                    'lng': row['lng'],
                    'temperature': temps[i],
                    'countdown_first': countdown_first[i] if i < len(countdown_first) else None,
                    'countdown_full': countdown_full[i] if i < len(countdown_full) else None,
                    'sequence_length': len(temps),
                    'data_index': idx  # Store original dataframe index
                })

        # Convert to DataFrame
        self.processed_df = pd.DataFrame(processed_data)

        # Create training dataset using only training indices
        train_mask = self.processed_df['data_index'].isin(self.train_indices)
        self.training = TimeSeriesDataSet(
            self.processed_df[train_mask],
            time_idx="time_idx",
            target=["countdown_first", "countdown_full"],
            group_ids=["site_name", "year"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["site_name"],
            static_reals=["lat", "lng"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["temperature", "countdown_first", "countdown_full"],
            target_normalizer=GroupNormalizer(
                groups=["site_name"], transformation=None  # No normalization as data is already scaled
            ),
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
        )

        # Create data loaders
        self.train_dataloader = self.training.to_dataloader(
            train=True,
            batch_size=64,
            num_workers=0
        )

        # Initialize TFT model
        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=torch.nn.MSELoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        # Move model to device
        self.tft = self.tft.to(self.device)

    def train(self, max_epochs: int = 50):
        """Train the TFT model"""

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=self.device,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",  # Using train_loss since we don't have validation set
                    min_delta=1e-4,
                    patience=10,
                    verbose=False,
                    mode="min"
                )
            ],
        )

        # Fit model
        trainer.fit(
            self.tft,
            train_dataloaders=self.train_dataloader
        )

    def test(self, sequence_offset: float = 1.0):
        """
        Test the model's predictions using partial sequences
        :param sequence_offset: Fraction of the sequence to use (0.0 to 1.0)
        """
        self.tft.eval()
        predictions = []

        for test_idx in self.test_indices:
            # Get original data row
            row = self.df.iloc[test_idx]

            # Calculate cutoff point
            full_length = len(row['temps_to_full'])
            cutoff = int(full_length * sequence_offset)

            # Create test dataset with truncated sequence
            test_data = []
            start_date = pd.to_datetime(row['data_start_date'])

            for i in range(cutoff):
                current_date = start_date + timedelta(days=i)
                test_data.append({
                    'time_idx': i,
                    'site_name': row['site_name'],
                    'year': row['year'],
                    'date': current_date,
                    'lat': row['lat'],
                    'lng': row['lng'],
                    'temperature': row['temps_to_full'][i],
                    'countdown_first': row['countdown_to_first'][i] if i < len(row['countdown_to_first']) else None,
                    'countdown_full': row['countdown_to_full'][i] if i < len(row['countdown_to_full']) else None,
                })

            # Create test TimeSeriesDataSet
            test_df = pd.DataFrame(test_data)
            test_dataset = TimeSeriesDataSet.from_dataset(
                self.training,
                test_df,
                predict=True,
                stop_randomization=True
            )
            test_dataloader = test_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            # Generate predictions
            raw_predictions = self.tft.predict(test_dataloader)

            # Get the predictions
            pred_first = raw_predictions[0, :, 0].numpy()  # First bloom predictions
            pred_full = raw_predictions[0, :, 1].numpy()  # Full bloom predictions

            # Inverse transform predictions
            unscaled_pred_first = self.scalers['countdown_to_first'].inverse_transform(
                pred_first.reshape(-1, 1)
            ).flatten()
            unscaled_pred_full = self.scalers['countdown_to_full'].inverse_transform(
                pred_full.reshape(-1, 1)
            ).flatten()

            # Get true values for comparison
            true_first = row['countdown_to_first'][:cutoff]
            true_full = row['countdown_to_full'][:cutoff]

            # Inverse transform true values
            unscaled_true_first = self.scalers['countdown_to_first'].inverse_transform(
                np.array(true_first).reshape(-1, 1)
            ).flatten()
            unscaled_true_full = self.scalers['countdown_to_full'].inverse_transform(
                np.array(true_full).reshape(-1, 1)
            ).flatten()

            # Calculate errors
            mae_first = np.mean(np.abs(unscaled_pred_first - unscaled_true_first))
            mae_full = np.mean(np.abs(unscaled_pred_full - unscaled_true_full))

            # Store predictions with metadata
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
                'cutoff_date': row['data_start_date'] + timedelta(days=cutoff),
                'pred_first_bloom_date': row['data_start_date'] + timedelta(days=cutoff) +
                                         timedelta(days=int(unscaled_pred_first[-1])),
                'pred_full_bloom_date': row['data_start_date'] + timedelta(days=cutoff) +
                                        timedelta(days=int(unscaled_pred_full[-1])),
                'full_length': full_length,
                'mae_first': mae_first,
                'mae_full': mae_full
            })

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Calculate overall metrics
        avg_mae_first = predictions_df['mae_first'].mean()
        avg_mae_full = predictions_df['mae_full'].mean()

        # Print summary statistics
        print(f"MAE (days):")
        print(f"  First bloom: {avg_mae_first:.2f}")
        print(f"  Full bloom: {avg_mae_full:.2f}")
        print(f"  Average: {(avg_mae_first + avg_mae_full) / 2:.2f}")

        metrics = {
            'mae_first': float(avg_mae_first),
            'mae_full': float(avg_mae_full)
        }

        return predictions_df, metrics

    def save_model(self, path: str):
        """Save the trained model"""
        torch.save(self.tft.state_dict(), path)

    def load_model(self, path: str):
        """Load a trained model"""
        self.tft.load_state_dict(torch.load(path))


def main(save_data_path: str,
         test_cutoff: list[float],
         hidden_layer_size: int = 64,
         training_set_size: float = 0.8,
         max_encoder_length: int = 10,
         head_size: int = 8,
         num_epochs: int = 5,
         dropout: float = 0.1,
         save_model_path: Optional[str] = None,
         do_plot: bool = True,
         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

    import os
    import json

    from utils import plot_mae_results

    # Initialize predictor
    predictor = SakuraTFTPredictor(
        train_percentage=training_set_size,
        max_prediction_length=30,
        max_encoder_length=max_encoder_length,
        hidden_size=hidden_layer_size,
        attention_head_size=head_size,
        dropout=dropout,
        device=device
    )

    # Train model
    predictor.train(max_epochs=num_epochs)

    # Test with different sequence cutoffs
    for cutoff in test_cutoff:
        print(f"\nTesting with sequence_offset = {cutoff}")
        predictions_df, metrics = predictor.test(sequence_offset=cutoff)

        # save predictions
        folder = save_data_path + f'test_cutoff_{cutoff}/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        predictions_df.to_parquet(folder + 'predictions.parquet')
        if do_plot:
            plot_mae_results(predictions_df=predictions_df, save_path=folder + 'mae')

    # Save model if needed
    if save_model_path is not None:
        predictor.save_model(save_model_path + 'sakura_tft_model.pth')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_id', type=int, default=0)
    parser.add_argument('--head_size', type=int, default=8)
    parser.add_argument('--training_set_size', type=float, default=0.8)
    parser.add_argument('--max_encoder_length', type=int, default=0)
    parser.add_argument('--hidden_layer_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    save_data_path = f'src_test/transformer_h{args.head_size}_max_enc{args.max_encoder_length}/sim_id_{args.sim_id}/'
    cutoffs = [0.500, 0.600, 0.700, 0.750, 0.800, 0.825, 0.850, 0.875, 0.900, 0.920, 0.940, 0.950, 0.960, 0.970, 0.980, 0.990,]

    main(save_data_path=save_data_path,
         test_cutoff=cutoffs,
         hidden_layer_size=args.hidden_layer_size,
         training_set_size=args.training_set_size,
         max_encoder_length=args.max_encoder_length,
         head_size=args.head_size,
         num_epochs=args.num_epochs,
         dropout=0.1,
         do_plot=True,
         device=args.device)
