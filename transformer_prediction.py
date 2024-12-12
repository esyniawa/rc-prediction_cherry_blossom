import pytorch_forecasting
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import torch
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import datetime
from sakura_data import load_sakura_data
import os


class SakuraTFTPredictor:
    def __init__(
            self,
            hidden_size: int = 64,
            attention_head_size: int = 8,
            dropout: float = 0.1,
            hidden_continuous_size: int = 32,
            train_percentage: float = 0.8,
            max_encoder_length: int = 30,
            max_prediction_length: int = 5,
            seed: Optional[int] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the Sakura TFT Predictor"""
        self.device = device
        self.train_percentage = train_percentage
        self.seed = seed
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length

        # Load and process data
        self.df, self.scalers = load_sakura_data()

        # Process data for TFT format
        self.processed_df = self._process_data_for_tft()

        # Create training and validation datasets
        self._create_datasets()

        # Initialize model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_data,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=pytorch_forecasting.metrics.QuantileLoss(),
            learning_rate=1e-3,
            reduce_on_plateau_patience=4
        )

        self.model.to(self.device)

    def _process_data_for_tft(self) -> pd.DataFrame:
        """Process the data into a format suitable for TFT"""
        processed_data = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]

            # Get sequences
            temps = row['temps_to_full']
            target_first = row['countdown_to_first']
            target_full = row['countdown_to_full']

            # Create time index
            timestamps = pd.date_range(
                start=row['data_start_date'],
                periods=len(temps),
                freq='D'
            )

            # Create sequence data
            for t in range(len(temps)):
                processed_data.append({
                    'series_id': idx,
                    'time_idx': t,
                    'lat': row['lat'],
                    'lng': row['lng'],
                    'temperature': temps[t],
                    'countdown_first': target_first[t] if t < len(target_first) else np.nan,
                    'countdown_full': target_full[t] if t < len(target_full) else np.nan,
                    'date': timestamps[t]
                })

        return pd.DataFrame(processed_data)

    def _create_datasets(self):
        """Create TimeSeriesDataSet for training and validation"""
        # Split data
        train_series_ids, val_series_ids = train_test_split(
            self.processed_df['series_id'].unique(),
            train_size=self.train_percentage,
            random_state=self.seed
        )

        # Create training dataset
        self.training_data = TimeSeriesDataSet(
            data=self.processed_df[self.processed_df['series_id'].isin(train_series_ids)],
            time_idx="time_idx",
            target=["countdown_first", "countdown_full"],
            group_ids=["series_id"],
            static_categoricals=[],
            static_reals=["lat", "lng"],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["temperature"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=["countdown_first", "countdown_full"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            add_relative_time_idx=True,
            add_target_scales=True,
            randomize_length=None,
            target_normalizer=GroupNormalizer(
                groups=["series_id"], transformation="softplus"
            )
        )

        # Create validation dataset
        self.validation_data = TimeSeriesDataSet.from_dataset(
            self.training_data,
            data=self.processed_df[self.processed_df['series_id'].isin(val_series_ids)],
            stop_randomization=True
        )

        # Create data loaders
        self.train_dataloader = self.training_data.to_dataloader(
            train=True, batch_size=32, num_workers=0
        )
        self.val_dataloader = self.validation_data.to_dataloader(
            train=False, batch_size=32, num_workers=0
        )

    def train(self, max_epochs: int = 50):
        """Train the TFT model"""
        trainer = pytorch_forecasting.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1
        )

        trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

    def predict(self, sequence_offset: float = 1.0):
        """Generate predictions"""
        predictions = []

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for series_id in self.validation_data.group_ids:
                # Get series data
                series_data = self.processed_df[self.processed_df['series_id'] == series_id]

                # Calculate cutoff point
                full_length = len(series_data)
                cutoff = int(full_length * sequence_offset)

                if cutoff <= self.max_encoder_length:
                    continue

                # Prepare prediction input
                pred_data = self.validation_data.get_prediction_sample(
                    series_data.iloc[:cutoff]
                )

                # Generate predictions
                raw_predictions = self.model.predict(pred_data)

                # Process predictions
                pred_first = raw_predictions[..., 0].numpy()
                pred_full = raw_predictions[..., 1].numpy()

                # Get original row data
                original_row = self.df.iloc[series_id]

                # Store results
                predictions.append({
                    'site_name': original_row['site_name'],
                    'year': original_row['year'],
                    'start_date': original_row['data_start_date'],
                    'date_first': original_row['first_bloom'],
                    'date_full': original_row['full_bloom'],
                    'cutoff': cutoff,
                    'cutoff_date': original_row['data_start_date'] + datetime.timedelta(days=cutoff),
                    'pred_first_bloom_date': original_row['data_start_date'] + datetime.timedelta(
                        days=cutoff + int(pred_first[-1])),
                    'pred_full_bloom_date': original_row['data_start_date'] + datetime.timedelta(
                        days=cutoff + int(pred_full[-1])),
                    'mae_first': np.mean(np.abs(pred_first - series_data['countdown_first'].iloc[-len(pred_first):])),
                    'mae_full': np.mean(np.abs(pred_full - series_data['countdown_full'].iloc[-len(pred_full):]))
                })

        predictions_df = pd.DataFrame(predictions)

        # Calculate overall metrics
        metrics = {
            'mae_first': float(predictions_df['mae_first'].mean()),
            'mae_full': float(predictions_df['mae_full'].mean())
        }

        return predictions_df, metrics

    def save_model(self, path: str):
        """Save the model"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load the model"""
        self.model.load_state_dict(torch.load(path))


def main(
        save_data_path: str,
        test_cutoff: list[float],
        hidden_size: int = 64,
        attention_head_size: int = 8,
        dropout: float = 0.1,
        hidden_continuous_size: int = 32,
        max_epochs: int = 5,
        training_set_size: float = 0.8,
        save_model_path: Optional[str] = None,
        seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Initialize predictor
    predictor = SakuraTFTPredictor(
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        train_percentage=training_set_size,
        seed=seed,
        device=device
    )

    # Train model
    predictor.train(max_epochs=max_epochs)

    # Save model if requested
    if save_model_path:
        predictor.save_model(save_model_path)

    # Test at different cutoff points
    for cutoff in test_cutoff:
        predictions_df, metrics = predictor.predict(sequence_offset=cutoff)

        # Save results
        folder = f"{save_data_path}/test_cutoff_{cutoff}/"
        os.makedirs(folder, exist_ok=True)

        predictions_df.to_parquet(f"{folder}/predictions.parquet")
        with open(f"{folder}/metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--attention_head_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_continuous_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Define cutoffs
    cutoffs = [0.500, 0.600, 0.700, 0.750, 0.800, 0.825, 0.850, 0.875,
               0.900, 0.920, 0.940, 0.950, 0.960, 0.970, 0.980, 0.990]

    # Run model
    main(
        save_data_path=f'tft_results/hidden_{args.hidden_size}/',
        test_cutoff=cutoffs,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        max_epochs=args.max_epochs,
        training_set_size=0.8,
        seed=args.seed,
        device=args.device
    )
