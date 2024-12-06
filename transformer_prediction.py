import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from tqdm import tqdm
import json
import os
import datetime
from sklearn.model_selection import train_test_split
from sakura_data import load_sakura_data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # Calculate positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (persistent state)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SakuraTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()

        self.device = device

        # Input projection for source (3D input)
        self.src_projection = nn.Linear(3, d_model)

        # Input projection for target (2D input)
        self.tgt_projection = nn.Linear(2, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection (from d_model to 2D output)
        self.output_projection = nn.Linear(d_model, 2)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Project source and target inputs to d_model dimensions
        src = self.src_projection(src)
        tgt = self.tgt_projection(tgt)

        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Transformer forward pass
        output = self.transformer(src, tgt, src_mask, tgt_mask)

        # Project to output dimensions
        output = self.output_projection(output)

        return output

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate mask for decoder self-attention"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)


class SakuraTransformerPredictor:
    def __init__(
            self,
            d_model: int = 64,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            train_percentage: float = 0.8,
            seed: Optional[int] = None,
            sim_id: int = 0,
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """Initialize the Sakura Transformer"""
        print(f"Simulation ID: {sim_id} | PyTorch version: {torch.__version__} | Using device: {device}")

        self.device = device
        self.train_percentage = train_percentage
        self.seed = seed
        self.tqdm_bar_position = sim_id

        # Load and process data
        self.df, self.scalers = load_sakura_data()

        # Initialize model
        self.model = SakuraTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=device
        ).to(device)

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

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

    def train(self, num_epochs: int = 50, batch_size: int = 32, normalize_batches: bool = False):
        """Train the transformer"""
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            # Create batches
            np.random.shuffle(self.train_indices)
            for i in range(0, len(self.train_indices), batch_size):
                batch_indices = self.train_indices[i:i + batch_size]

                # Initialize batch tensors
                max_seq_len = max([len(self.df.iloc[idx]['temps_to_full']) for idx in batch_indices])
                batch_inputs = torch.zeros((len(batch_indices), max_seq_len, 3)).to(self.device)
                batch_targets = torch.zeros((len(batch_indices), max_seq_len, 2)).to(self.device)

                # Fill batch tensors
                for batch_idx, train_idx in enumerate(batch_indices):
                    inputs, targets, seq_length = self._prepare_sequence(train_idx)
                    batch_inputs[batch_idx, :seq_length] = inputs
                    batch_targets[batch_idx, :seq_length] = targets

                if normalize_batches:
                    batch_inputs = (batch_inputs - batch_inputs.mean()) / batch_inputs.std()
                    batch_targets = (batch_targets - batch_targets.mean()) / batch_targets.std()

                # Create target mask
                tgt_mask = self.model.generate_square_subsequent_mask(max_seq_len)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch_inputs, batch_targets, tgt_mask=tgt_mask)

                # Calculate loss
                loss = self.criterion(output, batch_targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Print epoch statistics
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    def test(self, sequence_offset: float = 1.0):
        """
        Test the transformer's predictions
        :param sequence_offset: Fraction of the sequence to simulate (0.0 to 1.0)
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for test_idx in tqdm(self.test_indices, desc=f"Testing {self.tqdm_bar_position}",
                                 position=self.tqdm_bar_position):
                # Get metadata for this sequence
                row = self.df.iloc[test_idx]

                inputs, targets, seq_length = self._prepare_sequence(test_idx)
                inputs = inputs.unsqueeze(0).to(self.device)  # Add batch dimension
                targets = targets.unsqueeze(0).to(self.device)

                # Calculate sequence cutoff point
                cutoff = int(seq_length * sequence_offset)

                # Generate predictions
                tgt_mask = self.model.generate_square_subsequent_mask(cutoff)
                output = self.model(
                    inputs[:, :cutoff],
                    targets[:, :cutoff],
                    tgt_mask=tgt_mask
                )

                # Convert predictions to numpy array
                pred_tensor = output[0].cpu().numpy()

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
                    targets[0, :cutoff, 0].cpu().numpy(),
                    self.scalers['countdown_to_first']
                )
                unscaled_true_full = self._inverse_transform_predictions(
                    targets[0, :cutoff, 1].cpu().numpy(),
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
                    'pred_first_bloom_date': row['data_start_date'] + datetime.timedelta(
                        days=cutoff) + datetime.timedelta(days=int(unscaled_pred_first[-1])),
                    'pred_full_bloom_date': row['data_start_date'] + datetime.timedelta(
                        days=cutoff) + datetime.timedelta(days=int(unscaled_pred_full[-1])),
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
        print(f"MAE (days):")
        print(f"  First bloom: {avg_mae_first:.2f}")
        print(f"  Full bloom: {avg_mae_full:.2f}")
        print(f"  Average: {(avg_mae_first + avg_mae_full) / 2:.2f}")

        metrics = {
            'mae_first': float(avg_mae_first),
            'mae_full': float(avg_mae_full)
        }

        return predictions_df, metrics

    def save_model(self, save_path: str):
        """Save model state"""
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path: str):
        """Load model state"""
        self.model.load_state_dict(torch.load(load_path))


def main(save_data_path: str,
         test_cutoff: list[float],
         d_model: int = 64,
         nhead: int = 8,
         num_encoder_layers: int = 6,
         num_decoder_layers: int = 6,
         dim_feedforward: int = 256,
         dropout: float = 0.1,
         num_epochs: int = 100,
         batch_size: int = 32,
         training_set_size: float = 0.8,
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

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize transformer
    sakura_transformer = SakuraTransformerPredictor(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        train_percentage=training_set_size,
        seed=seed,
        device=device,
        sim_id=tqdm_bar_position
    )

    # Train
    sakura_transformer.train(num_epochs=num_epochs, batch_size=batch_size)

    # Save model
    if save_model_path is not None:
        sakura_transformer.save_model(save_path=save_model_path)

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
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Folder
    save_data_path = f'src_test/transformer_d{args.d_model}_h{args.nhead}/sim_id_{args.sim_id}/'
    cutoffs = [0.5, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]

    # Run model
    main(save_data_path=save_data_path,
         test_cutoff=cutoffs,
         d_model=args.d_model,
         nhead=args.nhead,
         num_encoder_layers=args.num_encoder_layers,
         num_decoder_layers=args.num_decoder_layers,
         dim_feedforward=args.dim_feedforward,
         dropout=args.dropout,
         num_epochs=args.num_epochs,
         batch_size=args.batch_size,
         training_set_size=0.8,
         seed=args.seed,
         do_plot=True,
         device=torch.device(args.device),
         tqdm_bar_position=args.sim_id)