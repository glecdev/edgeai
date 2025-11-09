"""
GLEC DTG Edge AI - LSTM-Autoencoder Training
Anomaly detection for dangerous driving, CAN intrusion, sensor faults
"""

import os
import yaml
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch

from typing import List, Tuple, Dict
from sklearn.metrics import f1_score, precision_score, recall_score


class LSTM_Autoencoder(nn.Module):
    """
    LSTM-based Autoencoder for anomaly detection

    Architecture:
    - LSTM Encoder: Compresses time-series into latent representation
    - LSTM Decoder: Reconstructs original time-series
    - Anomaly detection via reconstruction error

    Target Performance:
    - Size: < 3MB (INT8 quantized)
    - Latency: < 35ms
    - F1-Score: > 0.85
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 128,
                 num_layers: int = 2, latent_dim: int = 32, dropout: float = 0.2):
        super(LSTM_Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        # LSTM encoder
        _, (h_n, _) = self.encoder_lstm(x)

        # Use last hidden state
        h_n = h_n[-1]  # (batch_size, hidden_dim)

        # Project to latent space
        z = self.encoder_fc(h_n)  # (batch_size, latent_dim)

        return z

    def decode(self, z: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """
        Decode latent representation to output sequence

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            sequence_length: Length of output sequence

        Returns:
            Reconstructed sequence of shape (batch_size, sequence_length, input_dim)
        """
        batch_size = z.size(0)

        # Project from latent space
        h = self.decoder_fc(z)  # (batch_size, hidden_dim)

        # Repeat for sequence length
        h = h.unsqueeze(1).repeat(1, sequence_length, 1)  # (batch, seq, hidden)

        # LSTM decoder
        output, _ = self.decoder_lstm(h)

        # Output projection
        output = self.output_fc(output)  # (batch, seq, input_dim)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, sequence_length, input_dim)
        """
        sequence_length = x.size(1)

        # Encode
        z = self.encode(x)

        # Decode
        x_reconstructed = self.decode(z, sequence_length)

        return x_reconstructed

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error for anomaly detection

        Args:
            x: Input tensor

        Returns:
            Reconstruction error per sample
        """
        x_reconstructed = self.forward(x)
        error = torch.mean((x - x_reconstructed) ** 2, dim=(1, 2))
        return error


class AnomalyDataset(Dataset):
    """
    Dataset for anomaly detection

    Labels:
    - 0: normal
    - 1: eco_driving
    - 2: harsh_braking
    - 3: harsh_acceleration
    - 4: anomaly
    """

    def __init__(self, data_path: str, window_size: int = 60,
                 features: List[str] = None):
        self.data = pd.read_csv(data_path)
        self.window_size = window_size
        self.features = features or [
            'vehicle_speed', 'engine_rpm', 'throttle_position',
            'brake_pressure', 'fuel_level', 'coolant_temp',
            'acceleration_x', 'acceleration_y', 'steering_angle', 'gps_lat'
        ]

        # Normalize features
        self.data[self.features] = (
            (self.data[self.features] - self.data[self.features].mean()) /
            self.data[self.features].std()
        )

        # Binary label: 0 for normal, 1 for anomaly
        if 'label' in self.data.columns:
            self.data['is_anomaly'] = (self.data['label'] != 'normal').astype(int)
        else:
            # Default to all normal if no labels
            self.data['is_anomaly'] = 0

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract window
        x = self.data.iloc[idx:idx+self.window_size][self.features].values

        # Label for this window (majority vote)
        labels = self.data.iloc[idx:idx+self.window_size]['is_anomaly'].values
        y = int(np.mean(labels) > 0.5)

        return torch.FloatTensor(x), torch.LongTensor([y])


def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: str) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()

        # Reconstruct input
        output = model(data)

        # Loss is reconstruction error
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: str,
             threshold: float) -> Tuple[float, float, float, float]:
    """
    Validate model and calculate anomaly detection metrics

    Args:
        threshold: Reconstruction error threshold for anomaly detection
    """
    model.eval()
    total_loss = 0.0
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)

            # Get reconstruction error
            errors = model.get_reconstruction_error(data)

            # Reconstruct for loss calculation
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()

            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Convert to numpy
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels).flatten()

    # Predict anomalies based on threshold
    predictions = (all_errors > threshold).astype(int)

    # Calculate metrics
    f1 = f1_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)

    return avg_loss, f1, precision, recall


def train_lstm_ae(config: Dict) -> None:
    """
    Main training function

    Args:
        config: Configuration dictionary from config.yaml
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # MLflow setup
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name="lstm_ae_anomaly_detection"):
        # Log configuration
        mlflow.log_params(config['lstm_ae']['training'])
        mlflow.log_params(config['lstm_ae'])

        # Create dataloaders
        print("Loading datasets...")
        train_dataset = AnomalyDataset(
            config['dataset']['train_path'],
            window_size=config['dataset']['window_size'],
            features=config['dataset']['features']
        )
        val_dataset = AnomalyDataset(
            config['dataset']['val_path'],
            window_size=config['dataset']['window_size'],
            features=config['dataset']['features']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['lstm_ae']['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['lstm_ae']['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )

        # Create model
        print("Creating model...")
        model = LSTM_Autoencoder(
            input_dim=config['lstm_ae']['input_dim'],
            hidden_dim=config['lstm_ae']['hidden_dim'],
            num_layers=config['lstm_ae']['num_layers'],
            latent_dim=config['lstm_ae']['latent_dim'],
            dropout=config['lstm_ae']['dropout']
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lstm_ae']['training']['learning_rate']
        )

        # Calculate anomaly threshold from training data
        print("Calculating anomaly threshold...")
        model.eval()
        train_errors = []
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(device)
                errors = model.get_reconstruction_error(data)
                train_errors.extend(errors.cpu().numpy())

        threshold = np.percentile(train_errors, config['lstm_ae']['anomaly_threshold'] * 100)
        print(f"Anomaly threshold: {threshold:.4f}")
        mlflow.log_param("anomaly_threshold", threshold)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        print("Starting training...")
        for epoch in range(config['lstm_ae']['training']['epochs']):
            start_time = time.time()

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, f1, precision, recall = validate(
                model, val_loader, criterion, device, threshold
            )

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1}/{config['lstm_ae']['training']['epochs']} "
                  f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"| F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} "
                  f"| Time: {epoch_time:.2f}s")

            # MLflow logging
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("f1_score", f1, step=epoch)
            mlflow.log_metric("precision", precision, step=epoch)
            mlflow.log_metric("recall", recall, step=epoch)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                model_path = "models/lstm_ae_best.pth"
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'f1_score': f1,
                    'threshold': threshold,
                }, model_path)

                mlflow.log_artifact(model_path)
            else:
                patience_counter += 1
                if patience_counter >= config['lstm_ae']['training']['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "lstm_ae_model")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train LSTM-AE model for anomaly detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.epochs:
        config['lstm_ae']['training']['epochs'] = args.epochs
    if args.batch_size:
        config['lstm_ae']['training']['batch_size'] = args.batch_size

    # Train model
    train_lstm_ae(config)


if __name__ == "__main__":
    main()
