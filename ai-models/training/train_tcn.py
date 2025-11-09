"""
GLEC DTG Edge AI - TCN Model Training
Temporal Convolutional Network for fuel consumption prediction
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


class TCN(nn.Module):
    """
    Temporal Convolutional Network for fuel consumption prediction

    Architecture:
    - Multiple dilated causal convolutional layers
    - Residual connections for gradient flow
    - Dropout for regularization

    Target Performance:
    - Size: < 4MB (INT8 quantized)
    - Latency: < 25ms
    - Accuracy: > 85% (R² score)
    """

    def __init__(self, input_dim: int = 10, output_dim: int = 1,
                 num_channels: List[int] = [64, 128, 256],
                 kernel_size: int = 3, dropout: float = 0.2):
        super(TCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = num_channels

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Dilated causal convolution
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size-1) * dilation_size,
                    dilation=dilation_size
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # Output projection
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Transpose to (batch, channels, sequence)
        x = x.transpose(1, 2)

        # Apply TCN layers
        x = self.network(x)

        # Global average pooling
        x = torch.mean(x, dim=2)

        # Output projection
        x = self.fc(x)

        return x


class VehicleDataset(Dataset):
    """
    Dataset for vehicle CAN bus data

    Data format:
    - 60 timesteps (1Hz sampling rate)
    - 10 features per timestep
    - Target: fuel consumption
    """

    def __init__(self, data_path: str, window_size: int = 60,
                 features: List[str] = None, target: str = 'fuel_consumption'):
        self.data = pd.read_csv(data_path)
        self.window_size = window_size
        self.features = features or [
            'vehicle_speed', 'engine_rpm', 'throttle_position',
            'brake_pressure', 'fuel_level', 'coolant_temp',
            'acceleration_x', 'acceleration_y', 'steering_angle', 'gps_lat'
        ]
        self.target = target

        # Normalize features (TODO: save scaler for inference)
        self.data[self.features] = (self.data[self.features] - self.data[self.features].mean()) / self.data[self.features].std()

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract window of features
        x = self.data.iloc[idx:idx+self.window_size][self.features].values

        # Target is the future fuel consumption
        y = self.data.iloc[idx+self.window_size][self.target]

        return torch.FloatTensor(x), torch.FloatTensor([y])


def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: str) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: str) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calculate R² score
    predictions = np.array(predictions)
    targets = np.array(targets)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    return avg_loss, r2_score


def train_tcn(config: Dict) -> None:
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

    with mlflow.start_run(run_name="tcn_fuel_prediction"):
        # Log configuration
        mlflow.log_params(config['tcn']['training'])
        mlflow.log_params(config['tcn'])

        # Create dataloaders
        print("Loading datasets...")
        train_dataset = VehicleDataset(
            config['dataset']['train_path'],
            window_size=config['dataset']['window_size'],
            features=config['dataset']['features']
        )
        val_dataset = VehicleDataset(
            config['dataset']['val_path'],
            window_size=config['dataset']['window_size'],
            features=config['dataset']['features']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['tcn']['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['tcn']['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )

        # Create model
        print("Creating model...")
        model = TCN(
            input_dim=config['tcn']['input_dim'],
            output_dim=config['tcn']['output_dim'],
            num_channels=config['tcn']['num_channels'],
            kernel_size=config['tcn']['kernel_size'],
            dropout=config['tcn']['dropout']
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['tcn']['training']['learning_rate']
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        print("Starting training...")
        for epoch in range(config['tcn']['training']['epochs']):
            start_time = time.time()

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, r2_score = validate(model, val_loader, criterion, device)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1}/{config['tcn']['training']['epochs']} "
                  f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                  f"| R² Score: {r2_score:.4f} | Time: {epoch_time:.2f}s")

            # MLflow logging
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("r2_score", r2_score, step=epoch)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                model_path = "models/tcn_fuel_best.pth"
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'r2_score': r2_score,
                }, model_path)

                mlflow.log_artifact(model_path)
            else:
                patience_counter += 1
                if patience_counter >= config['tcn']['training']['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "tcn_model")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train TCN model for fuel prediction')
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
        config['tcn']['training']['epochs'] = args.epochs
    if args.batch_size:
        config['tcn']['training']['batch_size'] = args.batch_size

    # Train model
    train_tcn(config)


if __name__ == "__main__":
    main()
