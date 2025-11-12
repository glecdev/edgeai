"""
Simple TCN + LSTM-AE Training Script
Direct GPU training without complex dependencies
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import time
from pathlib import Path

# ============================================================================
# Models
# ============================================================================

class TCN(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = torch.mean(x, dim=2)  # Global average pooling
        return self.fc(x)

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=2, latent_dim=32):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (h_n, _) = self.encoder_lstm(x)
        z = self.encoder_fc(h_n[-1])

        # Decode
        h = self.decoder_fc(z)
        h = h.unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder_lstm(h)
        output = self.output_fc(output)
        return output

    def get_reconstruction_error(self, x):
        x_reconstructed = self.forward(x)
        error = torch.mean((x - x_reconstructed) ** 2, dim=(1, 2))
        return error

# ============================================================================
# Dataset
# ============================================================================

class VehicleDataset(Dataset):
    def __init__(self, csv_path, window_size=60, for_autoencoder=False):
        print(f'Loading {csv_path}...')
        self.df = pd.read_csv(csv_path)
        self.window_size = window_size
        self.for_autoencoder = for_autoencoder

        self.features = ['vehicle_speed', 'engine_rpm', 'throttle_position', 'brake_pressure',
                        'coolant_temp', 'fuel_level', 'acceleration_x', 'latitude', 'longitude', 'altitude']

        # Build samples
        self.samples = []
        max_idx = len(self.df) - window_size

        for i in range(0, max_idx, window_size):
            window = self.df.iloc[i:i+window_size][self.features].values
            fuel = self.df.iloc[i+window_size-1]['fuel_consumption']
            label = self.df.iloc[i+window_size-1]['label']

            if window.shape[0] == window_size:
                self.samples.append((window, fuel, 1 if label != 'normal' else 0))

        # Normalize
        all_data = np.array([s[0] for s in self.samples])
        self.mean = all_data.mean(axis=(0,1))
        self.std = all_data.std(axis=(0,1)) + 1e-8

        print(f'Loaded {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, fuel, anomaly = self.samples[idx]
        window = (window - self.mean) / self.std

        if self.for_autoencoder:
            return torch.FloatTensor(window), torch.LongTensor([anomaly])
        else:
            return torch.FloatTensor(window), torch.FloatTensor([fuel])

# ============================================================================
# Training Functions
# ============================================================================

def train_tcn(device, epochs=50, batch_size=32, lr=0.001):
    print('='*80)
    print('Training TCN Model')
    print('='*80)

    train_dataset = VehicleDataset('../../datasets/train.csv')
    val_dataset = VehicleDataset('../../datasets/val.csv')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TCN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Starting training...')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        preds, targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                preds.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())

        val_loss /= len(val_loader)

        # R2 score
        preds = np.array(preds)
        targets = np.array(targets)
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R2: {r2:.4f} | Time: {epoch_time:.1f}s')

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'r2_score': r2,
            }, 'models/tcn_fuel_best.pth')
            print(f'  -> Best model saved (R2: {r2:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Training complete! Best val loss: {best_val_loss:.4f}')
    return model

def train_lstm_ae(device, epochs=50, batch_size=32, lr=0.001):
    print('='*80)
    print('Training LSTM-AE Model')
    print('='*80)

    train_dataset = VehicleDataset('../../datasets/train.csv', for_autoencoder=True)
    val_dataset = VehicleDataset('../../datasets/val.csv', for_autoencoder=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM_Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # Calculate threshold from training data
    print('Calculating anomaly threshold...')
    model.eval()
    train_errors = []
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            errors = model.get_reconstruction_error(data)
            train_errors.extend(errors.cpu().numpy())

    threshold = np.percentile(train_errors, 95)
    print(f'Anomaly threshold (95th percentile): {threshold:.6f}')

    print('Starting training...')
    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        all_errors, all_labels = [], []
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                errors = model.get_reconstruction_error(data)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)

        # Calculate F1
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels).flatten()
        predictions = (all_errors > threshold).astype(int)

        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f} | Time: {epoch_time:.1f}s')

        # Save best
        if f1 > best_f1:
            best_f1 = f1
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'f1_score': f1,
                'threshold': threshold,
            }, 'models/lstm_ae_best.pth')
            print(f'  -> Best model saved (F1: {f1:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Training complete! Best F1: {best_f1:.4f}')
    return model

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    print()

    # Train TCN
    tcn_model = train_tcn(device, epochs=50, batch_size=32)

    print()
    print()

    # Train LSTM-AE
    lstm_model = train_lstm_ae(device, epochs=50, batch_size=32)

    print()
    print('='*80)
    print('All training complete!')
    print('='*80)
