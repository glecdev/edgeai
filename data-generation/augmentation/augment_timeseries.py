"""
GLEC DTG - Time-Series Data Augmentation
Augment vehicle telemetry data using tsaug library

Augmentation methods:
- Jitter: Add random noise
- Scaling: Scale amplitude
- TimeWarp: Warp time axis
- MagWarp: Warp magnitude
- Window slice/crop
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import tsaug
    from tsaug import TimeWarp, Drift, Quantize, Jitter, Reverse
except ImportError:
    print("tsaug not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tsaug'])
    import tsaug
    from tsaug import TimeWarp, Drift, Quantize, Jitter, Reverse


class TimeSeriesAugmentor:
    """
    Augment vehicle time-series data for AI training

    Preserves:
    - timestamp ordering
    - label information
    - GPS coordinates (not augmented)
    """

    def __init__(self, input_file: str, output_file: str, methods: list = None):
        self.input_file = input_file
        self.output_file = output_file
        self.methods = methods or ['jitter', 'scaling', 'timewarp']

        self.features_to_augment = [
            'vehicle_speed',
            'engine_rpm',
            'throttle_position',
            'brake_pressure',
            'fuel_level',
            'coolant_temp',
            'acceleration_x',
            'acceleration_y',
            'steering_angle',
            'fuel_consumption'
        ]

    def load_data(self):
        """Load CSV data"""
        print(f"Loading data from: {self.input_file}")
        df = pd.read_csv(self.input_file)
        print(f"✅ Loaded {len(df)} samples")
        return df

    def create_augmenters(self):
        """Create augmentation pipelines"""
        augmenters = {}

        if 'jitter' in self.methods:
            # Add random noise (±2% of value range)
            augmenters['jitter'] = Jitter(sigma=0.02)

        if 'scaling' in self.methods:
            # Scale amplitude (0.9-1.1x)
            augmenters['scaling'] = tsaug.Convolve(window='triang', size=3) @ tsaug.AddNoise(scale=0.01)

        if 'timewarp' in self.methods:
            # Warp time axis (speed up/slow down slightly)
            augmenters['timewarp'] = TimeWarp(n_speed_change=5, max_speed_ratio=1.2)

        if 'magwarp' in self.methods:
            # Warp magnitude
            augmenters['magwarp'] = tsaug.Drift(max_drift=0.05, n_drift_points=5)

        return augmenters

    def augment_window(self, window_data, augmenter):
        """Augment a single time window"""
        # Extract feature values
        features = window_data[self.features_to_augment].values

        # Transpose for tsaug (expects (n_features, n_timesteps))
        features_T = features.T

        # Apply augmentation
        try:
            augmented_T = augmenter.augment(features_T)

            # Transpose back
            augmented = augmented_T.T

            # Create augmented dataframe
            augmented_df = window_data.copy()

            for i, feat in enumerate(self.features_to_augment):
                augmented_df[feat] = augmented[:, i]

            # Ensure values stay within reasonable bounds
            augmented_df = self.clip_values(augmented_df)

            return augmented_df

        except Exception as e:
            print(f"⚠️  Augmentation failed: {e}")
            return None

    def clip_values(self, df):
        """Clip augmented values to reasonable ranges"""
        df['vehicle_speed'] = df['vehicle_speed'].clip(0, 200)
        df['engine_rpm'] = df['engine_rpm'].clip(500, 7000)
        df['throttle_position'] = df['throttle_position'].clip(0, 100)
        df['brake_pressure'] = df['brake_pressure'].clip(0, 100)
        df['fuel_level'] = df['fuel_level'].clip(0, 100)
        df['coolant_temp'] = df['coolant_temp'].clip(60, 120)
        df['acceleration_x'] = df['acceleration_x'].clip(-10, 10)
        df['acceleration_y'] = df['acceleration_y'].clip(-5, 5)
        df['steering_angle'] = df['steering_angle'].clip(-45, 45)
        df['fuel_consumption'] = df['fuel_consumption'].clip(0, 50)

        return df

    def augment_dataset(self, window_size=60, augmentation_factor=2):
        """
        Augment entire dataset

        Args:
            window_size: Size of time window for augmentation
            augmentation_factor: Number of augmented versions per original window
        """
        df = self.load_data()

        augmenters = self.create_augmenters()
        print(f"✅ Created {len(augmenters)} augmenters: {list(augmenters.keys())}")

        augmented_samples = []

        # Process in windows
        num_windows = len(df) // window_size
        print(f"Processing {num_windows} windows of size {window_size}...")

        for window_idx in range(num_windows):
            start_idx = window_idx * window_size
            end_idx = start_idx + window_size

            window_data = df.iloc[start_idx:end_idx].copy()

            # Add original window
            augmented_samples.append(window_data)

            # Generate augmented versions
            for aug_idx in range(augmentation_factor):
                # Randomly select augmentation method
                aug_method = np.random.choice(list(augmenters.keys()))
                augmenter = augmenters[aug_method]

                augmented_window = self.augment_window(window_data, augmenter)

                if augmented_window is not None:
                    augmented_samples.append(augmented_window)

            if window_idx % 100 == 0:
                print(f"  Progress: {window_idx}/{num_windows} windows")

        # Combine all augmented samples
        final_df = pd.concat(augmented_samples, ignore_index=True)

        print(f"\n✅ Augmentation completed:")
        print(f"   Original samples: {len(df)}")
        print(f"   Augmented samples: {len(final_df)}")
        print(f"   Augmentation ratio: {len(final_df)/len(df):.2f}x")

        # Save augmented dataset
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_df.to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")

        return final_df

    def analyze_augmentation_quality(self, original_df, augmented_df):
        """Analyze quality of augmented data"""
        print("\n📊 Augmentation Quality Analysis:")

        for feat in self.features_to_augment:
            orig_mean = original_df[feat].mean()
            orig_std = original_df[feat].std()

            aug_mean = augmented_df[feat].mean()
            aug_std = augmented_df[feat].std()

            print(f"  {feat}:")
            print(f"    Original: μ={orig_mean:.2f}, σ={orig_std:.2f}")
            print(f"    Augmented: μ={aug_mean:.2f}, σ={aug_std:.2f}")
            print(f"    Difference: {abs(aug_mean - orig_mean)/orig_mean*100:.1f}%")


def split_dataset(input_file: str, output_dir: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    print(f"\nSplitting dataset: {input_file}")
    df = pd.read_csv(input_file)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / 'train.csv'
    val_file = output_path / 'val.csv'
    test_file = output_path / 'test.csv'

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"✅ Dataset split:")
    print(f"   Train: {len(train_df)} samples → {train_file}")
    print(f"   Val:   {len(val_df)} samples → {val_file}")
    print(f"   Test:  {len(test_df)} samples → {test_file}")

    return train_file, val_file, test_file


def main():
    parser = argparse.ArgumentParser(description='Augment time-series vehicle data')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--methods', nargs='+', default=['jitter', 'scaling', 'timewarp'],
                        help='Augmentation methods')
    parser.add_argument('--window-size', type=int, default=60,
                        help='Window size for augmentation')
    parser.add_argument('--factor', type=int, default=2,
                        help='Augmentation factor (versions per window)')
    parser.add_argument('--split', action='store_true',
                        help='Split into train/val/test after augmentation')
    parser.add_argument('--split-dir', default='../datasets',
                        help='Directory for train/val/test splits')

    args = parser.parse_args()

    # Create augmentor
    augmentor = TimeSeriesAugmentor(
        input_file=args.input,
        output_file=args.output,
        methods=args.methods
    )

    # Augment dataset
    augmented_df = augmentor.augment_dataset(
        window_size=args.window_size,
        augmentation_factor=args.factor
    )

    # Analyze quality
    original_df = augmentor.load_data()
    augmentor.analyze_augmentation_quality(original_df, augmented_df)

    # Split if requested
    if args.split:
        split_dataset(
            input_file=args.output,
            output_dir=args.split_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

    print("\n✅ Augmentation pipeline completed!")


if __name__ == "__main__":
    main()
