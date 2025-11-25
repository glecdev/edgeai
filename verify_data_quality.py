"""
Verify generated dataset quality
"""
import pandas as pd
import sys

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/real_structure_test/val.csv'

    print(f'Loading: {dataset_path}')
    df = pd.read_csv(dataset_path)

    print('\n=== Dataset Info ===')
    print(f'Shape: {df.shape}')
    print(f'Features: {list(df.columns)}')

    print('\n=== Dangerous Driving Stats ===')
    dangerous_cols = [
        '급가속건수20KM이상', '급가속건수40KM이상', '급가속건수60KM이상',
        '급감속건수', '급좌회전건수', '우회전건수'
    ]
    print(df[dangerous_cols].describe())

    print('\n=== Label Distribution ===')
    print(df['label'].value_counts())

    print('\n=== Anomaly Ratio ===')
    anomaly_mask = df['label'] != 'normal'
    anomaly_ratio = anomaly_mask.mean()
    print(f'Anomaly rows: {anomaly_mask.sum()} / {len(df)} = {anomaly_ratio:.2%}')

    print('\n=== GPS Range ===')
    print(f'GPS X (lon): {df["gps_x"].min():.4f} ~ {df["gps_x"].max():.4f}')
    print(f'GPS Y (lat): {df["gps_y"].min():.4f} ~ {df["gps_y"].max():.4f}')

    print('\n=== Speed Stats ===')
    print(df['vehicle_speed'].describe())

if __name__ == '__main__':
    main()
