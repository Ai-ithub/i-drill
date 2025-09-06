import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from fastparquet import write

num_devices = 10
duration_days = 6 * 30
freq_seconds = 1
chunk_size = 10_000_000
start_time = datetime(2025, 1, 1, 0, 0, 0)
records_per_device = int((24 * 60 * 60 / freq_seconds) * duration_days)

failure_types = [
    'Motor_Failure', 'Pump_Leak', 'Hydraulic_Failure',
    'Bit_Wear', 'High_Vibration', 'Sensor_Fault', 'Compressor_Failure'
]

def generate_chunk(device_id, start_idx, chunk_size, total_records):
    timestamps = [start_time + timedelta(seconds=start_idx + i) for i in range(min(chunk_size, total_records - start_idx))]
    size = len(timestamps)

    data = {
        'Timestamp': timestamps,
        'Rig_ID': [f'RIG_{device_id:02d}'] * size,
        'Depth': np.cumsum(np.random.normal(0.002, 0.001, size)),
        'WOB': np.random.normal(1500, 100, size),
        'RPM': np.random.normal(80, 5, size),
        'Torque': np.random.normal(400, 30, size),
        'ROP': np.random.normal(12, 2, size),
        'Mud_Flow_Rate': np.random.normal(1200, 100, size),
        'Mud_Pressure': np.random.normal(3000, 200, size),
        'Mud_Temperature': np.random.normal(60, 3, size),
        'Mud_Density': np.random.normal(1200, 50, size),
        'Mud_Viscosity': np.random.normal(35, 5, size),
        'Mud_PH': np.random.normal(8.5, 0.2, size),
        'Gamma_Ray': np.random.normal(85, 15, size),
        'Resistivity': np.random.normal(20, 5, size),
        'Pump_Status': np.random.choice([0, 1], size=size, p=[0.01, 0.99]),
        'Compressor_Status': np.random.choice([0, 1], size=size, p=[0.02, 0.98]),
        'Power_Consumption': np.random.normal(200, 20, size),
        'Vibration_Level': np.random.normal(0.8, 0.3, size),
        'Bit_Temperature': np.random.normal(90, 5, size),
        'Motor_Temperature': np.random.normal(75, 4, size),
    }

    failure_flags = np.zeros(size, dtype=int)
    failure_count = int(0.05 * size)
    failure_indices = random.sample(range(size), failure_count)
    for idx in failure_indices:
        failure_flags[idx] = 1
    data['Maintenance_Flag'] = failure_flags

    failure_type_col = ['None'] * size
    for idx in failure_indices:
        failure_type_col[idx] = random.choice(failure_types)
    data['Failure_Type'] = failure_type_col

    df = pd.DataFrame(data)

    n_cells = df.size
    n_missing = int(n_cells * 0.05)
    for _ in range(n_missing):
        i = random.randint(0, df.shape[0] - 1)
        j = random.randint(0, df.shape[1] - 1)
        if df.columns[j] in ['Timestamp', 'Rig_ID', 'Failure_Type', 'Maintenance_Flag']:
            continue
        df.iat[i, j] = np.nan

    n_noisy = int(n_cells * 0.03)
    float_cols = df.select_dtypes(include=['float64']).columns
    for _ in range(n_noisy):
        i = random.randint(0, df.shape[0] - 1)
        j = random.choice(float_cols)
        original = df.at[i, j]
        if pd.isna(original):
            continue
        noise = original * 0.1 * (np.random.rand() * 2 - 1)
        df.at[i, j] = original + noise

    return df

def generate_device_data(device_id):
    total_records = records_per_device
    output_dir = "output_fastparquet"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, f"RIG_{device_id:02d}.parquet")

    for chunk_idx, start_idx in enumerate(range(0, total_records, chunk_size)):
        chunk_df = generate_chunk(device_id, start_idx, chunk_size, total_records)

        if chunk_idx == 0:
            write(filepath, chunk_df, compression='snappy')
        else:
            write(filepath, chunk_df, compression='snappy', append=True)

        print(f"Device {device_id} - Chunk {chunk_idx} saved.")

if __name__ == "__main__":
    for device_id in range(1, num_devices + 1):
        generate_device_data(device_id)

    print("Data generation with fastparquet completed.")