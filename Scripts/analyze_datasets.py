import pandas as pd
import os

# Define the directory where data files are located
data_directory = "dataset"

# List of files to check within that directory
files_to_check = [
    'newone.csv',
    'file.txt', # Checking this file as well from your screenshot
    'sensor_extended_processed.csv',
    'sensor_readings_extended.csv',
    'sensor_readings_noisy.csv'
]

print("="*60)
print(f"Analyzing files in the '{data_directory}' directory...")
print("="*60 + "\n")

for filename in files_to_check:
    # Construct the full path to the file
    file_path = os.path.join(data_directory, filename)
    
    if not os.path.exists(file_path):
        print(f"--- File not found: {file_path} ---\n")
        continue
    
    # Only try to read .csv files with pandas
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            print(f"--- Analysis of: {file_path} ---")
            print("Columns:", df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head())
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Could not read or process {file_path}. Error: {e}\n")
    else:
        print(f"--- Skipping non-CSV file: {file_path} ---\n")