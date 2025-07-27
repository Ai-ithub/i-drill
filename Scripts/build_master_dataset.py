import pandas as pd
import os

def create_master_dataset():
    print("Starting the master dataset creation process...")

    # --- Load both datasets ---
    df_master = pd.read_csv(os.path.join('dataset', 'newone.csv'))
    df_master.columns = df_master.columns.str.replace('\n', ' ').str.strip()
    print("Base dataset loaded and columns cleaned.")

    df_secondary = pd.read_csv(os.path.join('dataset', 'sensor_extended_processed.csv'))
    if 'Unnamed: 0' in df_secondary.columns:
        df_secondary = df_secondary.drop(columns=['Unnamed: 0'])
    print("Secondary dataset loaded and cleaned.")

    # --- NEW: Step 3: Find common columns and Concatenate ---
    print("\n" + "="*50 + "\n")
    print("Finding common columns to concatenate...")
    
    common_columns = list(set(df_master.columns) & set(df_secondary.columns))
    print(f"Found {len(common_columns)} common columns.")

    # Filter both dataframes to only have the common columns
    df1_common = df_master[common_columns]
    df2_common = df_secondary[common_columns]

    print("Concatenating dataframes...")
    # Stack the dataframes on top of each other
    df_unified = pd.concat([df1_common, df2_common], ignore_index=True)
    
    print("Concatenation complete.")
    print(f"Shape of the final unified dataset: {df_unified.shape}")
    
    # --- Final Step: Save the unified dataset ---
    output_dir = 'data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'master_dataset.parquet')
    print(f"\nSaving the final unified dataset to: {output_path}")
    df_unified.to_parquet(output_path, index=False)
    
    print("\nMaster dataset creation process finished!")
    return df_unified

if __name__ == "__main__":
    master_df = create_master_dataset()
    if master_df is not None:
        print("\n--- First 5 rows of the final unified dataset ---")
        print(master_df.head())