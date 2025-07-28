import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.Scripts.Dataset import generate_chunk, failure_types

@pytest.fixture
def sample_df():
    return generate_chunk(device_id=1, start_idx=0, chunk_size=10000, total_records=10000)

def test_structure_and_columns(sample_df):
    expected_columns = {
        'Timestamp', 'Rig_ID', 'Depth', 'WOB', 'RPM', 'Torque', 'ROP',
        'Mud_Flow_Rate', 'Mud_Pressure', 'Mud_Temperature', 'Mud_Density',
        'Mud_Viscosity', 'Mud_PH', 'Gamma_Ray', 'Resistivity',
        'Pump_Status', 'Compressor_Status', 'Power_Consumption',
        'Vibration_Level', 'Bit_Temperature', 'Motor_Temperature',
        'Maintenance_Flag', 'Failure_Type'
    }
    assert set(sample_df.columns) == expected_columns
    assert len(sample_df) == 10000

def test_maintenance_flag_ratio(sample_df):
    ratio = sample_df['Maintenance_Flag'].sum() / len(sample_df)
    assert 0.04 <= ratio <= 0.06  # حدود ۵٪ باید Maintenance داشته باشن

def test_failure_type_consistency(sample_df):
    # A list of valid failure types, imported from the source script
    from src.Scripts.Dataset import failure_types
    
    for index, row in sample_df.iterrows():
        flag = row['Maintenance_Flag']
        ftype = row['Failure_Type']
        
        try:
            if flag == 0:
                assert ftype == 'None'
            else: # flag == 1
                assert ftype in failure_types
        except AssertionError:
            print(f"\n\n--- TEST FAILED AT ROW {index} ---")
            print("Problematic Data Row:")
            print(row)
            print("-----------------------------------\n")
            # Re-raise the exception to ensure the test still fails
            raise

def test_missing_values_restriction(sample_df):
    forbidden_columns = ['Timestamp', 'Rig_ID', 'Failure_Type']
    for col in forbidden_columns:
        assert sample_df[col].isna().sum() == 0

def test_data_types(sample_df):
    assert pd.api.types.is_datetime64_any_dtype(sample_df['Timestamp'])
    assert sample_df['Rig_ID'].dtype == object
    assert sample_df['Failure_Type'].dtype == object
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) >= 10  # باید بیشتر از ده ستون عددی داشته باشیم

def test_numeric_distribution_bounds(sample_df):
    assert sample_df['RPM'].mean() > 70
    assert sample_df['Torque'].mean() > 300
    assert sample_df['Mud_PH'].between(7, 10).mean() > 0.9
