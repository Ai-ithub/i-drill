#!/usr/bin/env python3
"""
Data Loading and Preprocessing for RUL Prediction

This module handles loading sequenced, feature-engineered datasets from Task 3.1
and prepares them for deep learning models.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict, List
import logging
from pathlib import Path

class RULDataset(Dataset):
    """
    PyTorch Dataset class for RUL prediction data
    """
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 50):
        """
        Initialize RUL Dataset
        
        Args:
            sequences: Input sequences (samples, features, time_steps)
            targets: RUL target values
            sequence_length: Length of input sequences
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class RULDataLoader:
    """
    Main data loader class for RUL prediction
    """
    
    def __init__(self, data_path: str, sequence_length: int = 50, 
                 test_size: float = 0.2, val_size: float = 0.1,
                 scaler_type: str = 'standard'):
        """
        Initialize RUL Data Loader
        
        Args:
            data_path: Path to Task 3.1 processed datasets
            sequence_length: Length of input sequences for models
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            scaler_type: Type of scaling ('standard' or 'minmax')
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.scaler_type = scaler_type
        
        # Initialize scalers
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
            
        self.logger = logging.getLogger(__name__)
        
    def load_task31_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load sequenced, feature-engineered datasets from Task 3.1
        
        Returns:
            Dictionary containing loaded datasets
        """
        datasets = {}
        
        # Expected file patterns from Task 3.1
        file_patterns = [
            'sequenced_sensor_data.csv',
            'feature_engineered_data.csv', 
            'processed_drilling_data.csv',
            'rul_labeled_data.csv'
        ]
        
        for pattern in file_patterns:
            file_path = self.data_path / pattern
            if file_path.exists():
                self.logger.info(f"Loading {pattern}...")
                datasets[pattern.replace('.csv', '')] = pd.read_csv(file_path)
            else:
                self.logger.warning(f"File not found: {file_path}")
                
        if not datasets:
            # Create sample data if Task 3.1 files don't exist
            self.logger.info("Creating sample RUL dataset...")
            datasets = self._create_sample_data()
            
        return datasets
    
    def _create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        Create sample RUL data for demonstration purposes
        
        Returns:
            Dictionary with sample datasets
        """
        np.random.seed(42)
        
        # Simulate drilling equipment sensor data
        n_samples = 10000
        n_features = 15
        
        # Generate synthetic sensor readings
        features = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'equipment_id': np.random.choice(['DRILL_001', 'DRILL_002', 'DRILL_003'], n_samples),
            'temperature': np.random.normal(75, 10, n_samples),
            'pressure': np.random.normal(1500, 200, n_samples),
            'vibration_x': np.random.normal(0, 0.5, n_samples),
            'vibration_y': np.random.normal(0, 0.5, n_samples),
            'vibration_z': np.random.normal(0, 0.5, n_samples),
            'rotation_speed': np.random.normal(120, 15, n_samples),
            'torque': np.random.normal(800, 100, n_samples),
            'flow_rate': np.random.normal(50, 8, n_samples),
            'mud_weight': np.random.normal(1.2, 0.1, n_samples),
            'depth': np.random.uniform(1000, 5000, n_samples),
            'wear_indicator': np.random.exponential(0.1, n_samples),
            'operating_hours': np.cumsum(np.ones(n_samples)),
        }
        
        # Calculate synthetic RUL based on operating conditions
        base_rul = 1000  # Base RUL in hours
        degradation_factors = (
            (features['temperature'] - 75) / 100 +
            (features['pressure'] - 1500) / 2000 +
            features['wear_indicator'] * 10 +
            features['operating_hours'] / 10000
        )
        
        features['rul'] = np.maximum(0, base_rul - degradation_factors * 100)
        
        df = pd.DataFrame(features)
        
        return {
            'sequenced_sensor_data': df,
            'feature_engineered_data': df,
            'rul_labeled_data': df
        }
    
    def create_sequences(self, data: pd.DataFrame, 
                        feature_columns: List[str],
                        target_column: str = 'rul') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling
        
        Args:
            data: Input dataframe
            feature_columns: List of feature column names
            target_column: Name of target column (RUL)
            
        Returns:
            Tuple of (sequences, targets)
        """
        # Sort by equipment_id and timestamp
        if 'equipment_id' in data.columns and 'timestamp' in data.columns:
            data = data.sort_values(['equipment_id', 'timestamp'])
        
        sequences = []
        targets = []
        
        # Group by equipment to create sequences
        if 'equipment_id' in data.columns:
            for equipment_id in data['equipment_id'].unique():
                equipment_data = data[data['equipment_id'] == equipment_id]
                seq, tgt = self._create_equipment_sequences(
                    equipment_data, feature_columns, target_column
                )
                sequences.extend(seq)
                targets.extend(tgt)
        else:
            # Single equipment case
            seq, tgt = self._create_equipment_sequences(
                data, feature_columns, target_column
            )
            sequences.extend(seq)
            targets.extend(tgt)
            
        return np.array(sequences), np.array(targets)
    
    def _create_equipment_sequences(self, data: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_column: str) -> Tuple[List, List]:
        """
        Create sequences for a single equipment
        """
        sequences = []
        targets = []
        
        features = data[feature_columns].values
        rul_values = data[target_column].values
        
        for i in range(len(features) - self.sequence_length + 1):
            seq = features[i:i + self.sequence_length]
            target = rul_values[i + self.sequence_length - 1]
            
            sequences.append(seq)
            targets.append(target)
            
        return sequences, targets
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train, validation, and test data loaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load datasets
        datasets = self.load_task31_datasets()
        
        # Use the main dataset (prefer rul_labeled_data if available)
        if 'rul_labeled_data' in datasets:
            main_data = datasets['rul_labeled_data']
        elif 'feature_engineered_data' in datasets:
            main_data = datasets['feature_engineered_data']
        else:
            main_data = list(datasets.values())[0]
            
        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['timestamp', 'equipment_id', 'rul']
        feature_columns = [col for col in main_data.columns if col not in exclude_cols]
        
        self.logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
        
        # Create sequences
        sequences, targets = self.create_sequences(main_data, feature_columns)
        
        self.logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        
        # Scale features and targets
        n_samples, seq_len, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        sequences_scaled = self.feature_scaler.fit_transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(n_samples, seq_len, n_features)
        
        targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Split data
        n_total = len(sequences_scaled)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size)
        n_train = n_total - n_test - n_val
        
        # Create datasets
        train_dataset = RULDataset(
            sequences_scaled[:n_train], 
            targets_scaled[:n_train],
            self.sequence_length
        )
        
        val_dataset = RULDataset(
            sequences_scaled[n_train:n_train+n_val],
            targets_scaled[n_train:n_train+n_val],
            self.sequence_length
        )
        
        test_dataset = RULDataset(
            sequences_scaled[n_train+n_val:],
            targets_scaled[n_train+n_val:],
            self.sequence_length
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.logger.info(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def get_feature_dim(self) -> int:
        """
        Get the number of input features
        
        Returns:
            Number of input features
        """
        datasets = self.load_task31_datasets()
        main_data = list(datasets.values())[0]
        exclude_cols = ['timestamp', 'equipment_id', 'rul']
        feature_columns = [col for col in main_data.columns if col not in exclude_cols]
        return len(feature_columns)