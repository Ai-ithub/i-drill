#!/usr/bin/env python3
"""
Deep Learning Models for RUL Prediction

This module implements LSTM and Transformer architectures for predicting
the remaining useful life of drilling equipment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, RegressorMixin

class LSTMRULModel(nn.Module):
    """
    LSTM-based model for RUL (Remaining Useful Life) prediction.
    
    Uses bidirectional LSTM layers with multi-head attention mechanism
    to capture temporal patterns in sensor data sequences. Suitable for
    time-series prediction tasks where long-term dependencies are important.
    
    Architecture:
        - Bidirectional LSTM layers for sequence processing
        - Multi-head attention mechanism for feature importance
        - Fully connected layers for final prediction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2,
                 bidirectional: bool = True):
        """
        Initialize LSTM RUL Model
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMRULModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """
        Initialize model weights using appropriate initialization strategies.
        
        Uses Xavier uniform initialization for input-to-hidden weights,
        orthogonal initialization for hidden-to-hidden weights, and
        zero initialization for biases.
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step output
        last_output = attended_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc_layers(last_output)
        
        return output

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    
    Adds positional information to input embeddings using sinusoidal functions.
    This allows the Transformer to understand the order of elements in sequences
    since Transformers don't have inherent sequential processing.
    
    Attributes:
        pe: Buffer containing pre-computed positional encodings
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension (embedding size)
            max_len: Maximum sequence length to support
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (sequence_length, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

class TransformerRULModel(nn.Module):
    """
    Transformer-based model for RUL (Remaining Useful Life) prediction.
    
    Uses Transformer encoder architecture with self-attention mechanisms
    to capture complex temporal dependencies in sensor data. Well-suited
    for parallel processing and capturing long-range dependencies.
    
    Architecture:
        - Input projection layer
        - Positional encoding
        - Transformer encoder layers
        - Global average pooling
        - Fully connected output layers
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        """
        Initialize Transformer RUL Model
        
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super(TransformerRULModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """
        Initialize model weights using appropriate initialization strategies.
        
        Uses Xavier uniform initialization for input-to-hidden weights,
        orthogonal initialization for hidden-to-hidden weights, and
        zero initialization for biases.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions (batch_size, 1)
        """
        # Project input to model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        
        # Output prediction
        output = self.output_layers(pooled)
        
        return output

class CNNLSTMRULModel(nn.Module):
    """
    CNN-LSTM hybrid model for RUL (Remaining Useful Life) prediction.
    
    Combines convolutional neural networks for local feature extraction
    with LSTM layers for temporal sequence modeling. The CNN layers extract
    local patterns while LSTM layers capture long-term dependencies.
    
    Architecture:
        - 1D CNN layers for feature extraction
        - Batch normalization and pooling
        - Bidirectional LSTM for sequence modeling
        - Fully connected output layers
    """
    
    def __init__(self, input_dim: int, cnn_channels: list = [64, 128, 256],
                 lstm_hidden_dim: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize CNN-LSTM RUL Model
        
        Args:
            input_dim: Number of input features
            cnn_channels: List of CNN channel dimensions
            lstm_hidden_dim: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(CNNLSTMRULModel, self).__init__()
        
        self.input_dim = input_dim
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = 1
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output dimension
        # This is approximate and may need adjustment based on sequence length
        cnn_output_dim = cnn_channels[-1]
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layers
        lstm_output_dim = lstm_hidden_dim * 2  # Bidirectional
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions (batch_size, 1)
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Reshape for CNN: (batch_size, 1, seq_len * input_dim)
        x_cnn = x.view(batch_size, 1, -1)
        
        # CNN feature extraction
        cnn_features = self.cnn(x_cnn)
        
        # Reshape for LSTM: (batch_size, new_seq_len, cnn_channels[-1])
        cnn_seq_len = cnn_features.size(-1)
        cnn_features = cnn_features.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        output = self.output_layers(last_output)
        
        return output


class BaselineRULModel(BaseEstimator, RegressorMixin):
    """
    Wrapper class for scikit-learn baseline models to work with sequence data.
    
    Provides a scikit-learn compatible interface for traditional machine learning
    models (Linear Regression, Random Forest, SVR) to work with sequence data
    by flattening sequences into feature vectors.
    
    Supports model types:
        - 'linear': Linear Regression
        - 'random_forest': Random Forest Regressor
        - 'svr': Support Vector Regression
    """
    
    def __init__(self, model_type: str = 'linear', **kwargs):
        """
        Initialize baseline model
        
        Args:
            model_type: Type of baseline model ('linear', 'random_forest', 'svr')
            **kwargs: Additional parameters for the underlying model
        """
        self.model_type = model_type.lower()
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        
        # Create the underlying model
        if self.model_type == 'linear':
            self.model = LinearRegression(**kwargs)
        elif self.model_type == 'random_forest':
            # Default parameters for Random Forest
            default_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            default_params.update(kwargs)
            self.model = RandomForestRegressor(**default_params)
        elif self.model_type == 'svr':
            # Default parameters for SVR
            default_params = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
            default_params.update(kwargs)
            self.model = SVR(**default_params)
        else:
            raise ValueError(f"Unknown baseline model type: {model_type}. "
                           f"Supported types: 'linear', 'random_forest', 'svr'")
    
    def _flatten_sequences(self, X):
        """
        Flatten sequence data for traditional ML models
        
        Args:
            X: Input sequences of shape (batch_size, seq_len, features)
            
        Returns:
            Flattened features of shape (batch_size, seq_len * features)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        if len(X.shape) == 3:
            # Flatten sequence dimension
            batch_size, seq_len, features = X.shape
            return X.reshape(batch_size, seq_len * features)
        elif len(X.shape) == 2:
            # Already flattened
            return X
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
    
    def fit(self, X, y):
        """
        Fit the baseline model
        
        Args:
            X: Input sequences
            y: Target values
        """
        X_flat = self._flatten_sequences(X)
        
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        self.model.fit(X_flat, y.ravel())
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions with the baseline model
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_flat = self._flatten_sequences(X)
        predictions = self.model.predict(X_flat)
        
        return predictions
    
    def score(self, X, y) -> float:
        """
        Return the coefficient of determination R² of the prediction.
        
        Args:
            X: Input sequences
            y: True target values
            
        Returns:
            R² score (coefficient of determination)
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        predictions = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        from sklearn.metrics import r2_score
        return r2_score(y.ravel(), predictions)


def create_model(model_type: str, input_dim: int, **kwargs):
    """
    Factory function to create RUL prediction models
    
    Args:
        model_type: Type of model ('lstm', 'transformer', 'cnn_lstm', 'linear', 'random_forest', 'svr')
        input_dim: Number of input features
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model (PyTorch nn.Module or BaselineRULModel)
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMRULModel(input_dim, **kwargs)
    elif model_type == 'transformer':
        return TransformerRULModel(input_dim, **kwargs)
    elif model_type == 'cnn_lstm':
        return CNNLSTMRULModel(input_dim, **kwargs)
    elif model_type in ['linear', 'random_forest', 'svr']:
        return BaselineRULModel(model_type, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: 'lstm', 'transformer', 'cnn_lstm', 'linear', 'random_forest', 'svr'")

if __name__ == "__main__":
    # Test models
    batch_size, seq_len, input_dim = 32, 50, 15
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test LSTM model
    lstm_model = create_model('lstm', input_dim)
    lstm_output = lstm_model(x)
    print(f"LSTM output shape: {lstm_output.shape}")
    
    # Test Transformer model
    transformer_model = create_model('transformer', input_dim)
    transformer_output = transformer_model(x)
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Test CNN-LSTM model
    cnn_lstm_model = create_model('cnn_lstm', input_dim)
    cnn_lstm_output = cnn_lstm_model(x)
    print(f"CNN-LSTM output shape: {cnn_lstm_output.shape}")