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

class LSTMRULModel(nn.Module):
    """
    LSTM-based model for RUL prediction
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
        
    def _init_weights(self):
        """Initialize model weights"""
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
    Positional encoding for Transformer model
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
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
        return x + self.pe[:x.size(0), :]

class TransformerRULModel(nn.Module):
    """
    Transformer-based model for RUL prediction
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
        
    def _init_weights(self):
        """Initialize model weights"""
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
    CNN-LSTM hybrid model for RUL prediction
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

def create_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create RUL prediction models
    
    Args:
        model_type: Type of model ('lstm', 'transformer', 'cnn_lstm')
        input_dim: Number of input features
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMRULModel(input_dim, **kwargs)
    elif model_type == 'transformer':
        return TransformerRULModel(input_dim, **kwargs)
    elif model_type == 'cnn_lstm':
        return CNNLSTMRULModel(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: 'lstm', 'transformer', 'cnn_lstm'")

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