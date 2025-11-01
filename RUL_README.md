# RUL Prediction System

A comprehensive deep learning system for Remaining Useful Life (RUL) prediction of drilling equipment using PyTorch.

## Overview

This system implements state-of-the-art deep learning models for predicting the remaining useful life of drilling equipment based on sensor data and operational parameters. It supports multiple architectures including LSTM, Transformer, and CNN-LSTM hybrid models.

## Features

- **Multiple Model Architectures**: LSTM, Transformer, and CNN-LSTM models
- **Comprehensive Data Processing**: Automated data loading, preprocessing, and sequence generation
- **Advanced Training Pipeline**: Early stopping, learning rate scheduling, and model checkpointing
- **Detailed Evaluation**: Multiple metrics, visualizations, and statistical analysis
- **Configurable**: YAML-based configuration system
- **Production Ready**: Modular design with proper logging and error handling

## Installation

1. **Clone the repository** (if applicable) or ensure you have the RUL prediction files

2. **Install dependencies**:
   ```bash
   pip install -r requirements_rul.txt
   ```

3. **Verify PyTorch installation**:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())  # Should be True if you have CUDA
   ```

## Quick Start

### 1. Prepare Your Data

Ensure your Task 3.1 processed datasets are available in the specified directory:
```
data/
└── task31_processed/
    ├── sequenced_sensor_data.csv
    ├── feature_engineered_data.csv
    ├── processed_drilling_data.csv
    └── rul_labeled_data.csv
```

### 2. Configure the System

Modify `config.yaml` according to your needs:
```yaml
data:
  data_path: "./data/task31_processed"
  sequence_length: 50
  
model:
  type: "lstm"  # or "transformer" or "cnn_lstm"
  hidden_dim: 128
  
training:
  epochs: 100
  learning_rate: 0.001
```

### 3. Train and Evaluate

**Train a new model**:
```bash
cd src/rul_prediction
python main.py --mode train
```

**Train and evaluate**:
```bash
python main.py --mode both
```

**Evaluate existing model**:
```bash
python main.py --mode evaluate --model-path ./checkpoints/best_model.pth
```

## Model Architectures

### 1. LSTM Model
- Bidirectional LSTM layers
- Multi-head attention mechanism
- Fully connected output layers
- Best for: Sequential patterns in sensor data

### 2. Transformer Model
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Best for: Long-range dependencies

### 3. CNN-LSTM Hybrid
- 1D CNN for feature extraction
- LSTM for temporal modeling
- Combined architecture benefits
- Best for: Complex feature patterns

## Data Format

The system expects CSV files with the following structure:

```csv
timestamp,equipment_id,temperature,pressure,vibration_x,vibration_y,vibration_z,rotation_speed,torque,flow_rate,mud_weight,depth,wear_indicator,operating_hours,rul
2023-01-01 00:00:00,DRILL_001,75.2,1520.5,0.12,-0.08,0.15,118.5,785.2,48.7,1.18,1250.8,0.05,1.0,850.5
...
```

**Required columns**:
- `timestamp`: Time of measurement
- `equipment_id`: Unique equipment identifier
- `rul`: Remaining useful life (target variable)
- Feature columns: Sensor readings and operational parameters

## Configuration Options

### Data Configuration
```yaml
data:
  data_path: "./data/task31_processed"  # Path to datasets
  sequence_length: 50                    # Input sequence length
  test_size: 0.2                        # Test set proportion
  val_size: 0.1                         # Validation set proportion
  scaler_type: "standard"               # 'standard' or 'minmax'
```

### Model Configuration
```yaml
model:
  type: "lstm"                          # 'lstm', 'transformer', 'cnn_lstm'
  hidden_dim: 128                       # Hidden dimension
  num_layers: 2                         # Number of layers
  dropout: 0.2                          # Dropout rate
  bidirectional: true                   # Bidirectional LSTM
```

### Training Configuration
```yaml
training:
  epochs: 100                           # Maximum epochs
  learning_rate: 0.001                  # Learning rate
  weight_decay: 0.00001                 # L2 regularization
  scheduler_type: "cosine"              # LR scheduler
  patience: 10                          # Early stopping patience
  batch_size: 32                        # Batch size
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R² Score**: Coefficient of determination

### RUL-Specific Metrics
- **RUL Score**: Asymmetric scoring function (NASA PHM08)
- **Accuracy within thresholds**: ±10, ±20, ±30 time units
- **Early vs Late predictions**: Conservative vs risky predictions

### Statistical Analysis
- Error distribution analysis
- Confidence intervals
- Performance by RUL ranges
- Residual analysis

## Output Files

After training and evaluation, the system generates:

### Checkpoints Directory
```
checkpoints/
├── best_model.pth              # Best model weights
├── checkpoint_epoch_X.pth      # Regular checkpoints
├── training_history.png        # Training curves
└── tensorboard/                # TensorBoard logs
```

### Results Directory
```
results/
├── evaluation_results.png      # Comprehensive plots
├── evaluation_metrics.csv      # Numerical metrics
├── predictions.csv             # Predictions vs true values
├── error_statistics.csv        # Error analysis
└── range_analysis.csv          # Performance by RUL ranges
```

## Advanced Usage

### Custom Model Creation
```python
from src.rul_prediction.models import create_model

# Create custom LSTM model
model = create_model(
    model_type='lstm',
    input_dim=15,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3,
    bidirectional=True
)
```

### Custom Training Loop
```python
from src.rul_prediction.trainer import RULTrainer
from src.rul_prediction.data_loader import RULDataLoader

# Initialize components
data_loader = RULDataLoader(data_path='./data')
train_loader, val_loader, test_loader = data_loader.prepare_data()

trainer = RULTrainer(model=model)
history = trainer.train(train_loader, val_loader, epochs=50)
```

### Model Comparison
```python
from src.rul_prediction.evaluator import RULEvaluator

evaluator = RULEvaluator(model)
results = evaluator.evaluate(test_loader)

# Compare multiple models
model_results = {
    'LSTM': lstm_results,
    'Transformer': transformer_results,
    'CNN-LSTM': cnn_lstm_results
}
comparison = evaluator.compare_models(model_results)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config.yaml
   - Use smaller model dimensions
   - Enable gradient checkpointing

2. **Data Loading Errors**:
   - Check data path in config.yaml
   - Ensure CSV files have required columns
   - Verify data format and types

3. **Poor Model Performance**:
   - Increase sequence length
   - Try different model architectures
   - Adjust learning rate and regularization
   - Check data quality and preprocessing

### Performance Optimization

1. **Training Speed**:
   - Use GPU if available
   - Increase batch size (if memory allows)
   - Use mixed precision training
   - Optimize data loading (num_workers)

2. **Model Accuracy**:
   - Experiment with different architectures
   - Tune hyperparameters
   - Use ensemble methods
   - Improve feature engineering

## API Reference

### RULDataLoader
```python
class RULDataLoader:
    def __init__(self, data_path, sequence_length=50, test_size=0.2, val_size=0.1, scaler_type='standard')
    def load_task31_datasets(self) -> Dict[str, pd.DataFrame]
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]
    def get_feature_dim(self) -> int
```

### Model Creation
```python
def create_model(model_type: str, input_dim: int, **kwargs) -> nn.Module
```

### RULTrainer
```python
class RULTrainer:
    def __init__(self, model, device='auto', learning_rate=0.001, ...)
    def train(self, train_loader, val_loader, epochs=100, ...) -> Dict
    def save_checkpoint(self, epoch, is_best=False)
    def load_checkpoint(self, checkpoint_path)
```

### RULEvaluator
```python
class RULEvaluator:
    def __init__(self, model, device='auto', save_dir='./evaluation_results')
    def evaluate(self, test_loader, detailed_analysis=True) -> Dict
    def compare_models(self, model_results) -> pd.DataFrame
```

## Contributing

To extend the system:

1. **Add new model architectures** in `models.py`
2. **Implement custom metrics** in `evaluator.py`
3. **Add data preprocessing** in `data_loader.py`
4. **Extend training features** in `trainer.py`

## License

This RUL prediction system is part of the i-drill project.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files in the logs directory
3. Examine configuration settings
4. Verify data format and quality