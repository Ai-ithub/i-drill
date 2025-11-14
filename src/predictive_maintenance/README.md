# Wellbore Image Generation System

A comprehensive GAN-based system for generating synthetic wellbore images using StyleGAN2 architecture. This system is designed for predictive maintenance applications in drilling operations.

## 🚀 Features

- **StyleGAN2 Architecture**: State-of-the-art generator and discriminator networks
- **Comprehensive Training Pipeline**: Complete training system with loss functions and optimization
- **Advanced Data Processing**: Sophisticated preprocessing and augmentation for wellbore images
- **Multiple Evaluation Metrics**: FID, IS, LPIPS, SSIM, PSNR for quality assessment
- **Flexible Configuration**: YAML-based configuration management
- **Easy-to-Use CLI**: Command-line interface for training and inference
- **Modular Design**: Well-structured codebase with utilities and helper functions

## 📁 Project Structure

```
src/predictive_maintenance/
├── main.py                 # Main entry point
├── train.py               # Training script
├── inference.py           # Inference module
├── config.py              # Configuration classes
├── utils.py               # Legacy utilities
├── requirements.txt       # Dependencies
├── config/                # Configuration files
│   ├── __init__.py
│   ├── training_config.py
│   └── inference_config.py
├── data/                  # Data processing
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocessing.py
├── gan/                   # GAN models
│   ├── __init__.py
│   ├── generator.py
│   ├── discriminator.py
│   └── losses.py
├── evaluation/            # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py
└── utils/                 # Utility modules
    ├── __init__.py
    ├── image_utils.py
    ├── model_utils.py
    ├── training_utils.py
    ├── evaluation_utils.py
    └── file_utils.py
```

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training

### Install Dependencies

```bash
cd src/predictive_maintenance
pip install -r requirements.txt
```

### Required Packages

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- Pillow >= 8.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-image >= 0.18.0
- opencv-python >= 4.5.0
- PyYAML >= 5.4.0
- tqdm >= 4.62.0
- tensorboard >= 2.7.0
- lpips >= 0.1.4 (optional, for LPIPS metric)

## 🚀 Quick Start

### 1. Create Configuration File

```bash
python main.py create-config --output config/my_config.yaml --type training
```

### 2. Prepare Your Dataset

Organize your wellbore images in the following structure:

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. Train the Model

```bash
python main.py train --config config/my_config.yaml
```

### 4. Generate Images

```bash
python main.py generate --model checkpoints/best_model.pth --num_images 50 --output generated/
```

### 5. Evaluate Model

```bash
python main.py evaluate --model checkpoints/best_model.pth --data_dir data/test --output evaluation/
```

## 📖 Detailed Usage

### Training

#### Basic Training

```bash
python main.py train --config config/training_config.yaml
```

#### Resume Training

```bash
python main.py train --config config/training_config.yaml --resume checkpoints/latest.pth
```

#### Training with Specific GPU

```bash
python main.py train --config config/training_config.yaml --gpu 0
```

#### Debug Mode

```bash
python main.py train --config config/training_config.yaml --debug
```

### Image Generation

#### Basic Generation

```bash
python main.py generate --model checkpoints/best_model.pth --num_images 100
```

#### Custom Output Directory and Format

```bash
python main.py generate --model checkpoints/best_model.pth --num_images 50 --output my_images/ --format jpg
```

#### Batch Generation

```bash
python main.py generate --model checkpoints/best_model.pth --num_images 1000 --batch_size 16
```

#### Reproducible Generation

```bash
python main.py generate --model checkpoints/best_model.pth --num_images 10 --seed 42
```

### Model Evaluation

#### Comprehensive Evaluation

```bash
python main.py evaluate --model checkpoints/best_model.pth --data_dir data/test --output evaluation/
```

#### Large-scale Evaluation

```bash
python main.py evaluate --model checkpoints/best_model.pth --data_dir data/test --num_samples 5000 --batch_size 64
```

## ⚙️ Configuration

### Training Configuration

Example `training_config.yaml`:

```yaml
model:
  generator:
    latent_dim: 512
    num_layers: 8
    hidden_dim: 512
    output_channels: 3
    image_size: 256
  
  discriminator:
    input_channels: 3
    hidden_dim: 64
    num_layers: 6
    image_size: 256

training:
  epochs: 1000
  batch_size: 16
  learning_rate: 0.0002
  beta1: 0.5
  beta2: 0.999
  device: cuda
  device_id: 0
  
  # Loss weights
  adversarial_weight: 1.0
  gradient_penalty_weight: 10.0
  
  # Training schedule
  discriminator_steps: 1
  generator_steps: 1
  
  # Checkpointing
  save_frequency: 100
  eval_frequency: 50
  
data:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: 256
  batch_size: 16
  num_workers: 4
  
  # Augmentation
  horizontal_flip: true
  rotation_range: 10
  brightness_range: 0.1
  contrast_range: 0.1

logging:
  log_dir: "logs"
  level: "INFO"
  tensorboard: true
  
checkpoints:
  save_dir: "checkpoints"
  keep_best: 5
  keep_latest: 3
```

### Inference Configuration

Example `inference_config.yaml`:

```yaml
generation:
  num_images: 100
  batch_size: 8
  output_dir: "generated"
  image_format: "png"
  seed: null
  
model:
  device: "cuda"
  device_id: 0
```

## 🧠 Model Architecture

### StyleGAN2 Generator

- **Latent Dimension**: 512
- **Progressive Growing**: Multi-resolution synthesis
- **Style Modulation**: AdaIN layers for style control
- **Noise Injection**: Stochastic variation
- **Skip Connections**: Enhanced gradient flow

### StyleGAN2 Discriminator

- **Progressive Discrimination**: Multi-scale analysis
- **Spectral Normalization**: Training stability
- **Gradient Penalty**: WGAN-GP loss
- **Feature Matching**: Additional regularization

## 📊 Evaluation Metrics

### Image Quality Metrics

- **FID (Fréchet Inception Distance)**: Measures distribution similarity
- **IS (Inception Score)**: Evaluates image quality and diversity
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity
- **SSIM (Structural Similarity Index)**: Structural similarity
- **PSNR (Peak Signal-to-Noise Ratio)**: Pixel-level similarity

### Diversity Metrics

- **Mode Collapse Detection**: Identifies lack of diversity
- **Diversity Score**: Measures inter-image distances

## 🔧 Advanced Usage

### Custom Dataset

```python
from src.predictive_maintenance.data import WellboreDataset
from torch.utils.data import DataLoader

# Create custom dataset
dataset = WellboreDataset(
    data_dir="path/to/images",
    image_size=256,
    augment=True
)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
```

### Custom Training Loop

```python
from src.predictive_maintenance.train import GANTrainer
from src.predictive_maintenance.config import TrainingConfig

# Load configuration
config = TrainingConfig.from_yaml("config/training_config.yaml")

# Create trainer
trainer = GANTrainer(config)

# Custom training
for epoch in range(config.training.epochs):
    trainer.train_epoch(epoch)
    
    if epoch % config.training.eval_frequency == 0:
        trainer.evaluate(epoch)
    
    if epoch % config.training.save_frequency == 0:
        trainer.save_checkpoint(epoch)
```

### Custom Inference

```python
from src.predictive_maintenance.inference import GANInference
from src.predictive_maintenance.config import InferenceConfig

# Load configuration
config = InferenceConfig.from_yaml("config/inference_config.yaml")

# Create inference engine
inference = GANInference(config)

# Load model
inference.load_model("checkpoints/best_model.pth")

# Generate images
images = inference.generate_batch(batch_size=10)

# Save images
for i, image in enumerate(images):
    inference.save_image(image, f"generated/image_{i:04d}.png")
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory

- Reduce batch size in configuration
- Use gradient accumulation
- Enable mixed precision training

```yaml
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 2
  mixed_precision: true
```

#### Training Instability

- Adjust learning rates
- Increase gradient penalty weight
- Use spectral normalization

```yaml
training:
  learning_rate: 0.0001  # Reduce learning rate
  gradient_penalty_weight: 15.0  # Increase penalty
```

#### Poor Image Quality

- Increase model capacity
- Improve data quality
- Adjust loss weights

```yaml
model:
  generator:
    hidden_dim: 1024  # Increase capacity
    num_layers: 10
```

### Performance Optimization

#### Multi-GPU Training

```python
# In training configuration
training:
  multi_gpu: true
  gpu_ids: [0, 1, 2, 3]
```

#### Data Loading Optimization

```yaml
data:
  num_workers: 8  # Increase workers
  pin_memory: true
  prefetch_factor: 2
```

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Key Metrics to Monitor

- **Generator Loss**: Should decrease over time
- **Discriminator Loss**: Should remain stable
- **FID Score**: Lower is better
- **Inception Score**: Higher is better
- **Generated Samples**: Visual quality assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- StyleGAN2 paper and implementation
- PyTorch team for the framework
- Contributors to evaluation metrics libraries

## 📞 Support

For questions and support:

1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

## 🔄 Version History

### v1.0.0
- Initial release
- StyleGAN2 implementation
- Complete training pipeline
- Evaluation metrics
- CLI interface

---

**Happy generating! 🎨**