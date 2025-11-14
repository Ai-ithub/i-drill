# Wellbore Image Generation with StyleGAN2

A comprehensive system for generating synthetic wellbore images using StyleGAN2 architecture. This project is designed for predictive maintenance applications in drilling operations, providing high-quality synthetic data for training and testing machine learning models.

## 🎯 Features

- **StyleGAN2 Implementation**: State-of-the-art generative adversarial network for high-quality image synthesis
- **Wellbore-Specific Dataset Handling**: Specialized data processing pipeline for wellbore imagery
- **Comprehensive Evaluation**: Multiple metrics including FID, IS, and LPIPS for quality assessment
- **Flexible Training Pipeline**: Configurable training with checkpointing and monitoring
- **Advanced Generation Techniques**: Style mixing, interpolation, and custom latent manipulation
- **Easy-to-Use Examples**: Complete examples for common use cases
- **Production Ready**: Robust error handling and logging

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd i-drill
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

### Basic Usage

1. **Prepare sample data**:
   ```bash
   cd src/predictive_maintenance
   python examples/data_preparation.py --action sample --output-dir data/wellbore_images --num-samples 1000
   ```

2. **Train a model**:
   ```bash
   python main.py --mode train --config configs/default.yaml
   ```

3. **Generate images**:
   ```bash
   python main.py --mode inference --model-path checkpoints/stylegan2_generator_latest.pth --output-dir outputs/generated
   ```

4. **Evaluate model**:
   ```bash
   python main.py --mode evaluate --model-path checkpoints/stylegan2_generator_latest.pth --real-data-path data/wellbore_images
   ```

## 📁 Project Structure

```
i-drill/
├── src/predictive_maintenance/
│   ├── main.py                 # Main entry point
│   ├── config.py              # Configuration classes
│   ├── gan.py                 # StyleGAN2 implementation
│   ├── train.py               # Training pipeline
│   ├── inference.py           # Image generation
│   ├── evaluation.py          # Model evaluation
│   ├── data.py                # Dataset handling
│   ├── utils.py               # Utility functions
│   └── examples/
│       ├── data_preparation.py # Data preprocessing
│       └── basic_usage.py      # Usage examples
├── configs/
│   └── default.yaml           # Default configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```
# **Software Requirements Specification (SRS)**  
**Project:** **Intelligent Drilling Rig Automation System with Single Well Synthetic Data Generation**  

## 🔧 Configuration

The system uses YAML configuration files for easy customization. Here's an example configuration:

```yaml
# Data settings
data_path: "data/wellbore_images"
image_size: 256
batch_size: 16

# Model settings
latent_dim: 512
num_mapping_layers: 8
style_dim: 512

# Training settings
num_epochs: 100
learning_rate: 0.002
beta1: 0.0
beta2: 0.99
gradient_penalty_weight: 10.0

# Output settings
output_dir: "outputs"
checkpoint_dir: "checkpoints"
sample_dir: "samples"
### **1.1 Purpose**  
This document outlines the comprehensive requirements for an **Intelligent Drilling Automation System** with **synthetic data generation capabilities** for **1 well** simulating **6 months of drilling operations** with **1-second timestep resolution**. The system integrates **real-time monitoring, AI-driven optimization, predictive maintenance, Data Validation & Reconciliation (DVR), and full MLOps/DevOps capabilities** with **synthetic LWD/MWD data generation**.

### **1.2 Scope**  
The system includes:  
- **Synthetic LWD/MWD data generation** for 1 well with 6-month duration at 1-second intervals  
- **Real-time sensor monitoring** (WOB, RPM, torque, mud flow, pressure, gamma ray, resistivity, density)  
- **AI-driven optimization** (automated parameter tuning for drilling efficiency)  
- **Predictive maintenance** (failure forecasting, RUL estimation)  
- **Data Validation & Reconciliation (DVR)** (error detection, data correction)  
- **Kafka-based stream processing** (scalable real-time analytics)  
- **MLOps pipeline** (model training, deployment, monitoring, retraining)  
- **DevOps infrastructure** (CI/CD, containerization, monitoring)  
- **Comprehensive testing strategy** (unit, integration, performance testing)  
- **Unified React.js dashboard** for all user roles with responsive design  

### **1.3 Synthetic Data Generation Specifications**  

**Data Generation Scope:**
- **Number of Wells:** 1 comprehensive well profile
- **Duration:** 6 months continuous operation
- **Timestep Resolution:** 1-second intervals
- **Total Data Points:** ~15.5 million records
- **Daily Data Points:** ~86,400 records
- **Hourly Data Points:** ~3,600 records

**LWD/MWD Data Parameters:**
| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| **WOB** | 5,000-50,000 | lbs | Weight on Bit |
| **RPM** | 50-200 | rpm | Rotary Speed |
| **Torque** | 5,000-20,000 | ft-lbs | Drill String Torque |
| **Mud Flow** | 500-1,200 | gpm | Mud Circulation Rate |
| **Standpipe Pressure** | 1,000-5,000 | psi | Pump Pressure |
| **Gamma Ray** | 20-150 | API | Formation Radioactivity |
| **Resistivity** | 0.2-200 | ohm-m | Formation Resistivity |
| **Density** | 1.5-3.0 | g/cc | Formation Density |
| **Porosity** | 5-25 | % | Formation Porosity |
| **ROP** | 10-100 | ft/hr | Rate of Penetration |
| **Hook Load** | 100,000-500,000 | lbs | String Weight |
| **Mud Temperature** | 40-80 | °C | Mud Return Temperature |
| **Vibration** | 0-10 | g | Drill String Vibration |

### **1.4 Definitions & Acronyms**  

| Term | Definition |  
|------|------------|  
| **LWD** | Logging While Drilling |  
| **MWD** | Measurement While Drilling |  
| **WOB** | Weight on Bit |  
| **RPM** | Rotations per Minute |  
| **ROP** | Rate of Penetration |  
| **DVR** | Data Validation & Reconciliation |  
| **RUL** | Remaining Useful Life |  
| **Kafka** | Apache Kafka (real-time data streaming) |  

# Logging
log_interval: 100
save_interval: 1000
sample_interval: 500
```

## 📊 Data Preparation

The system includes comprehensive data preparation tools:
### **2.1 System Overview**  
The system provides a comprehensive AI-driven automation platform with integrated synthetic data generation for a single well:  
✔ **Synthetic LWD/MWD data generator** for 1 well with detailed drilling scenarios  
✔ **Real-time drilling parameter monitoring and control**  
✔ **AI-driven optimization** (automated drilling parameter adjustments)  
✔ **Predictive maintenance** (equipment health monitoring and RUL prediction)  
✔ **Data quality assurance** (DVR for sensor reliability)  
✔ **Full MLOps lifecycle management**  
✔ **Robust DevOps practices** (CI/CD, infrastructure as code, monitoring)  
✔ **Unified React.js dashboard** with role-based views  

### Supported Actions

- **preprocess**: Resize and normalize images
- **analyze**: Generate dataset statistics
- **split**: Create train/validation/test splits
- **clean**: Remove corrupted or invalid images
- **sample**: Create synthetic sample dataset
| Feature | Description |  
|---------|------------|  
| **Single Well Data Generator** | Comprehensive physics-based LWD/MWD simulation for one well |  
| **Real-Time Monitoring** | Live visualization of drilling parameters with <500ms latency |  
| **Optimization Engine** | **Reinforcement Learning (RL)** for optimal drilling parameters |  
| **Predictive Maintenance** | **LSTM/Transformer** for RUL prediction & anomaly detection |  
| **Data Validation (DVR)** | **Statistical/ML-based error detection & correction** |  
| **Kafka Stream Processing** | Real-time data ingestion and processing at 1,800+ events/sec |  
| **MLOps Pipeline** | End-to-end model management with synthetic data validation |  

### Examples

```bash
# Create sample dataset
python examples/data_preparation.py --action sample --output-dir data/samples --num-samples 1000
| Role | Access Level | Dashboard View |  
|------|-------------|----------------|  
| **Rig Operator** | Real-time control and monitoring | **Operator View** - Real-time controls, data streams |  
| **Drilling Engineer** | Analytics, optimization, configuration | **Engineering View** - Advanced analytics, scenario testing |  
| **Data Scientist** | Model development and experimentation | **Data Science View** - Model performance, data experiments |  
| **Maintenance Team** | Equipment health monitoring | **Maintenance View** - RUL predictions, work orders |  

# Analyze existing dataset
python examples/data_preparation.py --action analyze --input-dir data/wellbore_images

# Clean dataset
python examples/data_preparation.py --action clean --input-dir data/raw --output-dir data/cleaned

# Create train/val/test splits
python examples/data_preparation.py --action split --input-dir data/cleaned --output-dir data/splits
```

## 🎨 Image Generation

The system supports various generation techniques:

### Basic Generation
```python
from inference import ImageGenerator
from config import GANConfig

# Load configuration and model
config = GANConfig(latent_dim=512, image_size=256)
generator = ImageGenerator(config, "path/to/model.pth")

# Generate random images
images = generator.generate_random(num_images=16)
generator.save_images(images, "output/dir", prefix="random")
```
### **3.1 Single Well Synthetic Data Generation**  
- **FR-01:** **Comprehensive well profile** with detailed geological formation layers
- **FR-02:** **Physics-based drilling simulation** with realistic ROP, torque, and pressure responses
- **FR-03:** **Formation property generation** with realistic stratigraphy and lithology changes
- **FR-04:** **Equipment failure simulation** with progressive degradation patterns for all major components
- **FR-05:** **Drilling event simulation** (stick-slip, whirl, lost circulation, gas influx, wellbore instability)
- **FR-06:** **Real-time data streaming** at 1-second intervals with configurable noise levels
- **FR-07:** **Multiple data export formats** (CSV, Parquet, JSON, real-time Kafka streams)

### **3.2 Real-Time Monitoring Dashboard**  
- **FR-08:** **Unified React.js dashboard** with comprehensive data visualization
- **FR-09:** Display **all LWD/MWD parameters** in ≤ **500ms latency** from synthetic stream
- **FR-10:** **Interactive depth-based charts** with formation visualization
- **FR-11:** **Real-time data persistence** with configurable retention policies
- **FR-12:** **Historical data replay** capability for training and analysis

### **3.3 AI-Driven Optimization**  
- **FR-13:** **Reinforcement Learning (PPO/SAC)** trained on comprehensive synthetic drilling scenarios
- **FR-14:** **Digital Twin integration** using detailed well model
- **FR-15:** **Auto-adjustment of drilling parameters** based on real-time formation responses
- **FR-16:** **Optimization recommendations** with confidence scores and impact analysis
- **FR-17:** **Safety constraint enforcement** with automatic parameter limits

### **3.4 Predictive Maintenance**  
- **FR-18:** **LSTM/Transformer-based RUL prediction** for top drive, mud pumps, drawworks
- **FR-19:** **Real-time anomaly detection** on drilling dysfunction patterns
- **FR-20:** **Maintenance scheduling** based on equipment health forecasts
- **FR-21:** **Failure mode simulation** for comprehensive training of predictive models
- **FR-22:** **Spare parts optimization** based on predicted maintenance needs

### **3.5 Data Validation & Reconciliation (DVR)**  
- **FR-23:** **Statistical quality checks** on all data streams
- **FR-24:** **ML-based imputation** for simulated sensor failures and data gaps
- **FR-25:** **Real-time data quality scoring** for each sensor stream
- **FR-26:** **Automated calibration detection** and correction
- **FR-27:** **Data reconciliation reports** with audit trail

### **3.6 Kafka Stream Processing**  
- **FR-28:** **Ingest 1,800+ synthetic sensor readings/sec** continuously
- **FR-29:** **Real-time aggregation & filtering** of drilling data
- **FR-30:** **Integration with ML models** for real-time inference
- **FR-31:** **Stream processing monitoring** with performance metrics

### **3.7 MLOps Pipeline**  
- **FR-32:** **Model version control** with dataset versioning
- **FR-33:** **Automated model training** on synthetic data
- **FR-34:** **Model deployment automation** with validation
- **FR-35:** **Model performance monitoring** with drift detection
- **FR-36:** **Automated model retraining** based on performance metrics

### Advanced Techniques

- **Style Mixing**: Combine styles from different latent codes
- **Interpolation**: Create smooth transitions between images
- **Custom Latent Manipulation**: Fine-grained control over generation

## 📈 Evaluation Metrics

The system includes comprehensive evaluation metrics:

- **FID (Fréchet Inception Distance)**: Measures distribution similarity
- **IS (Inception Score)**: Evaluates image quality and diversity
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity

### Running Evaluation
### **4.1 Performance**  
- **≤ 500ms latency** for real-time data visualization
- **≤ 2 seconds** for ML model inference
- **Generate 1,800+ synthetic records/sec** continuously
- **Support 20+ concurrent users** with data access
- **99.9% uptime** for data generation services

### **4.2 Reliability**  
- **99.9% system uptime** for continuous data generation
- **Data loss < 0.1%** during stream processing
- **Graceful degradation** when services are overloaded
- **Automated failover** for critical components

### **4.3 Scalability**  
- **Containerized deployment** with resource optimization
- **Efficient memory usage** for single well data handling
- **Optimized storage** for time-series data
- **Horizontal scaling readiness** for future multi-well expansion

### **4.4 Data Quality**  
- **Realistic drilling physics** in data generation
- **Configurable noise levels** for sensor realism
- **Statistical validation** of data distributions
- **Physical consistency** across all parameters

```bash
python evaluation.py --model-path checkpoints/model.pth --real-data-path data/real --output-dir evaluation_results
```

## 🏋️ Training

### Training Configuration

Key training parameters:

- **Learning Rate**: Start with 0.002, adjust based on convergence
- **Batch Size**: Depends on GPU memory (16-32 recommended)
- **Gradient Penalty**: Weight for WGAN-GP loss (default: 10.0)
- **Progressive Growing**: Automatically handled by StyleGAN2
### **5.1 User Interfaces**  
- **Unified React.js Dashboard** - Data visualization and control
- **REST API** - For data access and configuration
- **WebSocket API** - Real-time data streaming
- **Configuration Interface** - Well profile and scenario setup

### **5.2 Software Interfaces**  
- **Kafka** (data streaming platform)
- **InfluxDB** (time-series data storage)
- **PostgreSQL** (well configuration, metadata)
- **Redis** (caching for data streams)
- **MLflow** (model management)

### Monitoring Training

The system provides comprehensive logging:

- **TensorBoard**: Real-time loss curves and sample images
- **Checkpoints**: Regular model saving for recovery
- **Sample Images**: Generated samples during training

### Training Tips

1. **Start Small**: Begin with lower resolution (128x128) for faster iteration
2. **Monitor Losses**: Both generator and discriminator should converge
3. **Check Samples**: Visual quality is as important as numerical metrics
4. **Use GPU**: CUDA-enabled GPU significantly speeds up training
### **6.1 Synthetic Data Generation Algorithms**  
| Algorithm | Use Case |  
|-----------|---------|  
| **Physics-based Drilling Models** | ROP, torque, pressure simulation |  
| **Formation Property Generators** | Gamma ray, resistivity, density sequences |  
| **Equipment Degradation Models** | Progressive failure simulation |  
| **Drilling Dynamics Models** | Vibration, stick-slip, whirl simulation |  

### **6.2 Optimization Algorithms**  
| Algorithm | Use Case |  
|-----------|---------|  
| **Reinforcement Learning** | Drilling parameter optimization |  
| **Bayesian Optimization** | Real-time parameter tuning |  

### **6.3 Predictive Maintenance Models**  
| Model | Use Case | Accuracy Target |  
|-------|---------|----------------|  
| **LSTM/Transformer** | RUL prediction | >90% accuracy |  
| **Isolation Forest** | Anomaly detection | <5% false positive rate |  

## 🔍 Examples

The `examples/` directory contains comprehensive usage examples:

### Basic Usage Examples

```bash
# Run all examples
python examples/basic_usage.py --example all

# Run specific example
python examples/basic_usage.py --example training
python examples/basic_usage.py --example inference
python examples/basic_usage.py --example evaluation

# Run complete pipeline demo
python examples/basic_usage.py --example pipeline
```

### Available Examples

1. **Training Example**: Complete training pipeline
2. **Inference Example**: Basic image generation
3. **Custom Generation**: Advanced generation techniques
4. **Evaluation Example**: Model quality assessment
5. **Data Analysis**: Dataset statistics and analysis
6. **Complete Pipeline**: End-to-end demonstration

## 🛠️ Advanced Usage

### Custom Dataset

To use your own dataset:

1. **Organize Data**: Place images in a single directory
2. **Preprocess**: Use data preparation tools
3. **Update Config**: Modify `data_path` in configuration
4. **Train**: Run training with custom data

### Model Customization

The StyleGAN2 implementation supports:

- **Custom Architecture**: Modify layer counts and dimensions
- **Different Resolutions**: Support for various image sizes
- **Transfer Learning**: Fine-tune from pretrained models

### Distributed Training

For large-scale training:

```bash
# Multi-GPU training (if available)
torchrun --nproc_per_node=2 main.py --mode train --config configs/distributed.yaml
```

## 📋 Requirements

### System Requirements

- **Python**: 3.12 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8GB+ RAM, 6GB+ GPU memory
- **Storage**: Sufficient space for datasets and checkpoints

### Key Dependencies

- **PyTorch**: ≥2.0.0 with CUDA support
- **torchvision**: ≥0.15.0
- **PIL/Pillow**: Image processing
- **NumPy/SciPy**: Numerical computing
- **Matplotlib**: Visualization
- **LPIPS**: Perceptual similarity metrics

See `requirements.txt` for complete dependency list.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in configuration
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Training Instability**:
   - Adjust learning rates
   - Check gradient penalty weight
   - Verify data preprocessing

3. **Poor Image Quality**:
   - Increase training epochs
   - Check dataset quality
   - Adjust model architecture

### Debug Mode

Enable verbose logging for debugging:

```bash
python main.py --mode train --config configs/default.yaml --verbose
```

## 📊 Performance Benchmarks

### Training Performance

| Resolution | Batch Size | GPU Memory | Training Time (100 epochs) |
|------------|------------|------------|----------------------------|
| 128x128    | 32         | 6GB        | ~4 hours                   |
| 256x256    | 16         | 8GB        | ~8 hours                   |
| 512x512    | 8          | 12GB       | ~16 hours                  |

*Benchmarks on NVIDIA RTX 3080*

### Quality Metrics

Typical quality metrics for well-trained models:

- **FID**: < 50 (excellent), < 100 (good)
- **IS**: > 3.0 (good diversity and quality)
- **LPIPS**: > 0.3 (good perceptual diversity)

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 conventions
2. **Documentation**: Update docstrings and README
3. **Testing**: Add tests for new features
4. **Issues**: Use GitHub issues for bug reports

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **StyleGAN2**: Based on the original StyleGAN2 paper by Karras et al.
- **PyTorch**: Built on the PyTorch deep learning framework
- **Community**: Thanks to the open-source community for tools and libraries

## 📞 Support

For questions and support:

1. **Documentation**: Check this README and code comments
2. **Examples**: Run the provided examples
3. **Issues**: Create GitHub issues for bugs
4. **Discussions**: Use GitHub discussions for questions

---

**Happy generating! 🎨**

*This project is designed to advance predictive maintenance in drilling operations through synthetic data generation.*
## **7. Implementation Phases**  

### **Phase 1: Core System (Months 1-3)**  
- Synthetic data generator development
- Basic LWD/MWD parameter simulation
- Kafka infrastructure setup
- React.js dashboard foundation

### **Phase 2: Advanced Features (Months 4-6)**  
- Physics-based drilling models
- Formation property generation
- Equipment failure simulation
- Advanced visualization

### **Phase 3: AI Integration (Months 7-9)**  
- ML model training on synthetic data
- Reinforcement Learning optimization
- Predictive maintenance implementation
- MLOps pipeline setup

### **Phase 4: Production Ready (Months 10-12)**  
- Performance optimization
- Comprehensive testing
- User acceptance testing
- Documentation completion

---

## **8. Single Well Specifications**  

### **8.1 Well Profile Configuration**  
**Comprehensive Well Type: Directional Development Well**
- **Total Depth:** 12,000 feet
- **Kick-off Point:** 2,000 feet
- **Build Rate:** 2-3°/100 feet
- **Maximum Inclination:** 45°
- **Target Zone:** 8,000-12,000 feet

### **8.2 Geological Formation Layers**  
| Depth (ft) | Formation | Lithology | Characteristics |
|------------|-----------|-----------|----------------|
| 0-2,000 | Surface | Sandstone/Shale | Unconsolidated, easy drilling |
| 2,000-5,000 | Intermediate | Limestone/Shale | Stable, moderate drilling |
| 5,000-8,000 | Target Zone | Dolomite | Hard, abrasive drilling |
| 8,000-12,000 | Reservoir | Porous Sandstone | Production zone, risk of lost circulation |

### **8.3 Data Volume Specifications**  
- **Total Records:** 15,552,000
- **Daily Records:** 86,400
- **Parameters per Record:** 15+ drilling parameters
- **Storage Requirement:** ~50 GB compressed
- **Streaming Rate:** 1,800 records/second

### **8.4 Data Quality Assurance**  
- **Physical Parameter Ranges:** All values within operational limits
- **Temporal Consistency:** Realistic time-series patterns
- **Formation Correlation:** Geologically accurate property sequences
- **Event Realism:** Physically plausible drilling events and dysfunctions

---

This **single well SRS** defines a **focused drilling automation system** with **comprehensive synthetic data generation** for **one well over 6 months**, providing **detailed data for robust AI/ML development** and **thorough system validation** with manageable data volume and complexity. 🚀
