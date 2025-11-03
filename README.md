# 🛢️ i-Drill - Intelligent Drilling Automation System

> Real-time drilling intelligence with AI-driven optimization and predictive maintenance

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2%2B-61DAFB)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2%2B-3178C6)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

i-Drill is a comprehensive intelligent drilling automation platform that combines **real-time monitoring**, **AI-driven optimization**, and **predictive maintenance** for drilling operations. The system provides advanced analytics, automated parameter tuning, and equipment health monitoring through a modern, responsive dashboard.

### ✨ Key Features

- 🎛️ **Real-Time Monitoring** - Live visualization of drilling parameters with < 500ms latency
- 🤖 **AI-Driven Optimization** - Reinforcement learning for automated drilling parameter tuning
- 🔧 **Predictive Maintenance** - LSTM/Transformer models for equipment RUL prediction
- 📊 **Advanced Analytics** - Comprehensive data analysis and visualization
- 🌊 **Stream Processing** - Kafka-based real-time data ingestion (1,800+ events/sec)
- 📱 **Modern Dashboard** - React + TypeScript responsive UI with TURBIN design
- 🔐 **Security** - JWT authentication with role-based access control (RBAC)
- 🚀 **MLOps Ready** - MLflow integration for model lifecycle management
- 🐳 **Docker Support** - Containerized deployment with Docker Compose

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + TS)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Sensor  │ │  Control │ │   RPM    │ │  Gauge   │           │
│  │   Page   │ │   Page   │ │   Page   │ │   Page   │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API / WebSocket
┌───────────────────────────┴─────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │     Auth     │  │  Predictions │  │  Maintenance │          │
│  │   Service    │  │   Service    │  │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└───────────┬──────────────────┬───────────────────┬──────────────┘
            │                  │                   │
    ┌───────▼────────┐  ┌─────▼──────┐   ┌───────▼────────┐
    │   PostgreSQL   │  │   Kafka    │   │    MLflow      │
    │   Database     │  │  Streaming │   │    MLOps       │
    └────────────────┘  └────────────┘   └────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.8+ (for backend development)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ai-ithub/i-drill.git
   cd i-drill
   ```

2. **Start all services** using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8001
   - API Docs: http://localhost:8001/docs

### Manual Setup

#### Backend

```bash
cd src/backend
pip install -r requirements_backend.txt
python setup_backend.py
uvicorn app:app --reload --port 8001
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## 📊 Dashboard Pages

### 🎯 SENSOR Page
- Real-time noise signal visualization
- Distribution histogram
- FFT spectrum analysis
- Statistical metrics (Mean, Std, SNR, RMS)

### 🎮 Control Page
- RUN controls (START/PAUSE/STOP)
- Emergency stop button
- Threshold management for 6 parameters
- Runtime information display

### ⚡ RPM Page
- Circular gauges (RPM, Torque, Pressure)
- Linear gauges (WOB, ROP)
- Temperature monitoring
- Performance indicators
- Alarms & warnings panel

### 📈 Gauge Page
- Multiple gauge visualizations
- Real-time data updates
- Custom gauge configurations

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Database**: PostgreSQL + SQLAlchemy ORM
- **Authentication**: JWT + bcrypt
- **Streaming**: Apache Kafka
- **ML/AI**: PyTorch, Scikit-learn, MLflow
- **Monitoring**: Prometheus, Grafana

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **Charts**: Recharts
- **State**: Zustand
- **HTTP Client**: React Query + Axios
- **Icons**: Lucide React

### DevOps
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Orchestration**: Kubernetes-ready

## 📖 Documentation

- [Critical Setup Guide](src/backend/CRITICAL_SETUP_GUIDE.md) - Backend setup instructions
- [Start Here (فارسی)](START_HERE_FA.md) - Persian getting started guide
- [API Documentation](http://localhost:8001/docs) - Interactive API docs (when running)

## 🎨 Features in Detail

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- User management
- Secure password hashing with bcrypt

### Real-Time Monitoring
- WebSocket integration for live data
- Custom React hooks (`useWebSocket`)
- Sub-500ms latency
- Automatic reconnection

### Predictive Maintenance
- LSTM/Transformer models for RUL prediction
- Anomaly detection with Isolation Forest
- Maintenance scheduling
- Equipment health scoring

### MLOps Pipeline
- Model versioning with MLflow
- Experiment tracking
- Model registry
- Automated retraining

## 📈 Performance

- ⚡ < 500ms latency for real-time visualization
- 🚀 1,800+ events/second processing capability
- 📊 Support for 20+ concurrent users
- 🎯 99.9% uptime target
- 💾 Efficient time-series data storage

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Parsa** - Lead Developer & Architect

- GitHub: [@Ai-ithub](https://github.com/Ai-ithub)
- Project: [i-Drill](https://github.com/Ai-ithub/i-drill)

### Contributions
- Full-stack development (Backend + Frontend)
- System architecture and design
- AI/ML integration and MLOps
- DevOps and infrastructure setup
- Dashboard UI/UX implementation

See [AUTHORS.md](AUTHORS.md) for detailed contribution information.

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- React and TypeScript communities
- Apache Kafka for real-time streaming
- MLflow for MLOps capabilities
- All open-source contributors

## 📞 Support

For questions and support:
- 📧 Create an issue on GitHub
- 📚 Check the documentation
- 💬 Start a discussion

---

**© 2025 i-Drill Project - Developed by Parsa**

*"Innovation in Real-time Drilling Intelligence"* 🚀

