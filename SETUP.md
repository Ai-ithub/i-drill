# Drilling Data Processing System - Setup Guide

## Overview
This system processes real-time drilling sensor data using Kafka for streaming, PostgreSQL for storage, and includes DVR (Data Validation and Reconciliation) capabilities along with predictive maintenance features.

## Prerequisites

### Software Requirements
- Python 3.11 or higher (recommended: 3.11+)
- PostgreSQL 12 or higher
- Apache Kafka 2.8 or higher
- Git

### System Requirements
- Minimum 4GB RAM
- 10GB free disk space
- Network connectivity for Kafka and database

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd i-drill
```

### 2. Create Virtual Environment
```bash
python -m venv drilling_env
# On Windows:
drilling_env\Scripts\activate
# On Linux/Mac:
source drilling_env/bin/activate
```

### 3. Install Dependencies
Choose the install that matches your workflow:

- **Full stack (backend + ML/RL tooling):**
  ```bash
  pip install -r requirements.txt
  ```
- **Backend service only:**
  ```bash
  pip install -r src/backend/requirements_backend.txt
  ```
- **ML/RUL experimentation only:**
  ```bash
  pip install -r requirements/ml.txt
  ```

### 4. Setup PostgreSQL Database
1. Install PostgreSQL and create a database:
```sql
CREATE DATABASE drilling_data;
CREATE USER drilling_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE drilling_data TO drilling_user;
```

2. The system will automatically create the required tables on first run.

### 5. Setup Apache Kafka
1. Download and start Kafka:
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka Server
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic drilling-data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### 6. Configure Environment Variables
1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` file with your configurations:
```
DB_PASSWORD=your_database_password
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### 7. Update Configuration
Edit `config.yaml` to match your environment:
```yaml
kafka:
  bootstrap_servers: "localhost:9092"
  topic: "drilling-data"
  
database:
  host: "localhost"
  port: 5432
  database: "drilling_data"
  user: "drilling_user"
  # password loaded from environment variable
  
logging:
  level: "INFO"
```

## Running the System

### Option 1: Run Complete System
```bash
cd src/backend
python main.py
```
This starts both Producer and Consumer in separate threads.

### Option 2: Run Components Separately

#### Start Producer (Data Generation)
```bash
cd src/backend
python Producer.py
```

#### Start Consumer (Data Processing)
```bash
cd src/backend
python Consumer.py
```

## System Components

### Producer (`Producer.py`)
- Generates simulated drilling sensor data
- Sends data to Kafka topic
- Configurable simulation parameters

### Consumer (`Consumer.py`)
- Consumes data from Kafka
- Processes data through DVR system
- Stores processed data to PostgreSQL

### DVR System (`src/processing/`)
- **dvr_controller.py**: Main processing logic
- **dvr_stats.py**: Statistical analysis and anomaly detection
- **dvr_reconciliation.py**: Data reconciliation and database storage

### Database Manager (`database_manager.py`)
- Handles PostgreSQL connections
- Manages sensor data storage
- Provides data retrieval methods

### Configuration (`config_loader.py`)
- Loads configuration from YAML and environment variables
- Provides centralized configuration management

## Monitoring and Logs

### Log Files
Logs are written to console with configurable levels:
- INFO: General system information
- WARNING: Non-critical issues
- ERROR: Critical errors requiring attention

### Database Monitoring
Check data insertion:
```sql
SELECT COUNT(*) FROM sensor_data;
SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 10;
```

### Kafka Monitoring
Check topic messages:
```bash
bin/kafka-console-consumer.sh --topic drilling-data --from-beginning --bootstrap-server localhost:9092
```

## Troubleshooting

### Common Issues

1. **Kafka Connection Error**
   - Ensure Kafka is running
   - Check `bootstrap_servers` configuration
   - Verify topic exists

2. **Database Connection Error**
   - Verify PostgreSQL is running
   - Check database credentials in `.env`
   - Ensure database and user exist

3. **Import Errors**
   - Activate virtual environment
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python path

4. **Permission Errors**
   - Ensure proper file permissions
   - Check database user privileges

### Performance Tuning

1. **Kafka Settings**
   - Adjust `batch.size` for throughput
   - Configure `linger.ms` for latency
   - Set appropriate `buffer.memory`

2. **Database Settings**
   - Configure connection pool size
   - Adjust `max_connections` in PostgreSQL
   - Consider indexing on timestamp and rig_id

3. **System Resources**
   - Monitor CPU and memory usage
   - Adjust thread counts if needed
   - Consider scaling horizontally

## Development

### Running Tests
```bash
cd tests
python -m pytest
```

### Code Structure
```
src/
├── backend/           # Kafka producers/consumers
├── processing/        # DVR system
├── predictive_maintenance/  # ML models
├── rul_prediction/    # Remaining Useful Life prediction
└── model/            # Core models and utilities
```

### Adding New Features
1. Follow existing code patterns
2. Add appropriate logging
3. Update configuration if needed
4. Add tests for new functionality
5. Update documentation

## Support
For issues and questions, refer to the project documentation or create an issue in the repository.