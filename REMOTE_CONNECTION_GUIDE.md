# Remote Connection Setup Guide

This guide explains how to configure the i-Drill application to connect to remote services instead of using local Docker containers.

## Overview

By default, the application uses Docker Compose to run all services locally. This guide shows you how to:
- Connect to remote PostgreSQL database
- Connect to remote Kafka cluster
- Connect to remote Redis instance
- Connect to remote MLflow server

## Quick Start

### Option 1: Using Environment Variables (Recommended)

1. **Copy the remote configuration template:**
   ```bash
   cd src/backend
   cp config.env.remote.example .env
   ```

2. **Edit `.env` file** with your remote service details:
   ```env
   DATABASE_URL=postgresql://user:password@remote-db.example.com:5432/drilling_db
   KAFKA_BOOTSTRAP_SERVERS=remote-kafka.example.com:9092
   REDIS_HOST=remote-redis.example.com
   REDIS_PORT=6379
   REDIS_PASSWORD=your_redis_password
   MLFLOW_TRACKING_URI=http://remote-mlflow.example.com:5000
   ```

3. **Run the application:**
   ```bash
   # Using Docker Compose with remote services
   docker-compose -f docker-compose.remote.yml up
   
   # Or run directly with Python
   cd src/backend
   uvicorn app:app --host 0.0.0.0 --port 8001
   ```

### Option 2: Using YAML Configuration

1. **Copy the remote Kafka config:**
   ```bash
   cp config/kafka_config.remote.yaml config/kafka_config.yaml
   ```

2. **Edit `config/kafka_config.yaml`** with your remote service details

3. **Set environment variables:**
   ```bash
   export DB_PASSWORD=your_database_password
   export KAFKA_USERNAME=your_kafka_username  # if needed
   export KAFKA_PASSWORD=your_kafka_password  # if needed
   ```

## Service-Specific Configuration

### PostgreSQL Database

**Environment Variable:**
```env
DATABASE_URL=postgresql://username:password@host:port/database
```

**Example:**
```env
DATABASE_URL=postgresql://drill_user:secure_password@db.example.com:5432/drilling_db
```

**SSL Connection (if required):**
Add SSL parameters to the connection string:
```env
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### Kafka

**Basic Configuration:**
```env
KAFKA_BOOTSTRAP_SERVERS=remote-kafka.example.com:9092
```

**Multiple Brokers:**
```env
KAFKA_BOOTSTRAP_SERVERS=kafka1.example.com:9092,kafka2.example.com:9092,kafka3.example.com:9092
```

**SASL Authentication:**
```env
KAFKA_BOOTSTRAP_SERVERS=remote-kafka.example.com:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=PLAIN
KAFKA_SASL_USERNAME=your_username
KAFKA_SASL_PASSWORD=your_password
```

**SSL/TLS (without SASL):**
```env
KAFKA_BOOTSTRAP_SERVERS=remote-kafka.example.com:9093
KAFKA_SECURITY_PROTOCOL=SSL
KAFKA_SSL_CA_LOCATION=/path/to/ca-cert
KAFKA_SSL_CERT_LOCATION=/path/to/client-cert
KAFKA_SSL_KEY_LOCATION=/path/to/client-key
```

### Redis

**Basic Configuration:**
```env
REDIS_HOST=remote-redis.example.com
REDIS_PORT=6379
REDIS_DB=0
```

**With Password:**
```env
REDIS_HOST=remote-redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
```

**Using Redis URL:**
```env
REDIS_URL=redis://:password@remote-redis.example.com:6379/0
```

**SSL/TLS Connection:**
```env
REDIS_URL=rediss://:password@remote-redis.example.com:6380/0
```

### MLflow

**Basic Configuration:**
```env
MLFLOW_TRACKING_URI=http://remote-mlflow.example.com:5000
```

**HTTPS:**
```env
MLFLOW_TRACKING_URI=https://remote-mlflow.example.com:5000
```

**With Authentication:**
```env
MLFLOW_TRACKING_URI=http://remote-mlflow.example.com:5000
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
```

## Testing Remote Connections

### Test Database Connection
```bash
python -c "from src.backend.database import init_database; init_database()"
```

### Test Kafka Connection
```bash
python -c "from kafka import KafkaProducer; p = KafkaProducer(bootstrap_servers='remote-kafka.example.com:9092'); print('Connected!')"
```

### Test Redis Connection
```bash
python -c "import redis; r = redis.Redis(host='remote-redis.example.com', port=6379); print(r.ping())"
```

### Test MLflow Connection
```bash
python -c "import mlflow; mlflow.set_tracking_uri('http://remote-mlflow.example.com:5000'); print('Connected!')"
```

## Common Cloud Provider Examples

### AWS

**RDS PostgreSQL:**
```env
DATABASE_URL=postgresql://username:password@your-db.region.rds.amazonaws.com:5432/drilling_db
```

**ElastiCache Redis:**
```env
REDIS_HOST=your-cluster.region.cache.amazonaws.com
REDIS_PORT=6379
```

**MSK (Managed Kafka):**
```env
KAFKA_BOOTSTRAP_SERVERS=b-1.your-cluster.region.kafka.amazonaws.com:9092,b-2.your-cluster.region.kafka.amazonaws.com:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=AWS_MSK_IAM
```

### Google Cloud

**Cloud SQL PostgreSQL:**
```env
DATABASE_URL=postgresql://username:password@/drilling_db?host=/cloudsql/project:region:instance
```

**Memorystore Redis:**
```env
REDIS_HOST=your-instance-ip
REDIS_PORT=6379
```

### Azure

**Azure Database for PostgreSQL:**
```env
DATABASE_URL=postgresql://username@server-name:password@server-name.postgres.database.azure.com:5432/drilling_db?sslmode=require
```

**Azure Cache for Redis:**
```env
REDIS_HOST=your-cache.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your_access_key
REDIS_SSL=true
```

## Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use strong passwords** for all remote services
3. **Enable SSL/TLS** for all connections when possible
4. **Use environment-specific secrets management** (AWS Secrets Manager, Azure Key Vault, etc.)
5. **Restrict network access** using firewall rules and security groups
6. **Rotate credentials regularly**

## Troubleshooting

### Connection Timeouts

If you experience connection timeouts:
- Check firewall rules and security groups
- Verify network connectivity: `ping remote-host.example.com`
- Increase timeout values in configuration
- Check if services are accessible from your network

### Authentication Failures

- Verify credentials are correct
- Check if authentication is required (some services don't require auth in dev)
- Ensure SSL certificates are valid and properly configured
- Check service logs for authentication errors

### SSL/TLS Issues

- Verify certificate paths are correct
- Ensure certificates are not expired
- Check certificate chain is complete
- For self-signed certificates, you may need to disable verification (development only)

## Migration from Local to Remote

1. **Backup local data** (if needed):
   ```bash
   pg_dump -h localhost -U drill_user drilling_db > backup.sql
   ```

2. **Restore to remote database**:
   ```bash
   psql -h remote-db.example.com -U username -d drilling_db < backup.sql
   ```

3. **Update configuration** as described above

4. **Test connections** before switching production traffic

5. **Monitor logs** for any connection issues

## Support

For issues or questions:
- Check service provider documentation
- Review application logs: `tail -f drilling_system.log`
- Verify network connectivity and firewall rules
- Test connections individually using the test commands above

