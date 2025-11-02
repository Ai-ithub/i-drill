# 🚀 Quick Start Guide

## اجرای سریع API

### 1️⃣ راه‌اندازی محیط

```bash
# نصب Dependencies
pip install -r requirements.txt

# راه‌اندازی Docker Services
docker-compose up -d
```

### 2️⃣ اجرای API Server

```bash
cd src/backend
python app.py
```

یا با uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3️⃣ دسترسی به Documentation

```
http://localhost:8000/docs
```

### 4️⃣ تست API

```bash
cd src/backend
python test_api.py
```

## 📌 Endpoints اصلی

### Health Check
```bash
curl http://localhost:8000/api/v1/health/
```

### Real-time Sensor Data
```bash
curl http://localhost:8000/api/v1/sensor-data/realtime?limit=10
```

### Historical Data
```bash
curl "http://localhost:8000/api/v1/sensor-data/historical?start_time=2025-01-01T00:00:00Z&end_time=2025-01-02T00:00:00Z&limit=100"
```

### RUL Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predictions/rul/auto?rig_id=RIG_01&lookback_hours=24
```

### Anomaly Detection
```bash
curl -X POST http://localhost:8000/api/v1/predictions/anomaly-detection \
  -H "Content-Type: application/json" \
  -d '{"rig_id":"RIG_01","bit_temperature":105.0,"vibration_level":2.5}'
```

## 🔍 منابع بیشتر

- [API_README.md](./API_README.md) - مستند کامل API
- [SUMMARY.md](./SUMMARY.md) - خلاصه پیاده‌سازی

## ⚠️ نکات مهم

1. مطمئن شوید Docker Services در حال اجرا هستند
2. Database باید جداول لازم را داشته باشد
3. برای RUL Prediction، مدل‌ها باید Train شده باشند

## 🆘 مشکلات متداول

**Database Connection Error**
- بررسی اجرای PostgreSQL در Docker

**Kafka Not Connected**
- بررسی اجرای Kafka و Zookeeper در Docker

**Import Errors**
- نصب Dependencies: `pip install -r requirements.txt`

