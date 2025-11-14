# 🚀 i-Drill Backend - راهنمای راه‌اندازی بحرانی

## ✅ تغییرات انجام شده (Critical Priority)

### 1️⃣ **Pydantic Schemas** ✅
- **فایل:** `api/models/schemas.py`
- **محتوا:** تمام schema های API شامل:
  - SensorDataPoint, SensorDataResponse
  - PredictionRequest, PredictionResponse
  - MaintenanceAlert, MaintenanceSchedule
  - User, Token (Authentication)
  - WebSocketMessage
  - ErrorResponse

### 2️⃣ **Database Models (SQLAlchemy ORM)** ✅
- **فایل:** `api/models/database_models.py`
- **محتوا:** تمام table models شامل:
  - SensorData
  - MaintenanceAlertDB, MaintenanceScheduleDB
  - UserDB
  - RULPredictionDB
  - AnomalyDetectionDB
  - ModelVersionDB
  - WellProfileDB
  - DrillingSessionDB
  - SystemLogDB

### 3️⃣ **Database Connection Management** ✅
- **فایل:** `database.py`
- **قابلیت‌ها:**
  - Connection pooling
  - Session management
  - Context managers
  - Health checks
  - Automatic retry

### 4️⃣ **Data Service (کامل)** ✅
- **فایل:** `services/data_service.py`
- **متدهای پیاده‌سازی شده:**
  - `get_latest_sensor_data()` - دریافت داده‌های real-time
  - `get_historical_data()` - query داده‌های تاریخی
  - `get_time_series_aggregated()` - داده‌های aggregated برای charts
  - `get_analytics_summary()` - خلاصه آماری
  - `insert_sensor_data()` - درج تک رکورد
  - `bulk_insert_sensor_data()` - درج bulk
  - `get_maintenance_alerts()` - دریافت alerts
  - `create_maintenance_alert()` - ایجاد alert
  - `get_maintenance_schedules()` - دریافت schedules
  - `update_maintenance_schedule()` - آپدیت schedule
  - `save_rul_prediction()` - ذخیره RUL prediction
  - `get_rul_predictions()` - دریافت RUL history

### 5️⃣ **API Integration (یکپارچه‌سازی کامل)** ✅
- **فایل:** `app.py`
- **قابلیت‌ها:**
  - تمام routers متصل شده
  - CORS middleware
  - Error handling
  - Request logging
  - Lifespan events (startup/shutdown)
  - Health checks

### 6️⃣ **API Routes (کامل)** ✅
- `api/routes/health.py` - Health check endpoints
- `api/routes/sensor_data.py` - Sensor data & WebSocket
- `api/routes/predictions.py` - RUL & Anomaly detection
- `api/routes/maintenance.py` - Maintenance alerts & schedules
- `api/routes/producer.py` - Kafka producer endpoints
- `api/routes/config.py` - Configuration management

---

## 📋 پیش‌نیازها

### 1. نرم‌افزارهای مورد نیاز:
```bash
- Python 3.12+
- PostgreSQL 12+
- Kafka (optional)
- Redis (optional)
```

### 2. نصب وابستگی‌ها:
```bash
cd src/backend
pip install -r requirements_backend.txt
```

---

## 🔧 راه‌اندازی (به ترتیب)

### گام 1: تنظیمات محیطی
```bash
# کپی کردن فایل نمونه
cp config.env.example .env

# ویرایش فایل .env
nano .env
```

**تنظیمات مهم در .env:**
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/drilling_db
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
API_PORT=8001
```

### گام 2: راه‌اندازی PostgreSQL

#### روش 1: نصب مستقیم
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres psql
postgres=# CREATE DATABASE drilling_db;
postgres=# CREATE USER postgres WITH PASSWORD 'postgres';
postgres=# GRANT ALL PRIVILEGES ON DATABASE drilling_db TO postgres;
postgres=# \q
```

#### روش 2: Docker
```bash
docker run --name postgres-idrill \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=drilling_db \
  -p 5432:5432 \
  -d postgres:15
```

### گام 3: Setup Database
```bash
cd src/backend
python setup_backend.py
```

این اسکریپت:
- ✅ اتصال به database را تست می‌کند
- ✅ جداول را ایجاد می‌کند
- ✅ داده‌های نمونه می‌سازد (admin user, well profile)

### گام 4: راه‌اندازی Backend
```bash
# روش 1: مستقیم
python app.py

# روش 2: با uvicorn
uvicorn app:app --host 0.0.0.0 --port 8001 --reload

# روش 3: Production
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```

---

## 🧪 تست Backend

### 1. Health Check
```bash
curl http://localhost:8001/health
```

**پاسخ مورد انتظار:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "kafka": "unhealthy"
  }
}
```

### 2. API Documentation
مراجعه کنید به:
- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

### 3. تست Endpoints

#### Get Sensor Data
```bash
curl http://localhost:8001/api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10
```

#### Get Analytics
```bash
curl http://localhost:8001/api/v1/sensor-data/analytics/RIG_01
```

#### Create Maintenance Alert
```bash
curl -X POST http://localhost:8001/api/v1/maintenance/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "rig_id": "RIG_01",
    "component": "top_drive",
    "alert_type": "vibration_high",
    "severity": "warning",
    "message": "Vibration levels elevated"
  }'
```

---

## 🐛 عیب‌یابی (Troubleshooting)

### مشکل 1: Database Connection Failed
```bash
# بررسی وضعیت PostgreSQL
sudo systemctl status postgresql

# بررسی اتصال
psql -h localhost -U postgres -d drilling_db

# اگر خطای authentication داد:
sudo nano /etc/postgresql/*/main/pg_hba.conf
# تغییر peer به md5 و restart:
sudo systemctl restart postgresql
```

### مشکل 2: Import Errors
```bash
# اطمینان از نصب dependencies
pip install -r requirements_backend.txt

# بررسی PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### مشکل 3: Port Already in Use
```bash
# پیدا کردن process
lsof -i :8001

# Kill process
kill -9 <PID>
```

### مشکل 4: Kafka Not Available
این طبیعی است! سیستم بدون Kafka هم کار می‌کند:
- Health status: `degraded` می‌شود
- Real-time streaming غیرفعال است
- بقیه قابلیت‌ها کار می‌کنند

---

## 📊 ساختار API

### Base URL
```
http://localhost:8001/api/v1
```

### Endpoints اصلی

#### Health & Status
- `GET /health` - Overall health check
- `GET /health/database` - Database health
- `GET /health/kafka` - Kafka health
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe

#### Sensor Data
- `GET /sensor-data/realtime` - Latest sensor readings
- `GET /sensor-data/historical` - Historical data query
- `GET /sensor-data/aggregated` - Aggregated time-series
- `GET /sensor-data/analytics/{rig_id}` - Analytics summary
- `POST /sensor-data/` - Insert sensor data
- `WS /sensor-data/ws/{rig_id}` - WebSocket stream

#### Predictions
- `POST /predictions/rul` - RUL prediction
- `POST /predictions/rul/auto` - Auto RUL (from DB)
- `POST /predictions/anomaly-detection` - Detect anomalies
- `GET /predictions/anomaly-detection/{rig_id}` - Anomaly history

#### Maintenance
- `GET /maintenance/alerts` - Get alerts
- `POST /maintenance/alerts` - Create alert
- `GET /maintenance/schedule` - Get schedules
- `PUT /maintenance/schedule/{id}` - Update schedule

#### Configuration
- `GET /config/well-profiles` - Get well profiles
- `GET /config/well-profiles/{well_id}` - Get specific profile
- `POST /config/well-profiles` - Create profile
- `PUT /config/well-profiles/{well_id}` - Update profile
- `DELETE /config/well-profiles/{well_id}` - Delete profile

#### Producer
- `POST /producer/sensor-data` - Send to Kafka
- `GET /producer/status` - Producer status

---

## 🔐 Authentication (آینده)

در حال حاضر authentication پیاده‌سازی نشده و تمام endpoints آزاد هستند.

برای پیاده‌سازی در آینده:
1. از JWT tokens استفاده شود
2. Role-based access control (RBAC)
3. OAuth2 با Password Flow

---

## 📈 Performance Tips

### 1. Database Indexing
جداول از قبل index دارند روی:
- `rig_id`
- `timestamp`
- `severity`
- `status`

### 2. Connection Pooling
تنظیمات فعلی:
- Pool size: 10
- Max overflow: 20
- Pool timeout: 30s

### 3. Query Optimization
- از pagination استفاده کنید (limit, offset)
- Time-range queries را محدود کنید
- از aggregated endpoints برای charts استفاده کنید

---

## 🚀 Production Deployment

### با Docker:
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

### با Docker Compose:
```yaml
services:
  backend:
    build: ./src/backend
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/drilling_db
    depends_on:
      - postgres
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=drilling_db
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

---

## 📞 کمک و پشتیبانی

- **Documentation:** http://localhost:8001/docs
- **Health Check:** http://localhost:8001/health
- **Logs:** بررسی کنید در terminal که backend در آن اجرا شده

---

## ✅ Checklist راه‌اندازی

- [ ] PostgreSQL نصب و راه‌اندازی شده
- [ ] Dependencies نصب شده (`pip install -r requirements_backend.txt`)
- [ ] فایل `.env` تنظیم شده
- [ ] Database setup اجرا شده (`python setup_backend.py`)
- [ ] Backend اجرا شده و healthy است
- [ ] Swagger UI قابل دسترسی است (http://localhost:8001/docs)
- [ ] Health endpoint پاسخ می‌دهد

---

**موفق باشید! 🎉**

