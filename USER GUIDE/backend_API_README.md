# i-Drill API Documentation

## Overview

این API جامع برای مدیریت داده‌های حفاری، پیش‌بینی‌ها و مانیتورینگ Real-time طراحی شده است.

## ساختار پروژه

```
src/backend/
├── app.py                          # Main FastAPI application
├── config_loader.py                # Configuration management
├── database_manager.py             # Database operations
├── Producer.py                     # Kafka producer
├── Consumer.py                     # Kafka consumer
├── main.py                         # Producer/Consumer orchestrator
├── api/
│   ├── models/
│   │   ├── schemas.py              # Pydantic models
│   │   └── __init__.py
│   └── routes/
│       ├── sensor_data.py          # Sensor data endpoints
│       ├── predictions.py          # Prediction endpoints
│       ├── maintenance.py          # Maintenance endpoints
│       ├── config.py               # Configuration endpoints
│       ├── health.py               # Health check endpoints
│       └── __init__.py
└── services/
    ├── data_service.py             # Data operations
    ├── prediction_service.py       # ML predictions
    ├── kafka_service.py            # Kafka streaming
    └── __init__.py
```

## نصب و راه‌اندازی

### 1. نصب Dependencies

```bash
pip install -r requirements.txt
```

### 2. راه‌اندازی Docker Services

```bash
docker-compose up -d
```

این دستور سرویس‌های زیر را راه‌اندازی می‌کند:
- PostgreSQL
- Kafka
- Zookeeper

### 3. اجرای API Server

```bash
cd src/backend
python app.py
```

یا با uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. مشاهده Documentation

بعد از راه‌اندازی، Documentation در Swagger UI در دسترس است:
- http://localhost:8000/docs

## API Endpoints

### Health & Status

#### GET `/api/v1/health/`
بررسی وضعیت کلی API و سرویس‌ها

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "services": {
        "database": true,
        "kafka": true,
        "api": true
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

#### GET `/api/v1/health/services`
جزئیات وضعیت سرویس‌ها

#### GET `/api/v1/health/ready`
Readiness check

#### GET `/api/v1/health/live`
Liveness check

---

### Sensor Data

#### GET `/api/v1/sensor-data/realtime`
دریافت آخرین داده‌های سنسور

**Query Parameters:**
- `rig_id` (optional): فیلتر بر اساس Rig ID
- `limit` (default=100): تعداد رکوردها

**Example:**
```bash
curl http://localhost:8000/api/v1/sensor-data/realtime?rig_id=RIG_01&limit=50
```

#### GET `/api/v1/sensor-data/historical`
دریافت داده‌های تاریخی

**Query Parameters:**
- `rig_id` (optional)
- `start_time` (required): تاریخ شروع
- `end_time` (required): تاریخ پایان
- `parameters` (optional): لیست پارامترها (comma-separated)
- `limit` (default=1000)
- `offset` (default=0)

**Example:**
```bash
curl "http://localhost:8000/api/v1/sensor-data/historical?start_time=2025-01-01T00:00:00Z&end_time=2025-01-02T00:00:00Z"
```

#### GET `/api/v1/sensor-data/aggregated`
دریافت داده‌های تجمیع‌شده

**Query Parameters:**
- `rig_id` (required)
- `time_bucket_seconds` (default=60)
- `start_time` (optional)
- `end_time` (optional)

#### GET `/api/v1/sensor-data/analytics/{rig_id}`
دریافت خلاصه Analytics

**Example Response:**
```json
{
    "success": true,
    "summary": {
        "rig_id": "RIG_01",
        "total_drilling_time_hours": 720.5,
        "current_depth": 5234.5,
        "average_rop": 12.5,
        "total_power_consumption": 144000.0,
        "maintenance_alerts_count": 2,
        "last_updated": "2025-01-15T10:30:00Z"
    }
}
```

#### POST `/api/v1/sensor-data/`
ایجاد رکورد جدید

**Request Body:**
```json
{
    "timestamp": "2025-01-15T10:30:00Z",
    "rig_id": "RIG_01",
    "depth": 5234.5,
    "wob": 1520.3,
    "rpm": 82.1,
    "torque": 412.8,
    "rop": 12.5,
    ...
}
```

#### WebSocket `/api/v1/sensor-data/ws/{rig_id}`
اتصال Real-time Streaming

**Client Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/sensor-data/ws/RIG_01');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Sensor data:', data);
};
```

---

### Predictions

#### POST `/api/v1/predictions/rul`
پیش‌بینی RUL

**Request Body:**
```json
{
    "rig_id": "RIG_01",
    "sensor_data": [...],
    "model_type": "lstm",
    "lookback_window": 50
}
```

**Response:**
```json
{
    "success": true,
    "rig_id": "RIG_01",
    "predicted_rul_hours": 480.5,
    "confidence_score": 0.85,
    "prediction_timestamp": "2025-01-15T10:30:00Z",
    "model_used": "lstm"
}
```

#### POST `/api/v1/predictions/rul/auto`
پیش‌بینی خودکار RUL با استفاده از داده‌های دیتابیس

**Query Parameters:**
- `rig_id` (required)
- `lookback_hours` (default=24)
- `model_type` (default="lstm")

#### POST `/api/v1/predictions/anomaly-detection`
تشخیص آنومالی

**Request Body:**
```json
{
    "timestamp": "2025-01-15T10:30:00Z",
    "rig_id": "RIG_01",
    "bit_temperature": 105.0,
    "vibration_level": 2.5,
    ...
}
```

#### GET `/api/v1/predictions/anomaly-detection/{rig_id}`
تاریخچه تشخیص آنومالی

---

### Maintenance

#### GET `/api/v1/maintenance/alerts`
دریافت هشدارهای تعمیر و نگهداری

**Query Parameters:**
- `rig_id` (optional)
- `severity` (optional): low, medium, high, critical
- `hours` (default=24)

#### GET `/api/v1/maintenance/alerts/{alert_id}`
دریافت هشدار خاص

#### GET `/api/v1/maintenance/schedule`
دریافت برنامه تعمیر و نگهداری

#### POST `/api/v1/maintenance/schedule`
ایجاد برنامه تعمیرات

#### PUT `/api/v1/maintenance/schedule/{schedule_id}`
به‌روزرسانی برنامه

#### DELETE `/api/v1/maintenance/schedule/{schedule_id}`
حذف برنامه

---

### Configuration

#### GET `/api/v1/config/well-profiles`
دریافت Well Profiles

#### GET `/api/v1/config/well-profiles/{well_id}`
دریافت Well Profile خاص

#### POST `/api/v1/config/well-profiles`
ایجاد Well Profile

#### PUT `/api/v1/config/well-profiles/{well_id}`
به‌روزرسانی Well Profile

#### DELETE `/api/v1/config/well-profiles/{well_id}`
حذف Well Profile

#### GET `/api/v1/config/parameters`
دریافت پارامترهای پیکربندی

#### PUT `/api/v1/config/parameters`
به‌روزرسانی پارامترها

---

## Models Available

### Model Types for RUL Prediction

1. **LSTM**: Long Short-Term Memory network
2. **Transformer**: Transformer-based architecture
3. **CNN-LSTM**: Hybrid CNN-LSTM model

---

## Error Handling

API از کدهای HTTP استاندارد استفاده می‌کند:

- `200 OK`: موفقیت
- `400 Bad Request`: خطای ورودی
- `404 Not Found`: منبع یافت نشد
- `500 Internal Server Error`: خطای سرور
- `503 Service Unavailable`: سرویس در دسترس نیست

**Example Error Response:**
```json
{
    "detail": "Error message here"
}
```

---

## Security Considerations

### Production Deployment

1. **CORS**: تنظیم `allow_origins` در `app.py`
2. **Authentication**: افزودن JWT یا API Key
3. **Rate Limiting**: محدود کردن درخواست‌ها
4. **SSL/TLS**: استفاده از HTTPS

### Environment Variables

```bash
export DB_PASSWORD=your_secure_password
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

---

## Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Manual Testing

استفاده از Swagger UI:

1. باز کردن http://localhost:8000/docs
2. انتخاب Endpoint
3. تست با دکمه "Try it out"

---

## Performance Considerations

### Optimization Tips

1. **Pagination**: استفاده از `limit` و `offset` در queries
2. **Caching**: استفاده از Redis برای cache
3. **Connection Pooling**: تنظیم در `database_manager.py`
4. **Async Operations**: استفاده از async/await

---

## Logging

لاگ‌ها در سطح‌های زیر ثبت می‌شوند:

- **INFO**: عملیات عادی
- **WARNING**: هشدارها
- **ERROR**: خطاها
- **DEBUG**: جزئیات دیباگ

---

## Extension Points

### Adding New Endpoints

1. ایجاد Route در `api/routes/`
2. ایجاد Service در `services/`
3. افزودن Schema در `api/models/schemas.py`
4. اتصال Route در `app.py`

### Adding New Models

1. Train model در `src/rul_prediction/`
2. Save model به `models/`
3. Load در `prediction_service.py`

---

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - بررسی اتصال Docker
   - بررسی credentials در `config/kafka_config.yaml`

2. **Kafka Not Connected**
   - بررسی کافکا در Docker
   - بررسی bootstrap servers

3. **Import Errors**
   - بررسی `sys.path` configuration
   - بررسی relative imports

---

## Contributors

Development Team

---

## License

Proprietary

