# راهنمای یکپارچه‌سازی React Dashboard و WebSocket Server

## خلاصه کارهای انجام شده

### ✅ 1. ساختار React Dashboard

ایجاد شد در پوشه `frontend/` با:
- **Vite + React + TypeScript**: Setup مدرن با Vite
- **TailwindCSS**: برای استایل‌دهی
- **React Router**: برای Navigation
- **React Query**: برای مدیریت State و API calls
- **Recharts**: برای نمایش نمودارها

#### صفحات اصلی:
- `/` - Dashboard (نمای کلی)
- `/realtime` - مانیتورینگ Real-time
- `/historical` - داده‌های تاریخی
- `/predictions` - پیش‌بینی‌ها
- `/maintenance` - تعمیر و نگهداری

### ✅ 2. WebSocket Server بهبود یافته

#### بهبودهای انجام شده:
- **WebSocketManager**: مدیریت متمرکز اتصالات WebSocket
- **Connection Management**: ردیابی و مدیریت چندین اتصال
- **Ping/Pong**: Keep-alive برای اتصالات
- **Error Handling**: مدیریت خطاهای بهتر
- **Async Improvements**: استفاده از asyncio برای performance بهتر

#### فایل‌های جدید:
- `src/backend/services/websocket_manager.py`: مدیریت اتصالات
- بهبود `src/backend/api/routes/sensor_data.py`: WebSocket endpoint بهتر

### ✅ 3. یکپارچه‌سازی Producer/Consumer

#### Data Bridge Service:
ایجاد `src/backend/services/data_bridge.py` که:
- **Kafka Consumer**: مصرف پیام‌ها از Kafka
- **Database Storage**: ذخیره در دیتابیس
- **WebSocket Broadcasting**: ارسال به کلاینت‌های متصل
- **Thread-based**: اجرا در Thread جداگانه

#### Integration Flow:
```
Producer → Kafka Topic → Data Bridge → [Database + WebSocket]
```

#### API Endpoint جدید:
- `POST /api/v1/producer/sensor-data`: ارسال داده به Kafka
- `GET /api/v1/producer/status`: وضعیت Data Bridge

## نحوه استفاده

### 1. اجرای Backend

```bash
cd src/backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2. اجرای Producer (اختیاری)

اگر می‌خواهید Producer جداگانه اجرا شود:
```bash
python Producer.py
```

### 3. اجرای Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard در `http://localhost:3000` اجرا می‌شود.

### 4. اتصال WebSocket

Frontend به صورت خودکار به WebSocket endpoint متصل می‌شود:
- Endpoint: `ws://localhost:8000/api/v1/sensor-data/ws/{rig_id}`
- در صفحه Real-time Monitoring می‌توانید Rig ID را وارد کنید

## جریان داده

```
┌─────────────┐
│  Producer   │ ────┐
└─────────────┘     │
                    │
┌─────────────┐     │     ┌─────────────┐     ┌──────────────┐
│   Kafka     │ ◄───┘     │ Data Bridge │ ───►│  Database    │
│   Topic     │           └─────────────┘     └──────────────┘
└─────────────┘                 │
                                │
                                ▼
                         ┌─────────────┐
                         │ WebSocket  │ ────► React Dashboard
                         │  Manager   │
                         └─────────────┘
```

## Configuration

### Backend
- Kafka config در `config/kafka_config.yaml`
- Database config در `database_manager.py`

### Frontend
ایجاد فایل `.env` در `frontend/`:
```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/api/v1
```

## نکات مهم

1. **Data Bridge** به صورت خودکار با شروع Backend اجرا می‌شود
2. **WebSocket connections** به صورت خودکار مدیریت می‌شوند
3. **Producer** می‌تواند به صورت جداگانه یا از طریق API اجرا شود
4. تمام داده‌های Kafka به صورت خودکار به Database و WebSocket ارسال می‌شوند

## Testing

### Test WebSocket:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/sensor-data/ws/RIG_01');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

### Test Producer API:
```bash
curl -X POST http://localhost:8000/api/v1/producer/sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "rig_id": "RIG_01",
    "depth": 5000,
    "wob": 1500,
    "rpm": 80,
    "timestamp": "2025-01-15T10:30:00Z"
  }'
```

## ساختار فایل‌ها

```
project/
├── frontend/                    # React Dashboard
│   ├── src/
│   │   ├── components/         # کامپوننت‌های React
│   │   ├── pages/              # صفحات اصلی
│   │   ├── services/           # API & WebSocket services
│   │   └── types/              # TypeScript types
│   └── package.json
│
└── src/backend/
    ├── app.py                   # FastAPI main app
    ├── api/routes/
    │   ├── sensor_data.py      # Sensor & WebSocket routes
    │   └── producer.py         # Producer API routes
    └── services/
        ├── websocket_manager.py # WebSocket connection manager
        ├── data_bridge.py       # Producer/Consumer bridge
        └── kafka_service.py     # Kafka operations
```

## وضعیت فعلی

✅ React Dashboard: **کامل**
✅ WebSocket Server: **بهبود یافته**
✅ Producer/Consumer Integration: **یکپارچه شده**
✅ Real-time Data Flow: **فعال**

## مراحل بعدی (پیشنهادی)

1. تکمیل صفحات Historical Data و Predictions
2. اضافه کردن Authentication
3. بهبود Error Handling در Frontend
4. اضافه کردن Tests
5. Docker Compose برای تمام سرویس‌ها

