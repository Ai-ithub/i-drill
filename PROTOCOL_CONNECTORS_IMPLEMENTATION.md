# پیاده‌سازی Protocol Connectors برای دکل 1000 اسب‌بخار

## خلاصه

این سند خلاصه‌ای از پیاده‌سازی نیازمندی‌های اتصال به سیستم‌های کنترل دکل (SCADA/PLC) است که در فایل `نیازمندی‌های_استقرار_دکل_1000_اسب_بخار.md` تعریف شده‌اند.

## نیازمندی‌های پیاده‌سازی شده

### ✅ 1. پروتکل Modbus RTU/TCP

**فایل:** `src/backend/services/protocol_connectors/modbus_connector.py`

**ویژگی‌ها:**
- پشتیبانی از Modbus TCP برای سنسورهای اصلی
- پشتیبانی از Modbus RTU برای سنسورهای قدیمی‌تر
- Mapping پارامترهای دکل به مدل داده سیستم
- اتصال خودکار و reconnect
- خواندن مداوم داده‌ها با interval قابل تنظیم

**استفاده:**
```python
from services.protocol_connectors import ModbusConnector, ModbusProtocol

# Modbus TCP
connector = ModbusConnector(
    protocol=ModbusProtocol.TCP,
    host="192.168.1.100",
    port=502,
    unit_id=1,
    rig_id="rig-001"
)
connector.connect()
connector.start_continuous_reading("rig-001", interval=1.0)
```

### ✅ 2. پروتکل OPC UA

**فایل:** `src/backend/services/protocol_connectors/opcua_connector.py`

**ویژگی‌ها:**
- اتصال به سرور OPC UA دکل
- خواندن داده‌های real-time از tags تعریف شده
- مدیریت اتصال و reconnect خودکار
- پشتیبانی از subscription-based updates (توصیه می‌شود)
- پشتیبانی از polling-based reading

**استفاده:**
```python
from services.protocol_connectors import OPCUAConnector

connector = OPCUAConnector(
    endpoint_url="opc.tcp://192.168.1.100:4840",
    rig_id="rig-001"
)
connector.connect()
connector.start_subscription("rig-001")  # یا start_continuous_reading
```

### ✅ 3. پروتکل MQTT

**فایل:** `src/backend/services/protocol_connectors/mqtt_connector.py`

**ویژگی‌ها:**
- Subscribe به topics مربوط به سنسورهای دکل
- پشتیبانی از QoS levels مختلف (0, 1, 2)
- اتصال خودکار و reconnect
- پشتیبانی از wildcard topics (+ و #)
- Authentication (username/password)

**استفاده:**
```python
from services.protocol_connectors import MQTTConnector

connector = MQTTConnector(
    broker_host="192.168.1.100",
    broker_port=1883,
    topics=["rig/+/wob", "rig/+/rpm"],
    qos=1,
    rig_id="rig-001"
)
connector.connect()
connector.start()
```

### ✅ 4. Data Bridge Service بهبود یافته

**فایل:** `src/backend/services/data_bridge.py`

**ویژگی‌های جدید:**
- Buffer و queue management برای جلوگیری از از دست رفتن داده
- تبدیل فرمت داده‌های دکل به فرمت استاندارد سیستم
- اعتبارسنجی اولیه داده‌ها
- پردازش async داده‌ها از protocol connectors

**ویژگی‌های Queue:**
- Queue size قابل تنظیم (default: 1000)
- Buffer برای دسترسی سریع (default: 100)
- پردازش async در thread جداگانه
- جلوگیری از از دست رفتن داده در صورت overload

### ✅ 5. Protocol Adapter

**فایل:** `src/backend/services/protocol_adapter.py`

**ویژگی‌ها:**
- تبدیل فرمت داده‌های دکل به فرمت استاندارد سیستم
- Mapping پارامترهای مختلف به نام‌های استاندارد
- تبدیل واحدها (unit conversion)
- اعتبارسنجی اولیه داده‌ها
- پشتیبانی از timestamp formats مختلف

**پارامترهای پشتیبانی شده:**
- WOB (Weight on Bit)
- RPM (Rotary Speed)
- Torque
- ROP (Rate of Penetration)
- Mud Flow Rate
- Standpipe Pressure
- Casing Pressure
- Hook Load
- Block Position
- Pump Status
- Power Consumption
- Temperature (Bit, Motor, Surface)
- و سایر پارامترها

### ✅ 6. Protocol Manager

**فایل:** `src/backend/services/protocol_manager.py`

**ویژگی‌ها:**
- مدیریت متمرکز تمام protocol connectors
- ثبت و راه‌اندازی connectors
- مانیتورینگ وضعیت connectors
- توقف و حذف connectors

**استفاده:**
```python
from services.protocol_manager import protocol_manager

# ثبت Modbus connector
protocol_manager.register_modbus_connector(
    connector_id="modbus-rig-001",
    protocol=ModbusProtocol.TCP,
    rig_id="rig-001",
    host="192.168.1.100",
    port=502
)

# بررسی وضعیت
status = protocol_manager.get_connector_status("modbus-rig-001")
```

### ✅ 7. API Routes

**فایل:** `src/backend/api/routes/protocol_connectors.py`

**Endpoints:**
- `POST /api/v1/protocol-connectors/modbus` - ثبت Modbus connector
- `POST /api/v1/protocol-connectors/opcua` - ثبت OPC UA connector
- `POST /api/v1/protocol-connectors/mqtt` - ثبت MQTT connector
- `GET /api/v1/protocol-connectors/` - لیست تمام connectors
- `GET /api/v1/protocol-connectors/{connector_id}/status` - وضعیت connector
- `GET /api/v1/protocol-connectors/status/all` - وضعیت تمام connectors
- `DELETE /api/v1/protocol-connectors/{connector_id}` - توقف connector
- `GET /api/v1/protocol-connectors/data-bridge/queue-status` - وضعیت queue

## Dependencies

**فایل:** `requirements/backend.txt`

**پکیج‌های اضافه شده:**
- `pymodbus>=3.6.0,<4.0.0` - برای Modbus RTU/TCP
- `asyncua>=1.0.0,<2.0.0` - برای OPC UA
- `paho-mqtt>=2.0.0,<3.0.0` - برای MQTT

## ساختار فایل‌ها

```
src/backend/services/
├── protocol_connectors/
│   ├── __init__.py
│   ├── modbus_connector.py      # Modbus RTU/TCP connector
│   ├── opcua_connector.py       # OPC UA connector
│   └── mqtt_connector.py        # MQTT connector
├── protocol_adapter.py           # Data transformation
├── protocol_manager.py           # Connector management
└── data_bridge.py                # Enhanced with queue management

src/backend/api/routes/
└── protocol_connectors.py       # API endpoints
```

## مثال استفاده

### مثال 1: اتصال Modbus TCP

```python
from services.protocol_manager import protocol_manager
from services.protocol_connectors import ModbusProtocol

# ثبت connector
success = protocol_manager.register_modbus_connector(
    connector_id="modbus-rig-001",
    protocol=ModbusProtocol.TCP,
    rig_id="rig-001",
    host="192.168.1.100",
    port=502,
    unit_id=1,
    reading_interval=1.0
)
```

### مثال 2: اتصال OPC UA

```python
from services.protocol_manager import protocol_manager

# ثبت connector
success = protocol_manager.register_opcua_connector(
    connector_id="opcua-rig-001",
    rig_id="rig-001",
    endpoint_url="opc.tcp://192.168.1.100:4840",
    use_subscription=True
)
```

### مثال 3: اتصال MQTT

```python
from services.protocol_manager import protocol_manager

# ثبت connector
success = protocol_manager.register_mqtt_connector(
    connector_id="mqtt-rig-001",
    rig_id="rig-001",
    broker_host="192.168.1.100",
    broker_port=1883,
    topics=["rig/rig-001/+"],
    qos=1
)
```

## Flow داده

```
Protocol Connector (Modbus/OPC UA/MQTT)
    ↓
Protocol Adapter (Data Transformation)
    ↓
Data Bridge Queue (Buffer Management)
    ↓
Data Bridge Processor (Async)
    ↓
Kafka Topic
    ↓
Data Bridge Consumer
    ↓
[DVR Processing] → [Database] → [WebSocket Broadcast]
```

## تنظیمات Environment Variables

```bash
# Data Bridge Queue Settings
DATA_BRIDGE_QUEUE_SIZE=1000        # Max queue size
DATA_BRIDGE_BUFFER_SIZE=100       # Buffer size

# Integration Settings
ENABLE_DVR_IN_BRIDGE=true          # Enable DVR processing
ENABLE_RL_IN_BRIDGE=false          # Enable RL integration
```

## نکات مهم

1. **Thread Safety:** تمام connectors از threading برای اجرای async استفاده می‌کنند
2. **Error Handling:** تمام connectors دارای error handling و reconnect logic هستند
3. **Resource Management:** connectors باید به درستی stop شوند تا منابع آزاد شوند
4. **Authentication:** OPC UA و MQTT از username/password پشتیبانی می‌کنند
5. **QoS Levels:** MQTT از QoS 0, 1, 2 پشتیبانی می‌کند

## تست

برای تست connectors:

```bash
# نصب dependencies
pip install -r requirements/backend.txt

# راه‌اندازی سرور
python src/backend/start_server.py

# تست API endpoints
curl -X POST http://localhost:8001/api/v1/protocol-connectors/modbus \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "connector_id": "test-modbus",
    "protocol": "tcp",
    "rig_id": "rig-001",
    "host": "192.168.1.100",
    "port": 502
  }'
```

## وضعیت پیاده‌سازی

✅ **تکمیل شده:**
- Modbus RTU/TCP connector
- OPC UA connector
- MQTT connector
- Protocol Adapter
- Data Bridge با queue management
- Protocol Manager
- API Routes
- Dependencies

## مراحل بعدی

1. تست integration با دکل واقعی
2. تنظیم register/tag mappings برای دکل خاص
3. بهینه‌سازی performance
4. اضافه کردن monitoring و metrics
5. مستندسازی کامل API

---

**تاریخ پیاده‌سازی:** 2025-01-27  
**وضعیت:** ✅ تکمیل شده

