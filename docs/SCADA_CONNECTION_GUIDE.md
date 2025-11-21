# راهنمای اتصال به سیستم‌های SCADA/PLC

این راهنما نحوه اتصال سیستم i-Drill به سیستم‌های کنترل دکل (SCADA/PLC) را توضیح می‌دهد.

## پروتکل‌های پشتیبانی شده

سیستم i-Drill از سه پروتکل اصلی برای اتصال به سیستم‌های کنترل دکل پشتیبانی می‌کند:

### 1. Modbus RTU/TCP
- **Modbus TCP**: برای اتصال از طریق شبکه Ethernet
- **Modbus RTU**: برای اتصال از طریق پورت سریال (RS-485)

### 2. OPC UA
- برای سیستم‌های SCADA مدرن
- پشتیبانی از امنیت و احراز هویت
- خواندن و نوشتن tags

### 3. MQTT
- برای سنسورهای IoT
- پشتیبانی از QoS levels
- Event-driven (بدون نیاز به polling)

## نصب وابستگی‌ها

کتابخانه‌های مورد نیاز در `requirements/backend.txt` تعریف شده‌اند:

```bash
pip install pymodbus>=3.6.0
pip install asyncua>=1.0.0
pip install paho-mqtt>=2.0.0
```

یا نصب تمام وابستگی‌ها:

```bash
pip install -r requirements.txt
```

## پیکربندی

### 1. فعال‌سازی SCADA Connector

در فایل `.env` یا متغیرهای محیطی:

```bash
SCADA_CONNECTOR_ENABLED=true
```

### 2. پیکربندی از طریق API

#### مثال: پیکربندی Modbus TCP

```bash
curl -X POST "http://localhost:8001/api/v1/scada/rigs/RIG_1000HP/configure" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol": "modbus",
    "connection_config": {
      "protocol": "TCP",
      "host": "192.168.1.100",
      "port": 502,
      "slave_id": 1,
      "timeout": 3.0
    },
    "parameter_mapping": {
      "WOB": {
        "address": 0,
        "register_type": "holding",
        "data_type": "float32"
      },
      "RPM": {
        "address": 2,
        "register_type": "holding",
        "data_type": "uint16"
      },
      "Torque": {
        "address": 3,
        "register_type": "holding",
        "data_type": "float32"
      }
    },
    "read_interval": 1.0
  }'
```

#### مثال: پیکربندی OPC UA

```bash
curl -X POST "http://localhost:8001/api/v1/scada/rigs/RIG_1000HP/configure" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol": "opcua",
    "connection_config": {
      "endpoint_url": "opc.tcp://192.168.1.100:4840",
      "username": "admin",
      "password": "password",
      "security_policy": "None",
      "security_mode": "None",
      "timeout": 10.0
    },
    "parameter_mapping": {
      "WOB": "ns=2;s=WOB",
      "RPM": "ns=2;s=RPM",
      "Torque": "ns=2;s=Torque"
    },
    "read_interval": 1.0
  }'
```

#### مثال: پیکربندی MQTT

```bash
curl -X POST "http://localhost:8001/api/v1/scada/rigs/RIG_1000HP/configure" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "protocol": "mqtt",
    "connection_config": {
      "broker_host": "192.168.1.100",
      "broker_port": 1883,
      "username": "mqtt_user",
      "password": "mqtt_password",
      "client_id": "i-drill-rig-1000hp",
      "keepalive": 60,
      "clean_session": true,
      "tls_enabled": false,
      "qos": 1
    },
    "parameter_mapping": {
      "WOB": "rig/1000hp/sensors/WOB",
      "RPM": "rig/1000hp/sensors/RPM",
      "Torque": "rig/1000hp/sensors/Torque"
    },
    "read_interval": 1.0
  }'
```

## API Endpoints

### مدیریت Rig ها

- `POST /api/v1/scada/rigs/{rig_id}/configure` - پیکربندی یک rig
- `GET /api/v1/scada/rigs` - لیست تمام rig های پیکربندی شده
- `GET /api/v1/scada/rigs/{rig_id}/status` - وضعیت اتصال یک rig
- `DELETE /api/v1/scada/rigs/{rig_id}` - حذف پیکربندی یک rig

### مدیریت SCADA Connector

- `POST /api/v1/scada/start` - شروع SCADA connector
- `POST /api/v1/scada/stop` - توقف SCADA connector

### لیست اتصالات پروتکل‌ها

- `GET /api/v1/scada/protocols/modbus/connections` - لیست اتصالات Modbus
- `GET /api/v1/scada/protocols/opcua/connections` - لیست اتصالات OPC UA
- `GET /api/v1/scada/protocols/mqtt/connections` - لیست اتصالات MQTT

## پارامترهای استاندارد دکل

پارامترهای زیر برای دکل 1000 اسب‌بخار تعریف شده‌اند:

- `WOB` - Weight on Bit
- `RPM` - Rotary Speed
- `Torque` - Torque
- `ROP` - Rate of Penetration
- `Mud_Flow_Rate` - Mud Flow Rate
- `Standpipe_Pressure` - Standpipe Pressure
- `Casing_Pressure` - Casing Pressure
- `Hook_Load` - Hook Load
- `Block_Position` - Block Position
- `Pump_Status` - Pump Status
- `Power_Consumption` - Power Consumption
- `Temperature_Bit` - Bit Temperature
- `Temperature_Motor` - Motor Temperature
- `Temperature_Surface` - Surface Temperature

## جریان داده

```
SCADA/PLC System
    ↓
Protocol Service (Modbus/OPC UA/MQTT)
    ↓
SCADA Connector Service
    ↓
Kafka (rig.sensor.stream)
    ↓
Data Bridge Service
    ↓
Database + WebSocket
```

## عیب‌یابی

### بررسی وضعیت اتصال

```bash
curl -X GET "http://localhost:8001/api/v1/scada/rigs/RIG_1000HP/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### بررسی لاگ‌ها

لاگ‌های مربوط به SCADA connector در console و log files قابل مشاهده است:

```python
# در کد Python
import logging
logging.getLogger("services.scada_connector_service").setLevel(logging.DEBUG)
```

### مشکلات رایج

1. **اتصال Modbus برقرار نمی‌شود**
   - بررسی IP address و port
   - بررسی firewall settings
   - بررسی slave_id

2. **اتصال OPC UA برقرار نمی‌شود**
   - بررسی endpoint URL
   - بررسی credentials
   - بررسی security policy

3. **MQTT messages دریافت نمی‌شود**
   - بررسی broker connection
   - بررسی topic subscriptions
   - بررسی QoS levels

## امنیت

- برای production، از TLS/SSL برای MQTT استفاده کنید
- از authentication برای OPC UA استفاده کنید
- Modbus TCP را در شبکه‌های امن استفاده کنید
- از VPN یا firewall برای محافظت از اتصالات استفاده کنید

## مثال‌های کامل

برای مثال‌های کامل پیکربندی، به فایل `config/scada_config.example.yaml` مراجعه کنید.

