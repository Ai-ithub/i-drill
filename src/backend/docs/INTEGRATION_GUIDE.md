# 🔗 Integration Guide: RL Models & DVR

این راهنما نحوه استفاده از Integration Service برای ارتباط بین Reinforcement Learning Models و Data Validation & Reconciliation (DVR) را توضیح می‌دهد.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Integration Pipelines](#integration-pipelines)
4. [API Endpoints](#api-endpoints)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)

---

## Overview

Integration Service یک لایه یکپارچه‌سازی بین سیستم‌های RL و DVR ارائه می‌دهد که امکان:

- **Validation** داده‌های sensor از طریق DVR قبل از feed کردن به RL
- **Validation** actions از RL از طریق DVR قبل از اعمال
- **Enhanced Validation** با استفاده از RL state context
- **Integrated Pipeline** برای جریان کامل داده‌ها

را فراهم می‌کند.

---

## Architecture

```
┌─────────────────┐
│  Sensor Data    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Data Bridge   │─────▶│  DVR Service │
└────────┬────────┘      └──────┬───────┘
         │                      │
         │                      ▼
         │              ┌──────────────┐
         │              │ Validated    │
         │              │ Data         │
         │              └──────┬───────┘
         │                     │
         ▼                     ▼
┌─────────────────┐      ┌──────────────┐
│   Database      │      │  RL Service  │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                        ┌──────────────┐
                        │ RL Actions   │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ DVR Validate │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ Apply Action │
                        └──────────────┘
```

---

## Integration Pipelines

### 1. Sensor Data → DVR → RL

**Endpoint**: `POST /api/v1/integration/sensor-to-rl`

**Pipeline**:
1. دریافت sensor data
2. Validation و reconciliation از طریق DVR
3. تبدیل به RL observation format
4. Feed کردن به RL environment (optional)

**Example**:
```python
import requests

sensor_data = {
    "rig_id": "RIG_01",
    "Depth": 1500.5,
    "WOB": 25000,
    "RPM": 120,
    "Flow_Rate": 800,
    "Torque": 15000,
    "Pressure": 3000,
    "Vibration_Axial": 2.5,
    "Vibration_Lateral": 1.8,
    "Vibration_Torsional": 3.2
}

response = requests.post(
    "http://localhost:8001/api/v1/integration/sensor-to-rl",
    json=sensor_data,
    params={"apply_to_rl": True}
)

result = response.json()
# {
#     "success": True,
#     "dvr_result": {...},
#     "rl_state": {...},
#     "message": "Integrated processing completed successfully"
# }
```

### 2. RL Action → DVR Validation → Apply

**Endpoint**: `POST /api/v1/integration/rl-action-with-dvr`

**Pipeline**:
1. دریافت RL action
2. Validation از طریق DVR anomaly detection
3. اعمال action به RL environment
4. بازگشت نتیجه یکپارچه

**Example**:
```python
action = {
    "wob": 30000,
    "rpm": 140,
    "flow_rate": 850
}

response = requests.post(
    "http://localhost:8001/api/v1/integration/rl-action-with-dvr",
    json=action,
    params={
        "validate_with_dvr": True,
        "history_size": 100
    }
)

result = response.json()
# {
#     "success": True,
#     "validation_result": {
#         "passed": True,
#         "anomaly_detected": False,
#         ...
#     },
#     "rl_result": {...},
#     "message": "Action processed successfully"
# }
```

### 3. Enhanced Validation with RL Context

**Endpoint**: `POST /api/v1/integration/validate-with-rl-context`

**Pipeline**:
1. دریافت sensor data
2. دریافت RL state
3. استفاده از RL context برای validation بهتر
4. بازگشت نتیجه enhanced

**Example**:
```python
response = requests.post(
    "http://localhost:8001/api/v1/integration/validate-with-rl-context",
    json=sensor_data,
    params={"use_rl_state": True}
)

result = response.json()
# {
#     "success": True,
#     "processed_record": {...},
#     "rl_context": {
#         "current_observation": [...],
#         "current_reward": 0.85,
#         ...
#     },
#     "validation_hints": {...}
# }
```

### 4. Integrated Auto Step

**Endpoint**: `POST /api/v1/integration/auto-step-integrated`

**Pipeline**:
1. دریافت action از RL policy
2. Validation از طریق DVR
3. اعمال action
4. بازگشت نتیجه یکپارچه

**Example**:
```python
response = requests.post(
    "http://localhost:8001/api/v1/integration/auto-step-integrated",
    params={"validate_action": True}
)

result = response.json()
# {
#     "success": True,
#     "rl_result": {...},
#     "validation_result": {...},
#     "message": "Integrated auto step completed"
# }
```

---

## API Endpoints

### Get Integration Status

**Endpoint**: `GET /api/v1/integration/status`

**Response**:
```json
{
    "success": true,
    "status": {
        "rl_available": true,
        "rl_policy_loaded": true,
        "rl_policy_mode": "auto",
        "dvr_available": true,
        "integration_active": true,
        "rl_episode": 5,
        "rl_step": 150
    }
}
```

---

## Configuration

### Environment Variables

```env
# Enable DVR processing in Data Bridge
ENABLE_DVR_IN_BRIDGE=true

# Enable RL integration in Data Bridge (optional, more resource intensive)
ENABLE_RL_IN_BRIDGE=false
```

### Data Bridge Integration

Data Bridge به صورت خودکار از Integration Service استفاده می‌کند اگر:
- `ENABLE_DVR_IN_BRIDGE=true` تنظیم شده باشد
- DVR Service در دسترس باشد

این باعث می‌شود که تمام داده‌های sensor که از Kafka می‌آیند:
1. از طریق DVR validate شوند
2. Reconciled شوند
3. سپس در database ذخیره شوند
4. به WebSocket clients broadcast شوند

---

## Usage Examples

### Complete Pipeline Example

```python
from services.integration_service import integration_service

# Step 1: Process sensor data
sensor_record = {
    "rig_id": "RIG_01",
    "Depth": 1500.5,
    "WOB": 25000,
    "RPM": 120,
    "Flow_Rate": 800
}

# Process through DVR and optionally feed to RL
result = integration_service.process_sensor_data_for_rl(
    sensor_record=sensor_record,
    apply_to_rl=True
)

if result["success"]:
    dvr_result = result["dvr_result"]
    rl_state = result["rl_state"]
    
    print(f"DVR: {dvr_result['message']}")
    print(f"RL State: {rl_state}")

# Step 2: Get action from RL and validate
rl_action = {
    "wob": 30000,
    "rpm": 140,
    "flow_rate": 850
}

# Validate and apply action
action_result = integration_service.process_rl_action_with_dvr(
    action=rl_action,
    validate_with_dvr=True,
    history_size=100
)

if action_result["success"]:
    validation = action_result["validation_result"]
    rl_result = action_result["rl_result"]
    
    print(f"Validation: {validation['message']}")
    print(f"RL Result: {rl_result}")
```

### Automated Pipeline with Data Bridge

با تنظیم `ENABLE_DVR_IN_BRIDGE=true`، تمام داده‌های sensor به صورت خودکار:

1. از Kafka دریافت می‌شوند
2. از طریق DVR validate و reconcile می‌شوند
3. در database ذخیره می‌شوند
4. به WebSocket clients broadcast می‌شوند

برای فعال کردن RL integration نیز، `ENABLE_RL_IN_BRIDGE=true` تنظیم کنید.

---

## Best Practices

### 1. Validation Strategy

- **DVR First**: همیشه داده‌ها را از طریق DVR validate کنید قبل از feed کردن به RL
- **Fail Open**: اگر DVR validation failed، لاگ کنید اما block نکنید (مگر در موارد critical)
- **Anomaly Detection**: از DVR anomaly detection برای شناسایی مشکلات استفاده کنید

### 2. RL Integration

- **Policy Loading**: مطمئن شوید policy load شده باشد قبل از استفاده
- **Action Validation**: Actions را از طریق DVR validate کنید قبل از apply
- **State Context**: از RL state برای enhanced validation استفاده کنید

### 3. Performance

- **Batch Processing**: برای داده‌های زیاد، batch processing استفاده کنید
- **Async Processing**: برای operations سنگین، async processing استفاده کنید
- **Caching**: نتایج validation را cache کنید برای performance بهتر

### 4. Error Handling

- **Graceful Degradation**: اگر یک بخش fail شد، بقیه سیستم باید کار کند
- **Logging**: تمام errors را log کنید برای debugging
- **Monitoring**: Integration status را monitor کنید

---

## Troubleshooting

### RL Not Available

**Problem**: `rl_available: false`

**Solution**: 
- مطمئن شوید `drilling_env` package نصب است
- بررسی کنید که `DrillingEnv` قابل import است

### DVR Not Available

**Problem**: `dvr_available: false`

**Solution**:
- مطمئن شوید database initialized است
- بررسی کنید که DVR tables موجود هستند

### Integration Not Active

**Problem**: `integration_active: false`

**Solution**:
- مطمئن شوید هم RL و هم DVR available هستند
- بررسی کنید configuration درست است

---

**تاریخ آخرین بروزرسانی**: 2024  
**نسخه**: 1.0.0

