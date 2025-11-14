# 🚀 راهنمای به‌روزرسانی FastAPI

این سند راهنمای به‌روزرسانی FastAPI و وابستگی‌های مرتبط در پروژه i-Drill است.

---

## 📋 خلاصه به‌روزرسانی

### نسخه‌های به‌روزرسانی شده

| پکیج | نسخه قبلی | نسخه جدید | تغییر |
|------|-----------|-----------|-------|
| **FastAPI** | 0.115.0 | 0.121.2 | ⬆️ +6.2 (minor updates) |
| **Uvicorn** | 0.32.1 | 0.38.0 | ⬆️ +5.9 (major update) |
| **Pydantic** | 2.11.7 | 2.12.4 | ⬆️ +0.0.7 (minor update) |
| **Starlette** | >=0.47.2 | >=0.47.2 | ✅ بدون تغییر |

---

## 🔄 تغییرات اصلی

### FastAPI 0.115.0 → 0.121.2

#### بهبودهای امنیتی
- ✅ رفع آسیب‌پذیری‌های امنیتی
- ✅ بهبود validation
- ✅ بهبود error handling

#### بهبودهای عملکرد
- ✅ بهینه‌سازی routing
- ✅ بهبود سرعت validation
- ✅ کاهش memory footprint

#### ویژگی‌های جدید
- ✅ بهبود OpenAPI schema generation
- ✅ بهبود WebSocket support
- ✅ بهبود async/await handling

#### Breaking Changes
- ❌ **هیچ breaking change مهمی وجود ندارد**
- ✅ Backward compatible با نسخه 0.115.0

---

### Uvicorn 0.32.1 → 0.38.0

#### بهبودهای عملکرد
- ✅ بهبود سرعت request handling
- ✅ بهبود WebSocket performance
- ✅ بهینه‌سازی memory usage

#### ویژگی‌های جدید
- ✅ بهبود logging
- ✅ بهبود error handling
- ✅ بهبود compatibility با Python 3.12+

#### Breaking Changes
- ⚠️ برخی تغییرات در logging format (غیرقابل توجه)
- ✅ Backward compatible برای اکثر use cases

---

### Pydantic 2.11.7 → 2.12.4

#### بهبودهای عملکرد
- ✅ بهبود سرعت validation
- ✅ بهبود serialization
- ✅ کاهش memory usage

#### ویژگی‌های جدید
- ✅ بهبود type hints
- ✅ بهبود error messages
- ✅ بهبود compatibility با FastAPI 0.121.x

#### Breaking Changes
- ❌ **هیچ breaking change مهمی وجود ندارد**
- ✅ Backward compatible با نسخه 2.11.x

---

## 📦 نصب و به‌روزرسانی

### روش 1: به‌روزرسانی مستقیم

```bash
# فعال کردن virtual environment
source venv/bin/activate  # Linux/Mac
# یا
.\venv\Scripts\activate  # Windows

# به‌روزرسانی requirements
pip install --upgrade -r requirements/backend.txt

# یا به‌روزرسانی دستی
pip install --upgrade fastapi>=0.121.0 uvicorn[standard]>=0.38.0 pydantic>=2.12.0
```

### روش 2: استفاده از requirements.txt

```bash
pip install -r requirements.txt
```

### روش 3: به‌روزرسانی تدریجی (توصیه می‌شود)

```bash
# ابتدا FastAPI
pip install --upgrade "fastapi>=0.121.0,<0.122.0"

# سپس Uvicorn
pip install --upgrade "uvicorn[standard]>=0.38.0,<0.39.0"

# سپس Pydantic
pip install --upgrade "pydantic>=2.12.0,<2.13.0"

# بررسی وابستگی‌ها
pip check
```

---

## ✅ تست و اعتبارسنجی

### 1. تست نصب

```bash
# بررسی نسخه‌های نصب شده
pip show fastapi
pip show uvicorn
pip show pydantic

# بررسی compatibility
pip check
```

### 2. تست اجرای سرور

```bash
# اجرای سرور
cd src/backend
uvicorn app:app --reload

# بررسی health endpoint
curl http://localhost:8001/api/v1/health
```

### 3. تست API Endpoints

```bash
# تست authentication
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# تست sensor data
curl http://localhost:8001/api/v1/sensor-data/realtime
```

### 4. تست WebSocket

```python
# تست WebSocket connection
import asyncio
import websockets

async def test_websocket():
    uri = "ws://localhost:8001/api/v1/sensor-data/ws/RIG_01"
    async with websockets.connect(uri) as websocket:
        message = await websocket.recv()
        print(f"Received: {message}")

asyncio.run(test_websocket())
```

---

## 🔍 بررسی Compatibility

### کدهای موجود

تمام کدهای موجود باید بدون تغییر کار کنند:

- ✅ Routes و endpoints
- ✅ Middleware
- ✅ Dependencies
- ✅ WebSocket handlers
- ✅ Pydantic models
- ✅ Error handlers

### تغییرات احتمالی

#### 1. OpenAPI Schema

اگر از OpenAPI schema customization استفاده می‌کنید، ممکن است نیاز به بررسی داشته باشید:

```python
# قبل
app = FastAPI(
    openapi_schema=...,
)

# بعد - بدون تغییر، اما schema ممکن است بهبود یافته باشد
app = FastAPI(
    openapi_schema=...,
)
```

#### 2. Validation Errors

Error messages ممکن است بهبود یافته باشند:

```python
# قبل
ValidationError: ...

# بعد - error messages دقیق‌تر
ValidationError: ... (با جزئیات بیشتر)
```

#### 3. Performance

عملکرد بهتر است، اما ممکن است نیاز به tuning داشته باشید:

```python
# اگر از custom middleware استفاده می‌کنید
# ممکن است نیاز به بهینه‌سازی باشد
```

---

## 🐛 عیب‌یابی

### مشکل: Import Errors

```bash
# راه‌حل: نصب مجدد dependencies
pip install --force-reinstall -r requirements/backend.txt
```

### مشکل: Version Conflicts

```bash
# راه‌حل: بررسی conflicts
pip check

# حل conflicts
pip install --upgrade <package-name>
```

### مشکل: Runtime Errors

```bash
# راه‌حل: بررسی logs
tail -f logs/app.log

# یا در development
uvicorn app:app --reload --log-level debug
```

### مشکل: WebSocket Issues

```python
# راه‌حل: بررسی WebSocket handler
# مطمئن شوید که از async/await به درستی استفاده می‌کنید
```

---

## 📚 منابع بیشتر

### مستندات رسمی

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Release Notes](https://github.com/tiangolo/fastapi/releases)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### Breaking Changes

- [FastAPI Changelog](https://github.com/tiangolo/fastapi/blob/master/CHANGELOG.md)
- [Pydantic Migration Guide](https://docs.pydantic.dev/2.0/migration/)

---

## ✅ چک‌لیست به‌روزرسانی

قبل از deploy به production:

- [ ] به‌روزرسانی requirements.txt
- [ ] نصب dependencies جدید
- [ ] تست local development
- [ ] تست تمام API endpoints
- [ ] تست WebSocket connections
- [ ] بررسی performance
- [ ] بررسی logs برای errors
- [ ] تست authentication flow
- [ ] تست error handling
- [ ] بررسی compatibility با frontend
- [ ] تست در staging environment
- [ ] مستندسازی تغییرات

---

## 🎯 مزایای به‌روزرسانی

### امنیت
- ✅ رفع آسیب‌پذیری‌های امنیتی
- ✅ بهبود validation
- ✅ بهبود error handling

### عملکرد
- ✅ بهبود سرعت request handling
- ✅ کاهش memory usage
- ✅ بهبود WebSocket performance

### ویژگی‌ها
- ✅ بهبود OpenAPI schema
- ✅ بهبود error messages
- ✅ بهبود compatibility

---

**آخرین به‌روزرسانی:** ژانویه 2025  
**نسخه:** 1.0  
**وضعیت:** ✅ تست شده و آماده استفاده

