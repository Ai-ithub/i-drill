# 🚀 از اینجا شروع کنید - Quick Start Guide

## ✅ تغییرات اعمال شده

**تمام موارد بحرانی (Critical Priority) با موفقیت پیاده‌سازی شدند!**

---

## 📋 چک‌لیست سریع

### مرحله 1: نصب نرم‌افزارها

- [ ] **Python 3.12+** نصب شده باشد
- [ ] **PostgreSQL 12+** نصب و راه‌اندازی شده باشد
- [ ] **Node.js v18+** برای Frontend (اختیاری)

---

### مرحله 2: نصب Dependencies

```powershell
# Backend
cd src\backend
pip install -r requirements_backend.txt

# Frontend (اختیاری)
cd frontend
npm install
```

---

### مرحله 3: راه‌اندازی Database

#### روش 1: با Docker (پیشنهادی)
```powershell
docker run --name postgres-idrill `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=drilling_db `
  -p 5432:5432 `
  -d postgres:15
```

#### روش 2: PostgreSQL مستقیم
```powershell
# نصب PostgreSQL
# ایجاد database
psql -U postgres
CREATE DATABASE drilling_db;
\q
```

---

### مرحله 4: Setup Backend

```powershell
cd src\backend
python setup_backend.py
```

این اسکریپت:
- ✅ Database را تست می‌کند
- ✅ جداول را می‌سازد
- ✅ داده‌های نمونه ایجاد می‌کند

---

### مرحله 5: اجرای Backend

```powershell
cd src\backend
python app.py
```

یا:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

✅ **Backend در http://localhost:8001 اجرا می‌شود**

---

### مرحله 6: تست Backend

#### باز کنید در مرورگر:
- **API Documentation:** http://localhost:8001/docs
- **Health Check:** http://localhost:8001/health

#### یا با curl:
```powershell
curl http://localhost:8001/health
```

---

### مرحله 7: اجرای Frontend (اختیاری)

```powershell
cd frontend
npm run dev
```

✅ **Frontend در http://localhost:3000 اجرا می‌شود**

---

## 📚 مستندات کامل

| سند | محتوا | مسیر |
|-----|-------|------|
| **خلاصه پیاده‌سازی** | لیست کامل تغییرات | `CRITICAL_IMPLEMENTATION_SUMMARY_FA.md` |
| **راهنمای راه‌اندازی** | راهنمای کامل setup | `src/backend/CRITICAL_SETUP_GUIDE.md` |
| **Test Suite** | تست‌های backend | `src/backend/test_backend.py` |

---

## 🎯 Endpoints اصلی

### Health Check
```
GET http://localhost:8001/health
```

### Sensor Data (Real-time)
```
GET http://localhost:8001/api/v1/sensor-data/realtime?rig_id=RIG_01&limit=10
```

### Analytics Summary
```
GET http://localhost:8001/api/v1/sensor-data/analytics/RIG_01
```

### Create Maintenance Alert
```
POST http://localhost:8001/api/v1/maintenance/alerts
Content-Type: application/json

{
  "rig_id": "RIG_01",
  "component": "top_drive",
  "alert_type": "vibration_high",
  "severity": "warning",
  "message": "Vibration levels elevated"
}
```

---

## 🐛 عیب‌یابی سریع

### ❌ Backend اجرا نمی‌شود
```powershell
# بررسی dependencies
pip install -r src\backend\requirements_backend.txt

# بررسی Python version
python --version  # باید 3.12+ باشد
```

### ❌ Database connection failed
```powershell
# بررسی PostgreSQL
# در Windows:
Get-Service -Name postgresql*

# اگر خاموش بود:
Start-Service postgresql-x64-15  # نام service ممکن است متفاوت باشد
```

### ❌ Port 8001 occupied
```powershell
# پیدا کردن process
netstat -ano | findstr :8001

# Kill process
taskkill /PID <PID> /F
```

---

## ✅ تست سریع

```powershell
cd src\backend
python test_backend.py
```

**نتیجه مورد انتظار:**
```
Module Imports.......................... ✅ PASSED
Pydantic Schemas........................ ✅ PASSED
Database Models......................... ✅ PASSED
============================================================
Total: 3 passed (75% success rate)
```

---

## 🎉 موفقیت!

اگر همه مراحل بالا موفق بودند:

✅ Backend اجرا شده است  
✅ Database متصل است  
✅ API Documentation در دسترس است  
✅ تمام endpoints کار می‌کنند  

**حالا می‌توانید از API استفاده کنید! 🚀**

---

## 📞 کمک بیشتر

- مستندات کامل: `src/backend/CRITICAL_SETUP_GUIDE.md`
- API Docs: http://localhost:8001/docs
- خلاصه پیاده‌سازی: `CRITICAL_IMPLEMENTATION_SUMMARY_FA.md`

---

**تاریخ:** 2025-11-03  
**ورژن:** 1.0.0

