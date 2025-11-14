# 🚀 راهنمای شروع سریع - i-Drill

راهنمای گام به گام برای شروع کار با i-Drill

---

## 📋 فهرست مطالب

1. [پیش‌نیازها](#پیش-نیازها)
2. [نصب و راه‌اندازی](#نصب-و-راه-اندازی)
3. [اجرای سیستم](#اجرای-سیستم)
4. [تست سیستم](#تست-سیستم)
5. [مراحل بعدی](#مراحل-بعدی)

---

## پیش‌نیازها

### نرم‌افزارهای مورد نیاز

- **Docker & Docker Compose** (برای اجرای آسان)
  - Docker Desktop: https://www.docker.com/products/docker-desktop
  - یا Docker Engine + Docker Compose

- **Python 3.12+** (برای توسعه Backend)
  - دانلود: https://www.python.org/downloads/
  - بررسی نسخه: `python --version`

- **Node.js 18+** (برای توسعه Frontend)
  - دانلود: https://nodejs.org/
  - بررسی نسخه: `node --version`

- **Git** (برای clone کردن پروژه)
  - دانلود: https://git-scm.com/downloads

### ابزارهای اختیاری

- **Postman** یا **Insomnia** (برای تست API)
- **VS Code** یا **PyCharm** (IDE)
- **pgAdmin** یا **DBeaver** (برای مدیریت PostgreSQL)

---

## نصب و راه‌اندازی

### روش 1: استفاده از Docker (توصیه می‌شود)

#### گام 1: Clone کردن پروژه

```bash
git clone https://github.com/Ai-ithub/i-drill.git
cd i-drill
```

#### گام 2: تنظیم Environment Variables

```bash
# کپی کردن فایل نمونه
cp i-drill/src/backend/config.env.example i-drill/src/backend/config.env

# ویرایش config.env و تنظیم مقادیر
# حداقل SECRET_KEY و DB_PASSWORD را تنظیم کنید
```

#### گام 3: راه‌اندازی با Docker Compose

```bash
# راه‌اندازی تمام سرویس‌ها
docker-compose up -d

# بررسی وضعیت سرویس‌ها
docker-compose ps

# مشاهده لاگ‌ها
docker-compose logs -f
```

#### گام 4: راه‌اندازی Backend

```bash
# ورود به container
docker-compose exec backend bash

# یا اجرای مستقیم
cd i-drill/src/backend
python setup_backend.py
```

#### گام 5: دسترسی به سیستم

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

---

### روش 2: نصب دستی (برای توسعه)

#### گام 1: راه‌اندازی Backend

```bash
cd i-drill/src/backend

# ایجاد virtual environment
python -m venv .venv

# فعال‌سازی virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# نصب dependencies
pip install -r ../../requirements/backend.txt
pip install -r ../../requirements/dev.txt

# تنظیم environment variables
cp config.env.example config.env
# ویرایش config.env

# راه‌اندازی database
python setup_backend.py

# اجرای server
uvicorn app:app --reload --port 8001
```

#### گام 2: راه‌اندازی Frontend

```bash
cd i-drill/frontend

# نصب dependencies
npm install

# ایجاد .env (اختیاری)
echo "VITE_API_URL=http://localhost:8001/api/v1" > .env
echo "VITE_WS_URL=ws://localhost:8001/api/v1" >> .env

# اجرای development server
npm run dev
```

#### گام 3: راه‌اندازی Services (PostgreSQL, Kafka)

```bash
# استفاده از Docker Compose فقط برای services
docker-compose up -d postgres kafka zookeeper

# یا نصب دستی (پیشرفته)
# برای جزئیات، DEPLOYMENT_GUIDE.md را ببینید
```

---

## اجرای سیستم

### بررسی Health Check

```bash
# Backend
curl http://localhost:8001/health

# پاسخ مورد انتظار:
# {"status":"healthy","version":"1.0.0"}
```

### ورود به سیستم

1. باز کردن http://localhost:3000
2. کلیک روی "Login"
3. استفاده از credentials پیش‌فرض:
   - **Username**: `admin`
   - **Password**: `admin`
4. **⚠️ مهم**: در اولین ورود، رمز عبور را تغییر دهید

### تست API

```bash
# Login
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"

# دریافت token و استفاده از آن
TOKEN="your_token_here"
curl -X GET http://localhost:8001/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

---

## تست سیستم

### اجرای Backend Tests

```bash
cd i-drill/src/backend

# اجرای تمام تست‌ها
pytest

# اجرای با coverage
pytest --cov=src/backend --cov-report=html

# اجرای تست‌های خاص
pytest tests/test_auth.py -v
```

### اجرای Frontend Tests

```bash
cd i-drill/frontend

# اجرای تمام تست‌ها
npm test

# اجرای با coverage
npm test -- --coverage

# اجرای در watch mode
npm test -- --watch
```

### اجرای Integration Tests

```bash
cd i-drill

# اجرای تمام تست‌ها (backend + frontend)
# برای جزئیات، TESTING_GUIDE.md را ببینید
```

---

## مراحل بعدی

### 📚 مطالعه مستندات

1. **[User Guide](USER_GUIDE.md)** - راهنمای استفاده از سیستم
2. **[Developer Guide](DEVELOPER_GUIDE.md)** - راهنمای توسعه
3. **[Architecture](ARCHITECTURE.md)** - معماری سیستم
4. **[API Reference](API_REFERENCE.md)** - مرجع API

### 🎯 شروع توسعه

1. **[Contributing Guide](CONTRIBUTING.md)** را مطالعه کنید
2. Issue ایجاد کنید یا Issue موجود را انتخاب کنید
3. Branch بسازید و شروع به کدنویسی کنید
4. Tests بنویسید
5. Pull Request بفرستید

### 🔧 پیکربندی پیشرفته

1. **[Environment Variables](ENVIRONMENT_VARIABLES.md)** - تنظیمات محیطی
2. **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - راهنمای استقرار
3. **[Security Guide](SECURITY_HEADERS_GUIDE.md)** - راهنمای امنیت

---

## 🐛 حل مشکلات رایج

### مشکل: Backend راه‌اندازی نمی‌شود

**راه‌حل:**
1. بررسی کنید که PostgreSQL در حال اجرا است
2. بررسی `config.env` و مقادیر database
3. بررسی لاگ‌ها: `docker-compose logs backend`

### مشکل: Frontend به Backend متصل نمی‌شود

**راه‌حل:**
1. بررسی `VITE_API_URL` در `.env`
2. بررسی CORS settings در backend
3. بررسی firewall settings

### مشکل: Database connection error

**راه‌حل:**
1. بررسی وضعیت PostgreSQL: `docker-compose ps postgres`
2. بررسی credentials در `config.env`
3. بررسی network connectivity

برای مشکلات بیشتر، **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** را ببینید.

---

## 📞 پشتیبانی

- 📧 ایجاد Issue در GitHub
- 📚 بررسی مستندات
- 💬 شروع Discussion

---

**آخرین به‌روزرسانی:** ژانویه 2025  
**نسخه:** 1.0

