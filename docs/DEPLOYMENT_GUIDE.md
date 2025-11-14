# راهنمای استقرار فاز تولید

این راهنما مراحل استقرار بک‌اند و فرانت‌اند i-Drill را روی سرور مجازی یا محیط Docker توضیح می‌دهد.

## ۱. پیش‌نیازها

- **سیستم‌عامل**: Ubuntu 22.04 یا Windows Server 2019
- **Docker & Docker Compose** (آخرین نسخه)
- **Python 3.12** و **Node.js 18** برای نصب محلی بدون Docker
- دسترسی به **PostgreSQL 15** و **Kafka 7.5** در شبکه داخلی یا سرویس مدیریت‌شده

## ۲. استقرار سریع با Docker Compose

1. فایل .env را براساس src/backend/config.env.example تکمیل کنید.
2. سرویس‌های پایگاه‌داده و Kafka را بالا بیاورید:
   `ash
   docker compose up -d postgres kafka zookeeper
   `
3. سپس سرویس FastAPI و Dashboard را اجرا کنید:
   `ash
   docker compose up -d fastapi frontend
   `
4. آدرس‌ها:
   - API: http://<host>:8001/docs
   - Dashboard: http://<host>:5173

> در محیط‌های تولید حتماً SECRET_KEY و ALLOWED_ORIGINS را مقداردهی امن کنید.

## ۳. استقرار دستی روی Ubuntu (بدون Docker)

1. نصب وابستگی‌های سیستم:
   `ash
   sudo apt update && sudo apt install build-essential python3.12 python3.12-venv nodejs npm
   `
2. ایجاد و فعال‌سازی محیط مجازی Python:
   `ash
   cd i-drill/src/backend
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements_backend.txt
   `
   > برای نصب کامل شامل پکیج‌های یادگیری ماشین، از ریشه مخزن دستور  
   > `pip install -r requirements.txt` را اجرا کنید.
3. نصب وابستگی‌های فرانت‌اند:
   `ash
   cd ../../frontend
   npm ci
   npm run build
   `
4. اجرای بک‌اند:
   `ash
   cd ../src/backend
   uvicorn app:app --host 0.0.0.0 --port 8001 --workers 4
   `
5. سرو کردن خروجی فرانت‌اند (نمونه با Nginx):
   - ساخت آرشیو rontend/dist
   - پیکربندی Nginx برای سرو فایل‌های استاتیک و پروکسی /api به پورت 8001.

## ۴. ادغام MLflow و DVR

- **MLflow**: در صورت نیاز به آموزش و رجیستری مدل، سرویس MLflow را اجرا کنید و مقدار MLFLOW_TRACKING_URI را در متغیرهای محیطی تنظیم نمایید. بدون آن، API پیام «MLflow not configured» باز می‌گرداند.
- **DVR**: دیتابیس Postgres باید جدول‌های sensor_data و dvr_process_history را داشته باشد. اسکریپت‌های Alembic یا Base.metadata.create_all() در database.py برای ایجاد اولیه کافی است.

## ۵. Backup & Restore

- **PostgreSQL**: از pg_dump برای پشتیبان‌گیری و pg_restore برای بازیابی استفاده کنید.
- **MLflow**: اگر از backend فایل استفاده می‌کنید، مسیر mlruns را در فضای ذخیره‌سازی پایدار قرار دهید.
- **Frontend Build**: پوشه‌ی rontend/dist را در کنار پیکربندی Nginx نسخه‌گذاری کنید تا در صورت نیاز به rollback سریع قابل بازگردانی باشد.

## ۶. مانیتورینگ و لاگینگ

- از Prometheus/Grafana برای جمع‌آوری متریک API (مانند زمان پاسخ /health) استفاده کنید.
- لاگ‌های FastAPI را به فایل ثبت کرده و با ELK Stack قابل جمع‌آوری است.
- برای Kafka، از kafkacat یا Control Center برای پایش lag بهره ببرید.

## ۷. چک‌لیست پس از استقرار

- [ ] اجرای pytest و 
pm run test در محیط CI/سرور build
- [ ] پاسخ‌دهی GET /api/v1/health
- [ ] بارگذاری صفحه Dashboard و نمایش داده تاریخی
- [ ] اتصال WebSocket و دریافت پیام وضعیت
- [ ] بررسی اتصال MLflow (اختیاری)
- [ ] راه‌اندازی Backup زمان‌بندی‌شده برای Postgres

> برای سناریوهای چندسرویسی (Cluster)، پیشنهاد می‌شود از Kubernetes و Helm استفاده شود. این سند در آن صورت به عنوان baseline عمل می‌کند.
