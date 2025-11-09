# متغیرهای محیطی و سناریوهای پشتیبان‌گیری

این سند متغیرهای کلیدی محیطی برای بک‌اند و فرانت‌اند را فهرست کرده و سناریوهای بکاپ پیشنهادی را توضیح می‌دهد.

## ۱. متغیرهای عمومی

| نام متغیر | پیش‌فرض | توضیح |
|-----------|---------|--------|
| APP_ENV | development | تعیین نوع محیط؛ در production خطا در صورت استفاده از SECRET_KEY پیش‌فرض رخ می‌دهد. |
| SECRET_KEY | your-secret-key | کلید امضای JWT. در production حتماً مقدار تصادفی ۳۲ کاراکتری تنظیم شود. |
| ALLOWED_ORIGINS | http://localhost:3000,http://localhost:5173 | لیست دامنه‌های مجاز CORS. در production محدود به دامنه‌های معتبر شود. |
| FORCE_HTTPS | alse | در صورت 	rue، FastAPI به HTTPs ریدایرکت می‌کند. |
| ENABLE_RATE_LIMIT | 	rue | فعال/غیرفعال کردن middleware محدودکننده نرخ. |
| RATE_LIMIT_DEFAULT | 100/minute | سقف درخواست در هر دقیقه. |

## ۲. پایگاه داده و DVR

| نام متغیر | توضیح |
|-----------|--------|
| DATABASE_URL | در صورت عدم استفاده از config loader، مسیر کامل اتصال Postgres. |
| DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD | متغیرهای سازگار با config_loader برای DatabaseManager. |
| DVR_EXPORT_PATH | (اختیاری) مسیر ذخیره خروجی CSV/PDF در صورت نیاز به نوشتن روی دیسک. پیش‌فرض خروجی در حافظه است. |

### سناریوی بکاپ

- اجرای روزانه pg_dump --format=custom --file=backups/dvr_$(date +%F) .dump.
- نگهداری حداقل ۷ نسخه اخیر و انتقال به فضای ابری (S3/MinIO).
- تست بازیابی ماهانه با استفاده از pg_restore در محیط staging.

## ۳. Kafka و استریم

| نام متغیر | توضیح |
|-----------|--------|
| KAFKA_BOOTSTRAP_SERVERS | مقداردهی در docker-compose یا محیط اجرا برای اتصال تولیدکننده/مصرف‌کننده. |
| KAFKA_SENSOR_TOPIC | موضوع اصلی سنسورها. پیش‌فرض ig.sensor.stream. |

### بکاپ Kafka

- در صورت استفاده از single-broker، Snapshot دیسک انجام شود.
- برای محیط‌های حیاتی، حداقل سه broker و فعال‌سازی replication ضروری است.

## ۴. MLflow و مدل‌ها

| نام متغیر | توضیح |
|-----------|--------|
| MLFLOW_TRACKING_URI | مسیر سرور MLflow (مثلاً http://mlflow.internal:5000). |
| MLFLOW_EXPERIMENT_NAME | نام پیش‌فرض آزمایشات (i-drill-models). |

### بکاپ MLflow

- اگر از backend فایل استفاده می‌کنید، فولدر mlruns/ را در فضای مشترک (NFS یا S3) قرار دهید.
- در صورت استفاده از backend پایگاه داده (مثلاً MySQL)، سیاست بکاپ مطابق با DB اعمال گردد.

## ۵. فرانت‌اند

| نام متغیر | توضیح |
|-----------|--------|
| VITE_API_URL | آدرس پایه API در زمان build (مثلاً https://api.i-drill.io/api/v1). |
| VITE_WS_URL | آدرس WebSocket (مثلاً wss://api.i-drill.io/api/v1/sensor-data/ws). |

### بکاپ و نسخه‌بندی

- پس از هر build پایدار، خروجی dist/ در مخزن artifact یا فضای ابری ذخیره شود.
- توصیه می‌شود نام نسخه در package.json به‌روزرسانی و با همان نسخه tag ایجاد شود.

## ۶. چک‌لیست امنیتی

- فعال‌سازی HTTPS و تنظیم گواهی SSL (Let's Encrypt یا داخلی).
- فعال کردن Rate Limit و در صورت نیاز WAF در سطح Nginx/Load Balancer.
- تعریف کاربر جداگانه با حداقل دسترسی برای Postgres و Kafka.
- فعال‌سازی لاگ‌های audit برای عملیات حساس (به‌ویژه ماژول auth).

## ۷. Disaster Recovery

1. **Postgres**: وجود بکاپ روزانه + تست بازیابی ماهانه.
2. **Kafka**: استفاده از replication و نگهداشت snapshot از حجم داده حیاتی.
3. **MLflow**: ذخیره آرشیو مدل‌ها در مخزن شیء (S3/MinIO).
4. **Frontend**: نگهداشت build پایدار و امکان rollback در CDN یا Reverse Proxy.
5. **پایش سلامت**: /api/v1/health در مانیتورینگ قرار گیرد تا در صورت خرابی سرویس هشدار ارسال شود.
