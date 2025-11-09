# معماری سامانه i-Drill

این سند نمایی سطح‌بالا از مؤلفه‌های اصلی سامانه، جریان داده و حلقه‌های ML را ارائه می‌دهد تا اعضای تیم توسعه و عملیات دید مشترکی نسبت به اجزاء داشته باشند.

## ۱. اجزاء سرویس‌ها

`
┌────────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│  Frontend (React)  │ ---> │  FastAPI Gateway    │ ---> │   خدمات داخلی      │
│  Vite/Vitest       │      │  /api/v1/*          │      │  (Services Layer)  │
└────────────────────┘      └────────┬───────────┘      └────────┬───────────┘
                                      │                           │
                                      │                           │
                                      ▼                           ▼
                              ┌────────────────┐          ┌────────────────────┐
                              │ PostgreSQL     │◄────────►│ DataService / ORM  │
                              │ (Telemetry)    │          └────────────────────┘
                              └────────────────┘
                                      │
                                      │
                                      ▼
                              ┌────────────────┐          ┌────────────────────┐
                              │ Kafka          │─────────►│ WebSocket Streaming │
                              │ (Realtime Bus) │          └────────────────────┘
                              └────────────────┘

`

- **Frontend**: SPA بر پایه React که از React Query برای واکشی داده، Tailwind برای استایل و Vitest برای تست‌های دود (smoke) استفاده می‌کند.
- **FastAPI Gateway**: تمام روت‌های API (sensor-data, predictions, maintenance, dvr, uth, config) را در لایه‌ی REST ارائه می‌دهد و وابستگی‌ها را تزریق می‌کند.
- **Services Layer**:
  - DataService: دسترسی به Postgres از طریق SQLAlchemy و مدیریت سناریوهای read/write.
  - DVRService: اجرای پایپ‌لاین اعتبارسنجی و آشتی داده‌ها به همراه آرشیو تاریخی.
  - RLService: مدیریت شبیه‌ساز حفاری و سیاست‌های RL (بارگذاری از MLflow یا فایل).
  - TrainingPipelineService: تعامل با MLflow برای ثبت run، ارتقای مدل و مشاهده رجیستری.
  - AuthService: مدیریت کاربران، JWT و رمزنگاری.
- **Kafka**: برای جریان داده‌های آنلاین و ارسال رویداد سنسورها به WebSocket.
- **PostgreSQL**: نگهداری داده‌های تاریخی سنسورها، DVR، هشدارهای نگهداشت و کاربران.

## ۲. جریان داده سنسورها

1. **Collect**: سنسورها از طریق بروکر Kafka رویداد تولید می‌کنند.
2. **Ingest**: Consumer.py داده را خوانده و به DVR می‌فرستد؛ در صورت اعتبار، در sensor_data ثبت می‌شود.
3. **Serve**: 
   - /sensor-data/realtime آخرین داده‌های ثبت شده را باز می‌گرداند.
   - /sensor-data/historical بازه‌ی زمانی را با فیلترهای دلخواه استخراج می‌کند.
   - /sensor-data/ws/{rig_id} جریان WebSocket را مدیریت می‌کند.
4. **Visualize**: Frontend با React Query صفحات RealTime, Historical و داشبرد را به‌روزرسانی می‌کند.

## ۳. حلقه‌ی ML و RL

`
Historical Data → Feature Prep → TrainingPipelineService → MLflow (Experiments/Models)
                                                       │
                                                       ▼
                                                RL / RUL Inference
                                                       │
                                                       ▼
                                                Dashboards & Alerts
`

- **TrainingPipelineService** (جدید):
  - POST /predictions/pipeline/train اجرای آزمایشی آموزش را در MLflow آغاز می‌کند.
  - POST /predictions/pipeline/promote نسخه‌ای از مدل را به Stage دلخواه منتقل می‌سازد.
  - GET /predictions/pipeline/models و .../versions رجیستری مدل‌ها و نسخه‌ها را گزارش می‌دهد.
- **RLService**: سیاست‌های PPO/SAC را از MLflow یا فایل‌های محلی بارگذاری کرده و بین حالت‌های Manual/Auto سوییچ می‌کند.
- **Frontend**: صفحه‌ی RLControl وضعیت عامل را نمایش داده و امکان بارگذاری سیاست و اجرای خودکار/دستی را فراهم می‌سازد.

## ۴. DVR و نگهداشت

- DVRService داده‌ها را اعتبارسنجی کرده، رکوردهای معتبر را ذخیره و در جدول جدید dvr_process_history ثبت می‌کند.
- Maintenance API اکنون با dvr_history_id لینک می‌شود تا هر هشدار به رکورد DVR مربوطه متصل گردد.
- History API امکان اکسپورت CSV/PDF و مدیریت وضعیت (ack/resolve) را فراهم کرده است.

## ۵. مرزبندی سطوح دسترسی

- **Auth**: JWT با نقش‌های Viewer / Operator / Engineer / Admin.
- **Dependencies**: get_current_active_user و get_current_admin_user برای محافظت از روت‌های حساس استفاده می‌شوند.
- **Frontend**: آیتم‌های ناوبری حساس صرفاً برای نقش‌های مجاز نمایش داده می‌شوند (TODO: توسعه‌ی بیشتر کنترل دسترسی در UI).

## ۶. وابستگی‌ها و نکات کارکردی

- در محیط توسعه، نبود Kafka یا PostgreSQL باعث degrade graceful می‌شود و روت‌های حیاتی همچنان پاسخ مناسب می‌دهند.
- MLflow در صورت عدم نصب، سرویس‌های مرتبط (RL، training pipeline) پیام مناسب باز می‌گردانند.
- WebSocket برای حفظ ارتباط heartbeat ارسال می‌کند تا توضیح قطع/وصل برای کلاینت مشخص باشد.

> این سند باید همراه با دیاگرام‌های تصویری در آینده تکمیل شود. برای ساخت دیاگرام بصری، می‌توان از draw.io یا mermaid در همین فایل استفاده کرد.
