# 🗄️ Database Schema Documentation

مستندات کامل schema دیتابیس i-Drill

## 📋 فهرست مطالب

- [نمای کلی](#نمای-کلی)
- [جداول اصلی](#جداول-اصلی)
- [روابط و Foreign Keys](#روابط-و-foreign-keys)
- [Indexes](#indexes)
- [Constraints](#constraints)
- [ER Diagram](#er-diagram)

---

## نمای کلی

دیتابیس i-Drill از **PostgreSQL** استفاده می‌کند و شامل **15 جدول اصلی** است که داده‌های حفاری، کاربران، تعمیرات، و پیش‌بینی‌ها را مدیریت می‌کنند.

### آمار کلی

- **تعداد جداول**: 15
- **تعداد Indexes**: 50+
- **Foreign Keys**: 8
- **Unique Constraints**: 5

---

## جداول اصلی

### 1. sensor_data

داده‌های سنسورهای حفاری

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| timestamp | TIMESTAMP | ❌ | زمان ثبت |
| depth | FLOAT | ❌ | عمق (متر) |
| wob | FLOAT | ❌ | Weight on Bit (تن) |
| rpm | FLOAT | ❌ | دور بر دقیقه |
| torque | FLOAT | ❌ | گشتاور (N.m) |
| rop | FLOAT | ❌ | Rate of Penetration (m/h) |
| mud_flow | FLOAT | ❌ | جریان گل (L/min) |
| mud_pressure | FLOAT | ❌ | فشار گل (bar) |
| mud_temperature | FLOAT | ✅ | دمای گل (°C) |
| gamma_ray | FLOAT | ✅ | اشعه گاما |
| resistivity | FLOAT | ✅ | مقاومت |
| density | FLOAT | ✅ | چگالی |
| porosity | FLOAT | ✅ | تخلخل |
| hook_load | FLOAT | ✅ | بار قلاب (تن) |
| vibration | FLOAT | ✅ | ارتعاش |
| status | VARCHAR(20) | ✅ | وضعیت (default: 'normal') |

**Indexes:**
- `ix_sensor_data_id` (id)
- `ix_sensor_data_rig_id` (rig_id)
- `ix_sensor_data_timestamp` (timestamp)

---

### 2. users

کاربران سیستم

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| username | VARCHAR(50) | ❌ | نام کاربری (Unique) |
| email | VARCHAR(100) | ❌ | ایمیل (Unique) |
| hashed_password | VARCHAR(255) | ❌ | رمز عبور hash شده |
| full_name | VARCHAR(100) | ✅ | نام کامل |
| role | VARCHAR(20) | ❌ | نقش (default: 'viewer') |
| is_active | BOOLEAN | ✅ | فعال/غیرفعال (default: true) |
| created_at | TIMESTAMP | ✅ | تاریخ ایجاد |
| last_login | TIMESTAMP | ✅ | آخرین ورود |
| failed_login_attempts | INTEGER | ✅ | تعداد تلاش‌های ناموفق (default: 0) |
| locked_until | TIMESTAMP | ✅ | قفل شده تا |
| password_changed_at | TIMESTAMP | ✅ | تاریخ تغییر رمز |

**Indexes:**
- `ix_users_id` (id)
- `ix_users_username` (username) - Unique
- `ix_users_email` (email) - Unique

**Roles:**
- `admin` - دسترسی کامل
- `operator` - اپراتور
- `viewer` - فقط مشاهده

---

### 3. maintenance_alerts

هشدارهای تعمیرات

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| component | VARCHAR(100) | ❌ | قطعه |
| alert_type | VARCHAR(50) | ❌ | نوع هشدار |
| severity | VARCHAR(20) | ❌ | شدت (critical, warning, info) |
| message | TEXT | ❌ | پیام |
| predicted_failure_time | TIMESTAMP | ✅ | زمان پیش‌بینی شده خرابی |
| created_at | TIMESTAMP | ✅ | تاریخ ایجاد |
| acknowledged | BOOLEAN | ✅ | تایید شده (default: false) |
| acknowledged_by | VARCHAR(100) | ✅ | تایید کننده |
| acknowledged_at | TIMESTAMP | ✅ | زمان تایید |
| acknowledgement_notes | TEXT | ✅ | یادداشت تایید |
| resolved | BOOLEAN | ✅ | حل شده (default: false) |
| resolved_at | TIMESTAMP | ✅ | زمان حل |
| resolved_by | VARCHAR(100) | ✅ | حل کننده |
| resolution_notes | TEXT | ✅ | یادداشت حل |
| dvr_history_id | INTEGER | ✅ | Foreign Key به dvr_process_history |

**Indexes:**
- `ix_maintenance_alerts_id` (id)
- `ix_maintenance_alerts_rig_id` (rig_id)
- `ix_maintenance_alerts_severity` (severity)
- `ix_maintenance_alerts_created_at` (created_at)

**Foreign Keys:**
- `dvr_history_id` → `dvr_process_history.id` (ON DELETE SET NULL)

---

### 4. maintenance_schedules

برنامه تعمیرات

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| component | VARCHAR(100) | ❌ | قطعه |
| maintenance_type | VARCHAR(50) | ❌ | نوع تعمیرات |
| scheduled_date | TIMESTAMP | ❌ | تاریخ برنامه‌ریزی شده |
| estimated_duration_hours | FLOAT | ❌ | مدت زمان تخمینی (ساعت) |
| priority | VARCHAR(20) | ❌ | اولویت |
| status | VARCHAR(20) | ✅ | وضعیت (default: 'scheduled') |
| assigned_to | VARCHAR(100) | ✅ | اختصاص داده شده به |
| notes | TEXT | ✅ | یادداشت‌ها |
| created_at | TIMESTAMP | ✅ | تاریخ ایجاد |
| updated_at | TIMESTAMP | ✅ | تاریخ به‌روزرسانی |

**Indexes:**
- `ix_maintenance_schedules_id` (id)
- `ix_maintenance_schedules_rig_id` (rig_id)
- `ix_maintenance_schedules_scheduled_date` (scheduled_date)
- `ix_maintenance_schedules_status` (status)

**Status Values:**
- `scheduled` - برنامه‌ریزی شده
- `in_progress` - در حال انجام
- `completed` - تکمیل شده
- `cancelled` - لغو شده

---

### 5. password_reset_tokens

توکن‌های بازنشانی رمز عبور

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| user_id | INTEGER | ❌ | Foreign Key به users |
| token | VARCHAR(255) | ❌ | توکن (Unique) |
| expires_at | TIMESTAMP | ❌ | تاریخ انقضا |
| used | BOOLEAN | ✅ | استفاده شده (default: false) |
| created_at | TIMESTAMP | ✅ | تاریخ ایجاد |

**Indexes:**
- `ix_password_reset_tokens_id` (id)
- `ix_password_reset_tokens_user_id` (user_id)
- `ix_password_reset_tokens_token` (token) - Unique
- `ix_password_reset_tokens_expires_at` (expires_at)

**Foreign Keys:**
- `user_id` → `users.id` (ON DELETE CASCADE)

---

### 6. blacklisted_tokens

توکن‌های JWT بلاک شده

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| token | VARCHAR(500) | ❌ | توکن (Unique) |
| user_id | INTEGER | ✅ | Foreign Key به users |
| expires_at | TIMESTAMP | ❌ | تاریخ انقضا |
| created_at | TIMESTAMP | ✅ | تاریخ ایجاد |
| reason | VARCHAR(100) | ✅ | دلیل (logout, password_change, etc.) |

**Indexes:**
- `ix_blacklisted_tokens_id` (id)
- `ix_blacklisted_tokens_token` (token) - Unique
- `ix_blacklisted_tokens_expires_at` (expires_at)
- `ix_blacklisted_tokens_user_id` (user_id)

**Foreign Keys:**
- `user_id` → `users.id` (ON DELETE SET NULL)

---

### 7. login_attempts

تاریخچه تلاش‌های ورود

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| username | VARCHAR(50) | ❌ | نام کاربری |
| ip_address | VARCHAR(45) | ✅ | آدرس IP (پشتیبانی IPv6) |
| success | BOOLEAN | ✅ | موفق/ناموفق (default: false) |
| attempted_at | TIMESTAMP | ✅ | زمان تلاش |
| user_agent | VARCHAR(255) | ✅ | User Agent |

**Indexes:**
- `ix_login_attempts_id` (id)
- `ix_login_attempts_username` (username)
- `ix_login_attempts_attempted_at` (attempted_at)
- `ix_login_attempts_ip_address` (ip_address)

---

### 8. change_requests

درخواست‌های تغییر پارامترهای حفاری

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| change_type | VARCHAR(20) | ❌ | نوع تغییر |
| component | VARCHAR(100) | ❌ | قطعه |
| parameter | VARCHAR(100) | ❌ | پارامتر |
| old_value | TEXT | ✅ | مقدار قدیمی |
| new_value | TEXT | ❌ | مقدار جدید |
| status | VARCHAR(20) | ✅ | وضعیت (default: 'pending') |
| auto_execute | BOOLEAN | ✅ | اجرای خودکار (default: false) |
| requested_by | INTEGER | ✅ | Foreign Key به users |
| approved_by | INTEGER | ✅ | Foreign Key به users |
| applied_by | INTEGER | ✅ | Foreign Key به users |
| requested_at | TIMESTAMP | ✅ | زمان درخواست |
| approved_at | TIMESTAMP | ✅ | زمان تایید |
| applied_at | TIMESTAMP | ✅ | زمان اعمال |
| rejection_reason | TEXT | ✅ | دلیل رد |
| error_message | TEXT | ✅ | پیام خطا |
| metadata | JSONB | ✅ | داده‌های اضافی |

**Indexes:**
- `ix_change_requests_id` (id)
- `ix_change_requests_rig_id` (rig_id)
- `ix_change_requests_change_type` (change_type)
- `ix_change_requests_status` (status)
- `ix_change_requests_requested_at` (requested_at)
- `ix_change_requests_requested_by` (requested_by)

**Foreign Keys:**
- `requested_by` → `users.id` (ON DELETE SET NULL)
- `approved_by` → `users.id` (ON DELETE SET NULL)
- `applied_by` → `users.id` (ON DELETE SET NULL)

**Status Values:**
- `pending` - در انتظار
- `approved` - تایید شده
- `rejected` - رد شده
- `applied` - اعمال شده
- `failed` - ناموفق

---

### 9. dvr_process_history

تاریخچه پردازش DVR (Data Validation and Reconciliation)

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ✅ | شناسه rig |
| raw_record | JSONB | ❌ | رکورد خام |
| reconciled_record | JSONB | ✅ | رکورد تطبیق شده |
| is_valid | BOOLEAN | ❌ | معتبر (default: true) |
| reason | TEXT | ✅ | دلیل |
| anomaly_flag | BOOLEAN | ❌ | پرچم anomaly (default: false) |
| anomaly_details | JSONB | ✅ | جزئیات anomaly |
| status | VARCHAR(20) | ❌ | وضعیت (default: 'processed') |
| notes | TEXT | ✅ | یادداشت‌ها |
| source | VARCHAR(50) | ✅ | منبع |
| created_at | TIMESTAMP | ✅ | تاریخ ایجاد |
| updated_at | TIMESTAMP | ✅ | تاریخ به‌روزرسانی |

**Indexes:**
- `ix_dvr_process_history_id` (id)
- `ix_dvr_process_history_rig_id` (rig_id)
- `ix_dvr_process_history_status` (status)
- `ix_dvr_process_history_created_at` (created_at)

---

### 10. rul_predictions

پیش‌بینی‌های RUL (Remaining Useful Life)

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| component | VARCHAR(100) | ❌ | قطعه |
| predicted_rul | FLOAT | ❌ | RUL پیش‌بینی شده (ساعت) |
| confidence | FLOAT | ❌ | سطح اطمینان (0-1) |
| timestamp | TIMESTAMP | ✅ | زمان پیش‌بینی |
| model_used | VARCHAR(50) | ❌ | مدل استفاده شده |
| recommendation | TEXT | ✅ | توصیه |
| actual_failure_time | TIMESTAMP | ✅ | زمان واقعی خرابی |

**Indexes:**
- `ix_rul_predictions_id` (id)
- `ix_rul_predictions_rig_id` (rig_id)
- `ix_rul_predictions_timestamp` (timestamp)

---

### 11. anomaly_detections

نتایج تشخیص Anomaly

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| timestamp | TIMESTAMP | ✅ | زمان تشخیص |
| is_anomaly | BOOLEAN | ❌ | anomaly است |
| anomaly_score | FLOAT | ❌ | امتیاز anomaly |
| affected_parameters | JSONB | ❌ | پارامترهای تاثیر گرفته |
| severity | VARCHAR(20) | ❌ | شدت |
| description | TEXT | ✅ | توضیحات |
| investigated | BOOLEAN | ✅ | بررسی شده (default: false) |
| investigation_notes | TEXT | ✅ | یادداشت بررسی |

**Indexes:**
- `ix_anomaly_detections_id` (id)
- `ix_anomaly_detections_rig_id` (rig_id)
- `ix_anomaly_detections_timestamp` (timestamp)

---

### 12. model_versions

ورژن‌های مدل‌های ML

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| model_name | VARCHAR(100) | ❌ | نام مدل |
| version | VARCHAR(50) | ❌ | ورژن |
| model_type | VARCHAR(50) | ❌ | نوع مدل |
| file_path | VARCHAR(255) | ❌ | مسیر فایل |
| metrics | JSONB | ✅ | متریک‌های مدل |
| training_date | TIMESTAMP | ✅ | تاریخ آموزش |
| is_active | BOOLEAN | ✅ | فعال (default: false) |
| description | TEXT | ✅ | توضیحات |
| created_by | VARCHAR(100) | ✅ | ایجاد کننده |

**Indexes:**
- `ix_model_versions_id` (id)
- `ix_model_versions_model_name` (model_name)

---

### 13. well_profiles

پروفایل چاه‌ها

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| well_id | VARCHAR(50) | ❌ | شناسه چاه (Unique) |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| total_depth | FLOAT | ❌ | عمق کل (متر) |
| kick_off_point | FLOAT | ❌ | نقطه شروع انحراف (متر) |
| build_rate | FLOAT | ❌ | نرخ ساخت (درجه/متر) |
| max_inclination | FLOAT | ❌ | حداکثر انحراف (درجه) |
| target_zone_start | FLOAT | ❌ | شروع منطقه هدف (متر) |
| target_zone_end | FLOAT | ❌ | پایان منطقه هدف (متر) |
| geological_data | JSONB | ✅ | داده‌های زمین‌شناسی |
| created_at | TIMESTAMP | ✅ | تاریخ ایجاد |
| updated_at | TIMESTAMP | ✅ | تاریخ به‌روزرسانی |

**Indexes:**
- `ix_well_profiles_id` (id)
- `ix_well_profiles_well_id` (well_id) - Unique
- `ix_well_profiles_rig_id` (rig_id)

---

### 14. drilling_sessions

جلسات حفاری

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| rig_id | VARCHAR(50) | ❌ | شناسه rig |
| well_id | VARCHAR(50) | ❌ | شناسه چاه |
| start_time | TIMESTAMP | ❌ | زمان شروع |
| end_time | TIMESTAMP | ✅ | زمان پایان |
| start_depth | FLOAT | ❌ | عمق شروع (متر) |
| end_depth | FLOAT | ✅ | عمق پایان (متر) |
| average_rop | FLOAT | ✅ | میانگین ROP (m/h) |
| total_drilling_time_hours | FLOAT | ✅ | کل زمان حفاری (ساعت) |
| status | VARCHAR(20) | ✅ | وضعیت (default: 'active') |
| notes | TEXT | ✅ | یادداشت‌ها |

**Indexes:**
- `ix_drilling_sessions_id` (id)
- `ix_drilling_sessions_rig_id` (rig_id)
- `ix_drilling_sessions_well_id` (well_id)

**Status Values:**
- `active` - فعال
- `completed` - تکمیل شده
- `paused` - متوقف شده
- `cancelled` - لغو شده

---

### 15. system_logs

لاگ‌های سیستم

| ستون | نوع | Nullable | توضیحات |
|------|-----|----------|---------|
| id | INTEGER | ❌ | Primary Key |
| timestamp | TIMESTAMP | ✅ | زمان |
| level | VARCHAR(20) | ❌ | سطح (INFO, WARNING, ERROR) |
| service | VARCHAR(50) | ❌ | سرویس |
| message | TEXT | ❌ | پیام |
| details | JSONB | ✅ | جزئیات |
| user_id | INTEGER | ✅ | Foreign Key به users |

**Indexes:**
- `ix_system_logs_id` (id)
- `ix_system_logs_timestamp` (timestamp)
- `ix_system_logs_level` (level)
- `ix_system_logs_service` (service)
- `ix_system_logs_user_id` (user_id)

**Foreign Keys:**
- `user_id` → `users.id` (ON DELETE SET NULL)

---

## روابط و Foreign Keys

### نمودار روابط

```
users
  ├── password_reset_tokens (user_id → users.id, CASCADE)
  ├── blacklisted_tokens (user_id → users.id, SET NULL)
  ├── change_requests (requested_by, approved_by, applied_by → users.id, SET NULL)
  └── system_logs (user_id → users.id, SET NULL)

dvr_process_history
  └── maintenance_alerts (dvr_history_id → dvr_process_history.id, SET NULL)
```

### خلاصه Foreign Keys

| جدول | ستون | جدول مرجع | ستون مرجع | ON DELETE |
|------|------|-----------|-----------|-----------|
| password_reset_tokens | user_id | users | id | CASCADE |
| blacklisted_tokens | user_id | users | id | SET NULL |
| change_requests | requested_by | users | id | SET NULL |
| change_requests | approved_by | users | id | SET NULL |
| change_requests | applied_by | users | id | SET NULL |
| maintenance_alerts | dvr_history_id | dvr_process_history | id | SET NULL |
| system_logs | user_id | users | id | SET NULL |

---

## Indexes

### Indexes برای Performance

تمام جداول دارای index روی `id` (Primary Key) هستند که به صورت خودکار ایجاد می‌شوند.

**Indexes مهم برای Query Performance:**

1. **sensor_data**
   - `rig_id` + `timestamp` - برای query های زمانی بر اساس rig
   
2. **maintenance_alerts**
   - `rig_id` + `severity` - برای فیلتر کردن alerts
   - `created_at` - برای مرتب‌سازی زمانی

3. **change_requests**
   - `status` + `requested_at` - برای فیلتر کردن درخواست‌ها

4. **dvr_process_history**
   - `rig_id` + `status` - برای query های پردازش

5. **rul_predictions**
   - `rig_id` + `timestamp` - برای تاریخچه پیش‌بینی‌ها

---

## Constraints

### Unique Constraints

1. `users.username` - نام کاربری یکتا
2. `users.email` - ایمیل یکتا
3. `password_reset_tokens.token` - توکن یکتا
4. `blacklisted_tokens.token` - توکن یکتا
5. `well_profiles.well_id` - شناسه چاه یکتا

### Check Constraints

(در صورت نیاز می‌توان اضافه کرد)

---

## ER Diagram

```
┌─────────────┐
│    users    │
└──────┬──────┘
       │
       ├─── password_reset_tokens
       ├─── blacklisted_tokens
       ├─── change_requests (requested_by, approved_by, applied_by)
       └─── system_logs
       
┌──────────────────┐
│ sensor_data      │
└──────────────────┘

┌──────────────────┐
│ maintenance_     │
│ alerts           │──┐
└──────────────────┘  │
                      │
┌──────────────────┐  │
│ dvr_process_     │──┘
│ history          │
└──────────────────┘

┌──────────────────┐
│ maintenance_     │
│ schedules        │
└──────────────────┘

┌──────────────────┐
│ change_requests  │
└──────────────────┘

┌──────────────────┐
│ rul_predictions  │
└──────────────────┘

┌──────────────────┐
│ anomaly_         │
│ detections       │
└──────────────────┘

┌──────────────────┐
│ model_versions   │
└──────────────────┘

┌──────────────────┐
│ well_profiles    │
└──────────────────┘

┌──────────────────┐
│ drilling_        │
│ sessions         │
└──────────────────┘
```

---

## نکات مهم

### 1. Data Types

- استفاده از `FLOAT` برای مقادیر عددی با دقت اعشار
- استفاده از `TIMESTAMP` برای زمان‌ها
- استفاده از `JSONB` برای داده‌های ساختار یافته (PostgreSQL)
- استفاده از `TEXT` برای رشته‌های طولانی

### 2. Naming Conventions

- نام جداول: `snake_case`
- نام ستون‌ها: `snake_case`
- Foreign Keys: `{table_name}_id`
- Indexes: `ix_{table_name}_{column_name}`

### 3. Performance Considerations

- Indexes روی ستون‌های پر استفاده
- Foreign Keys برای یکپارچگی داده
- استفاده از JSONB برای داده‌های انعطاف‌پذیر

---

## به‌روزرسانی Schema

برای تغییر schema، از [Database Migrations Guide](./DATABASE_MIGRATIONS.md) استفاده کنید.

---

## منابع

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)

