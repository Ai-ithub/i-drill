# 🗄️ Database Migrations Guide

راهنمای کامل مدیریت migrations دیتابیس با استفاده از Alembic

## 📋 فهرست مطالب

- [مقدمه](#مقدمه)
- [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
- [دستورات پایه](#دستورات-پایه)
- [ایجاد Migration جدید](#ایجاد-migration-جدید)
- [اعمال Migrations](#اعمال-migrations)
- [بازگشت Migrations](#بازگشت-migrations)
- [بهترین روش‌ها](#بهترین-روش‌ها)
- [عیب‌یابی](#عیب-یابی)

---

## مقدمه

این پروژه از **Alembic** برای مدیریت migrations دیتابیس استفاده می‌کند. Alembic یک ابزار قدرتمند برای version control دیتابیس است که تغییرات schema را به صورت versioned نگهداری می‌کند.

### ساختار فایل‌ها

```
src/backend/
├── alembic/
│   ├── versions/          # فایل‌های migration
│   │   └── 001_initial_schema.py
│   ├── env.py            # پیکربندی Alembic
│   └── script.py.mako    # Template برای migrations
├── alembic.ini           # تنظیمات Alembic
├── api/models/
│   └── database_models.py  # مدل‌های SQLAlchemy
└── scripts/
    └── manage_migrations.py  # اسکریپت مدیریت migrations
```

---

## نصب و راه‌اندازی

### 1. نصب Dependencies

Alembic در `requirements/backend.txt` موجود است:

```bash
pip install -r requirements/backend.txt
```

### 2. تنظیم DATABASE_URL

مطمئن شوید که متغیر محیطی `DATABASE_URL` تنظیم شده است:

```bash
# Windows PowerShell
$env:DATABASE_URL="postgresql://user:password@localhost:5432/drilling_db"

# Linux/Mac
export DATABASE_URL="postgresql://user:password@localhost:5432/drilling_db"

# یا در فایل .env
DATABASE_URL=postgresql://user:password@localhost:5432/drilling_db
```

### 3. بررسی وضعیت

```bash
cd src/backend
python scripts/manage_migrations.py current
```

---

## دستورات پایه

### استفاده از اسکریپت مدیریت

```bash
# نمایش کمک
python scripts/manage_migrations.py help

# نمایش وضعیت فعلی
python scripts/manage_migrations.py current

# نمایش تاریخچه migrations
python scripts/manage_migrations.py history

# ایجاد migration جدید
python scripts/manage_migrations.py create "description"

# اعمال migrations
python scripts/manage_migrations.py upgrade

# بازگشت migration
python scripts/manage_migrations.py downgrade
```

### استفاده مستقیم از Alembic

```bash
cd src/backend

# نمایش وضعیت فعلی
alembic current

# نمایش تاریخچه
alembic history

# ایجاد migration جدید
alembic revision --autogenerate -m "description"

# اعمال migrations
alembic upgrade head

# بازگشت یک migration
alembic downgrade -1
```

---

## ایجاد Migration جدید

### 1. تغییر مدل‌های SQLAlchemy

ابتدا مدل‌های دیتابیس را در `api/models/database_models.py` تغییر دهید:

```python
class NewTable(Base):
    __tablename__ = "new_table"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
```

### 2. ایجاد Migration

```bash
python scripts/manage_migrations.py create "add new_table"
```

یا:

```bash
alembic revision --autogenerate -m "add new_table"
```

### 3. بررسی Migration ایجاد شده

فایل migration در `alembic/versions/` ایجاد می‌شود. همیشه قبل از اعمال، آن را بررسی کنید:

```python
def upgrade() -> None:
    op.create_table(
        'new_table',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('new_table')
```

### 4. ویرایش دستی Migration (در صورت نیاز)

گاهی Alembic نمی‌تواند تمام تغییرات را تشخیص دهد. در این صورت باید migration را دستی ویرایش کنید.

---

## اعمال Migrations

### اعمال تمام Migrations

```bash
python scripts/manage_migrations.py upgrade
# یا
alembic upgrade head
```

### اعمال تا یک Revision خاص

```bash
alembic upgrade 001_initial
```

### بررسی قبل از اعمال

```bash
# نمایش SQL بدون اجرا
alembic upgrade head --sql
```

---

## بازگشت Migrations

### بازگشت یک Migration

```bash
python scripts/manage_migrations.py downgrade
# یا
alembic downgrade -1
```

### بازگشت به Revision خاص

```bash
alembic downgrade 001_initial
```

### بازگشت تمام Migrations

```bash
alembic downgrade base
```

⚠️ **هشدار**: این کار تمام جداول را حذف می‌کند!

---

## بهترین روش‌ها

### ✅ کارهایی که باید انجام دهید

1. **همیشه Backup بگیرید**
   ```bash
   pg_dump -U user -d drilling_db > backup.sql
   ```

2. **Migration ها را در Development تست کنید**
   - قبل از اعمال در Production، در Development تست کنید

3. **Migration ها را Review کنید**
   - همیشه فایل migration را قبل از اعمال بررسی کنید

4. **از پیام‌های واضح استفاده کنید**
   ```bash
   # ❌ بد
   alembic revision -m "update"
   
   # ✅ خوب
   alembic revision -m "add user preferences table"
   ```

5. **Migration ها را کوچک نگه دارید**
   - هر migration یک تغییر منطقی انجام دهد

6. **از Transaction استفاده کنید**
   - Alembic به صورت پیش‌فرض از transaction استفاده می‌کند

### ❌ کارهایی که نباید انجام دهید

1. **مستقیماً دیتابیس را تغییر ندهید**
   - همیشه از migrations استفاده کنید

2. **Migration های اعمال شده را تغییر ندهید**
   - اگر migration اعمال شده، migration جدید ایجاد کنید

3. **Migration ها را Skip نکنید**
   - همیشه به ترتیب اعمال شوند

4. **در Production بدون تست اعمال نکنید**

---

## عیب‌یابی

### مشکل: Migration اعمال نمی‌شود

```bash
# بررسی وضعیت فعلی
alembic current

# بررسی تاریخچه
alembic history

# بررسی SQL بدون اجرا
alembic upgrade head --sql
```

### مشکل: Conflict در Migration

اگر migration با دیتابیس فعلی conflict دارد:

```bash
# Stamp کردن دیتابیس با revision فعلی
alembic stamp head
```

### مشکل: Migration ناقص اعمال شده

```bash
# بازگشت به revision قبلی
alembic downgrade -1

# بررسی و اصلاح migration
# سپس دوباره اعمال کنید
alembic upgrade head
```

### مشکل: جداول از Alembic خارج هستند

اگر جداول به صورت دستی ایجاد شده‌اند:

```bash
# Stamp کردن با revision اولیه
alembic stamp 001_initial

# یا ایجاد migration خالی و stamp
alembic revision -m "initial state"
alembic stamp head
```

---

## مثال‌های کاربردی

### مثال 1: اضافه کردن ستون جدید

```python
# 1. تغییر مدل
class SensorData(Base):
    # ... existing columns ...
    new_field = Column(String(50), nullable=True)

# 2. ایجاد migration
alembic revision --autogenerate -m "add new_field to sensor_data"

# 3. بررسی و اعمال
alembic upgrade head
```

### مثال 2: اضافه کردن Index

```python
# در migration
def upgrade() -> None:
    op.create_index('ix_sensor_data_new_field', 'sensor_data', ['new_field'])

def downgrade() -> None:
    op.drop_index('ix_sensor_data_new_field', 'sensor_data')
```

### مثال 3: تغییر نوع داده

```python
# در migration
def upgrade() -> None:
    op.alter_column('sensor_data', 'status',
                    type_=sa.String(length=50),
                    existing_type=sa.String(length=20))

def downgrade() -> None:
    op.alter_column('sensor_data', 'status',
                    type_=sa.String(length=20),
                    existing_type=sa.String(length=50))
```

---

## Migration های موجود

### 001_initial_schema

این migration اولیه تمام جداول پایه را ایجاد می‌کند:

- `sensor_data` - داده‌های سنسور
- `users` - کاربران
- `maintenance_alerts` - هشدارهای تعمیرات
- `maintenance_schedules` - برنامه تعمیرات
- `password_reset_tokens` - توکن‌های بازنشانی رمز
- `blacklisted_tokens` - توکن‌های بلاک شده
- `login_attempts` - تلاش‌های ورود
- `change_requests` - درخواست‌های تغییر
- `dvr_process_history` - تاریخچه پردازش DVR
- `rul_predictions` - پیش‌بینی‌های RUL
- `anomaly_detections` - تشخیص‌های anomaly
- `model_versions` - ورژن‌های مدل
- `well_profiles` - پروفایل چاه‌ها
- `drilling_sessions` - جلسات حفاری
- `system_logs` - لاگ‌های سیستم

---

## منابع بیشتر

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

## پشتیبانی

در صورت بروز مشکل:
1. لاگ‌های Alembic را بررسی کنید
2. وضعیت دیتابیس را بررسی کنید
3. با تیم توسعه تماس بگیرید

