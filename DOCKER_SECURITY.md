# 🔐 Docker Security Guide

این راهنما نحوه تنظیم امن Docker Compose را برای i-Drill توضیح می‌دهد.

## ⚠️ هشدارهای امنیتی

**هرگز فایل `.env` را به repository commit نکنید!**

تمام رمزهای عبور و اطلاعات حساس باید در فایل `.env` قرار گیرند که در `.gitignore` است.

## 📋 راه‌اندازی اولیه

### 1. ایجاد فایل `.env`

```bash
cp docker-compose.env.example .env
```

### 2. ویرایش فایل `.env`

فایل `.env` را باز کرده و تمام رمزهای عبور پیش‌فرض را تغییر دهید:

```bash
# حداقل 16 کاراکتر برای رمز عبور دیتابیس
POSTGRES_PASSWORD=YourStrongPassword123!@#

# حداقل 32 کاراکتر برای SECRET_KEY
SECRET_KEY=YourSecureRandomKey32CharactersMinimum!

# حداقل 12 کاراکتر برای رمز عبور ادمین
DEFAULT_ADMIN_PASSWORD=YourAdminPassword123!

# رمز عبور Grafana
GF_SECURITY_ADMIN_PASSWORD=YourGrafanaPassword123!
```

### 3. تولید SECRET_KEY امن

برای تولید یک SECRET_KEY امن:

```bash
python scripts/generate_secret_key.py
```

یا از Python:

```python
import secrets
print(secrets.token_urlsafe(32))
```

## 🔒 متغیرهای محیطی اجباری

این متغیرها **باید** در فایل `.env` تنظیم شوند (بدون مقدار پیش‌فرض):

- `POSTGRES_PASSWORD` - رمز عبور دیتابیس PostgreSQL
- `SECRET_KEY` - کلید مخفی برای JWT tokens
- `DEFAULT_ADMIN_PASSWORD` - رمز عبور حساب ادمین پیش‌فرض
- `GF_SECURITY_ADMIN_PASSWORD` - رمز عبور Grafana (برای monitoring)

## 📝 متغیرهای محیطی اختیاری

این متغیرها می‌توانند در `.env` تنظیم شوند یا از مقادیر پیش‌فرض استفاده کنند:

- `POSTGRES_DB` - نام دیتابیس (پیش‌فرض: `drilling_db`)
- `POSTGRES_USER` - نام کاربری دیتابیس (پیش‌فرض: `drill_user`)
- `REDIS_PASSWORD` - رمز عبور Redis (اختیاری، توصیه می‌شود برای production)
- `DEFAULT_ADMIN_USERNAME` - نام کاربری ادمین (پیش‌فرض: `admin`)
- `DEFAULT_ADMIN_EMAIL` - ایمیل ادمین (پیش‌فرض: `admin@example.com`)
- `GF_SECURITY_ADMIN_USER` - نام کاربری Grafana (پیش‌فرض: `admin`)

## 🚀 استفاده

### Development (Local Services)

```bash
docker-compose up -d
```

### Production (Remote Services)

```bash
docker-compose -f docker-compose.remote.yml up -d
```

### Monitoring Stack

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

## ✅ بررسی امنیت

قبل از deploy در production، این موارد را بررسی کنید:

1. ✅ فایل `.env` در `.gitignore` است
2. ✅ تمام رمزهای عبور پیش‌فرض تغییر کرده‌اند
3. ✅ `SECRET_KEY` حداقل 32 کاراکتر است
4. ✅ `POSTGRES_PASSWORD` حداقل 16 کاراکتر است
5. ✅ `DEFAULT_ADMIN_PASSWORD` حداقل 12 کاراکتر است
6. ✅ `GF_SECURITY_ADMIN_PASSWORD` تغییر کرده است
7. ✅ `APP_ENV=production` تنظیم شده است

## 🔍 بررسی رمزهای عبور در فایل‌های Docker Compose

برای اطمینان از اینکه هیچ رمز عبور پیش‌فرضی در فایل‌های docker-compose باقی نمانده است:

```bash
# بررسی رمزهای عبور پیش‌فرض
grep -r "password\|PASSWORD" docker-compose*.yml | grep -v "POSTGRES_PASSWORD\|REDIS_PASSWORD\|SECRET_KEY\|DEFAULT_ADMIN_PASSWORD\|GF_SECURITY_ADMIN_PASSWORD\|KAFKA.*PASSWORD"

# بررسی SECRET_KEY پیش‌فرض
grep -r "dev-secret-change-me\|CHANGE_THIS" docker-compose*.yml
```

## 📚 منابع بیشتر

- [Docker Secrets](https://docs.docker.com/engine/swarm/secrets/)
- [Environment Variables Best Practices](https://12factor.net/config)
- [OWASP Docker Security](https://owasp.org/www-project-docker-security/)

