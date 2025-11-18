# 🔐 اصلاح امنیت SECRET_KEY

این سند تغییرات انجام شده برای حذف کامل SECRET_KEY پیش‌فرض را توضیح می‌دهد.

## ⚠️ مشکل امنیتی

قبلاً سیستم در حالت development به صورت خودکار یک SECRET_KEY موقت تولید می‌کرد. این رفتار خطرناک بود چون:
- ممکن بود توسعه‌دهندگان متوجه نشوند که از یک کلید موقت استفاده می‌کنند
- کلید موقت در هر restart تغییر می‌کرد که باعث مشکل در session management می‌شد
- امکان استفاده ناخواسته از کلید ناامن در production وجود داشت

## ✅ تغییرات انجام شده

### 1. حذف تولید خودکار SECRET_KEY

**فایل:** `src/backend/utils/security.py`

- تابع `get_or_generate_secret_key()` دیگر کلید موقت تولید نمی‌کند
- در صورت عدم تنظیم SECRET_KEY، خطا می‌دهد (حتی در development)
- پیام خطای واضح با دستورالعمل‌های تولید کلید امن

**قبل:**
```python
if not secret_key:
    if app_env == "production":
        raise RuntimeError(...)
    else:
        # Development mode - generate temporary key
        secret_key = generate_secret_key()
        logger.warning(...)
```

**بعد:**
```python
if not secret_key:
    raise RuntimeError(
        "SECRET_KEY environment variable is REQUIRED and must be set.\n"
        "No default values are allowed for security reasons.\n\n"
        "To generate a secure SECRET_KEY:\n"
        "  python scripts/generate_secret_key.py\n\n"
        ...
    )
```

### 2. بهبود Validation الگوهای ناامن

**فایل:** `src/backend/utils/security.py`

- افزودن الگوهای جدید به لیست الگوهای ناامن:
  - `dev-secret-change-me`
  - `dev-secret`
  - `change_this_to_a_secure_random_key_min_32_chars`
  - `placeholder`, `temp`, `temporary`
- تغییر رفتار: به جای warning، خطا می‌دهد (حتی در development)

### 3. ساده‌سازی Validation در app.py

**فایل:** `src/backend/app.py`

- حذف validation تکراری (چون `get_or_generate_secret_key()` خودش validate می‌کند)
- فقط بررسی طول کلید برای production باقی مانده است

### 4. بررسی فایل‌های Docker Compose

**فایل‌ها:** `docker-compose.yml`, `docker-compose.remote.yml`

- ✅ هیچ fallback برای SECRET_KEY وجود ندارد
- ✅ از `${SECRET_KEY}` استفاده می‌شود بدون مقدار پیش‌فرض
- ✅ کامنت‌های امنیتی اضافه شده است

## 📋 نحوه استفاده

### تولید SECRET_KEY

```bash
# روش 1: استفاده از اسکریپت
python scripts/generate_secret_key.py

# روش 2: استفاده مستقیم از Python
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

### تنظیم در .env

```bash
# فایل .env
SECRET_KEY=your-generated-secure-key-here-min-32-chars
```

### اجرای برنامه

```bash
# اگر SECRET_KEY تنظیم نشده باشد، برنامه خطا می‌دهد:
RuntimeError: SECRET_KEY environment variable is REQUIRED and must be set.
```

## 🔒 الزامات امنیتی

1. **SECRET_KEY باید حتماً تنظیم شود** - هیچ مقدار پیش‌فرضی وجود ندارد
2. **حداقل طول:** 32 کاراکتر
3. **الگوهای ممنوع:** 
   - `dev-secret-change-me`
   - `CHANGE_THIS_TO_A_SECURE_RANDOM_KEY_MIN_32_CHARS`
   - `change_this`, `placeholder`, `temp`, و غیره
4. **تولید:** باید از cryptographically secure random generator استفاده شود

## ✅ مزایای این تغییرات

1. **امنیت بیشتر:** هیچ کلید پیش‌فرضی وجود ندارد
2. **واضح بودن:** توسعه‌دهندگان مجبورند SECRET_KEY را تنظیم کنند
3. **یکنواختی:** کلید در تمام restartها یکسان است
4. **جلوگیری از خطا:** استفاده ناخواسته از کلید ناامن غیرممکن است

## 🧪 تست

برای تست اینکه SECRET_KEY به درستی کار می‌کند:

```bash
# تست 1: بدون SECRET_KEY (باید خطا بدهد)
unset SECRET_KEY
python -c "from utils.security import get_or_generate_secret_key; get_or_generate_secret_key()"
# Expected: RuntimeError

# تست 2: با SECRET_KEY معتبر
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
python -c "from utils.security import get_or_generate_secret_key; print('OK')"
# Expected: OK

# تست 3: با SECRET_KEY ناامن (باید خطا بدهد)
export SECRET_KEY="dev-secret-change-me"
python -c "from utils.security import get_or_generate_secret_key; get_or_generate_secret_key()"
# Expected: RuntimeError
```

## 📚 منابع

- [OWASP Secret Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [12-Factor App: Config](https://12factor.net/config)

