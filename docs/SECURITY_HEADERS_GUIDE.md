# 🔐 راهنمای Security Headers و CSP

این سند راهنمای کامل برای پیکربندی Security Headers و Content Security Policy (CSP) در پروژه i-Drill است.

---

## 📋 فهرست مطالب

1. [مقدمه](#مقدمه)
2. [Security Headers پیاده‌سازی شده](#security-headers-پیاده‌سازی-شده)
3. [Content Security Policy (CSP)](#content-security-policy-csp)
4. [پیکربندی](#پیکربندی)
5. [تست و اعتبارسنجی](#تست-و-اعتبارسنجی)
6. [بهترین روش‌ها](#بهترین-روش‌ها)

---

## مقدمه

Security Headers و CSP مکانیزم‌های امنیتی مهمی هستند که از حملات رایج وب مانند XSS، Clickjacking، و MIME sniffing جلوگیری می‌کنند.

### مزایا

- ✅ محافظت در برابر XSS (Cross-Site Scripting)
- ✅ جلوگیری از Clickjacking
- ✅ جلوگیری از MIME type sniffing
- ✅ کنترل دسترسی به منابع خارجی
- ✅ بهبود امنیت کلی برنامه

---

## Security Headers پیاده‌سازی شده

پروژه i-Drill به صورت خودکار Security Headers زیر را به تمام پاسخ‌های HTTP اضافه می‌کند:

### 1. X-Content-Type-Options

```
X-Content-Type-Options: nosniff
```

**هدف:** جلوگیری از MIME type sniffing

**توضیح:** مرورگر را مجبور می‌کند که Content-Type اعلام شده را بپذیرد و از sniffing خودکار جلوگیری می‌کند.

---

### 2. X-Frame-Options

```
X-Frame-Options: DENY
```

**هدف:** جلوگیری از Clickjacking

**توضیح:** از embed شدن صفحه در iframe جلوگیری می‌کند.

**مقادیر ممکن:**
- `DENY`: هیچ iframe مجاز نیست
- `SAMEORIGIN`: فقط iframe از همان origin مجاز است
- `ALLOW-FROM uri`: فقط iframe از URI مشخص شده مجاز است

---

### 3. X-XSS-Protection

```
X-XSS-Protection: 1; mode=block
```

**هدف:** فعال‌سازی XSS filter مرورگر

**توضیح:** در صورت تشخیص XSS، صفحه را block می‌کند.

---

### 4. Referrer-Policy

```
Referrer-Policy: strict-origin-when-cross-origin
```

**هدف:** کنترل اطلاعات ارسال شده در Referer header

**توضیح:** اطلاعات referrer را فقط در صورت نیاز و به صورت محدود ارسال می‌کند.

**مقادیر ممکن:**
- `no-referrer`: هیچ referrer ارسال نمی‌شود
- `strict-origin-when-cross-origin`: فقط origin در cross-origin requests
- `same-origin`: فقط برای same-origin requests

---

### 5. Content-Security-Policy (CSP)

```
Content-Security-Policy: default-src 'self'; script-src 'self'; ...
```

**هدف:** کنترل منابع قابل بارگذاری و اجرا

**توضیح:** مشخص می‌کند که چه منابعی (scripts, styles, images, etc.) می‌توانند بارگذاری شوند.

جزئیات بیشتر در بخش [CSP](#content-security-policy-csp) آمده است.

---

### 6. Permissions-Policy

```
Permissions-Policy: geolocation=(), microphone=(), camera=(), ...
```

**هدف:** کنترل دسترسی به API های مرورگر

**توضیح:** دسترسی به ویژگی‌های مرورگر مانند geolocation، camera، microphone را محدود می‌کند.

**ویژگی‌های غیرفعال شده:**
- `geolocation`: موقعیت جغرافیایی
- `microphone`: میکروفون
- `camera`: دوربین
- `payment`: Payment Request API
- `usb`: USB API
- `magnetometer`: مغناطیس‌سنج
- `gyroscope`: ژیروسکوپ
- `accelerometer`: شتاب‌سنج

---

### 7. Strict-Transport-Security (HSTS)

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

**هدف:** اجباری کردن HTTPS

**توضیح:** مرورگر را مجبور می‌کند که همیشه از HTTPS استفاده کند.

**فعال فقط در:**
- Production mode
- زمانی که `FORCE_HTTPS=true` تنظیم شده باشد

**پارامترها:**
- `max-age`: مدت زمان اعتبار (ثانیه)
- `includeSubDomains`: شامل subdomain ها
- `preload`: برای HSTS preload list

---

## Content Security Policy (CSP)

CSP یکی از مهم‌ترین Security Headers است که منابع قابل بارگذاری را کنترل می‌کند.

### CSP در Development

```http
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https: http:; font-src 'self' data:; connect-src 'self' ws: wss: http: https:; frame-ancestors 'self';
```

**ویژگی‌ها:**
- `unsafe-inline` و `unsafe-eval` برای HMR (Hot Module Replacement)
- اجازه HTTP و HTTPS برای development
- WebSocket support برای real-time updates

### CSP در Production

```http
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https://api.yourdomain.com wss://api.yourdomain.com; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; upgrade-insecure-requests;
```

**ویژگی‌ها:**
- بدون `unsafe-inline` یا `unsafe-eval` در script-src
- فقط HTTPS برای connect-src
- `upgrade-insecure-requests` برای ارتقای خودکار HTTP به HTTPS
- `frame-ancestors 'none'` برای جلوگیری از embed شدن

### Directives مهم

| Directive | توضیح | مثال |
|-----------|-------|------|
| `default-src` | منبع پیش‌فرض برای تمام directives | `'self'` |
| `script-src` | منابع مجاز برای JavaScript | `'self' 'unsafe-inline'` |
| `style-src` | منابع مجاز برای CSS | `'self' 'unsafe-inline'` |
| `img-src` | منابع مجاز برای تصاویر | `'self' data: https:` |
| `font-src` | منابع مجاز برای فونت‌ها | `'self' data:` |
| `connect-src` | منابع مجاز برای AJAX/WebSocket | `'self' https://api.example.com` |
| `frame-ancestors` | چه کسی می‌تواند صفحه را embed کند | `'none'` یا `'self'` |
| `base-uri` | مجاز برای `<base>` tag | `'self'` |
| `form-action` | مجاز برای `<form>` action | `'self'` |
| `upgrade-insecure-requests` | ارتقای خودکار HTTP به HTTPS | (بدون مقدار) |

---

## پیکربندی

### 1. پیکربندی از طریق Environment Variables

#### CSP Policy سفارشی

```env
# استفاده از CSP پیش‌فرض (توصیه می‌شود)
# CSP_POLICY=

# یا تعریف CSP سفارشی
CSP_POLICY="default-src 'self'; script-src 'self' https://cdn.example.com; style-src 'self' 'unsafe-inline';"
```

#### API URL برای CSP

```env
# در production، برای allow کردن API و WebSocket connections
API_URL=https://api.yourdomain.com
```

#### HSTS Configuration

```env
# فعال کردن HTTPS redirect
FORCE_HTTPS=true

# تنظیمات HSTS
HSTS_MAX_AGE=31536000  # 1 year
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=false  # فقط اگر می‌خواهید به HSTS preload list اضافه شوید
```

### 2. پیکربندی در Frontend

CSP در `frontend/index.html` به صورت meta tag اضافه شده است:

```html
<meta http-equiv="Content-Security-Policy" content="..." />
```

**نکته:** در production، بهتر است CSP از طریق HTTP header (از backend) ارسال شود تا meta tag.

---

## تست و اعتبارسنجی

### 1. بررسی Headers با curl

```bash
curl -I https://api.yourdomain.com/api/v1/health
```

خروجی باید شامل تمام Security Headers باشد.

### 2. استفاده از ابزارهای آنلاین

- **SecurityHeaders.com**: https://securityheaders.com
- **Mozilla Observatory**: https://observatory.mozilla.org

### 3. بررسی CSP Violations

در Console مرورگر، در صورت violation، خطا نمایش داده می‌شود:

```
Content Security Policy: The page's settings blocked the loading of a resource at ...
```

### 4. تست با Browser DevTools

1. باز کردن DevTools (F12)
2. رفتن به تب Network
3. انتخاب یک request
4. بررسی Response Headers

---

## بهترین روش‌ها

### 1. Development vs Production

- **Development:** CSP ساده‌تر برای HMR و debugging
- **Production:** CSP سخت‌گیرانه برای امنیت بیشتر

### 2. CSP Reporting

برای دریافت گزارش violations:

```http
Content-Security-Policy: ...; report-uri /api/v1/csp-report
```

یا استفاده از `report-to`:

```http
Content-Security-Policy: ...; report-to csp-endpoint
Report-To: {"group": "csp-endpoint", "max_age": 10886400, "endpoints": [{"url": "/api/v1/csp-report"}]}
```

### 3. Nonce برای Inline Scripts

به جای `unsafe-inline`، از nonce استفاده کنید:

```python
# در backend
nonce = secrets.token_urlsafe(16)
csp = f"script-src 'self' 'nonce-{nonce}'"
```

```html
<!-- در frontend -->
<script nonce="{{ nonce }}">...</script>
```

### 4. Hash برای Inline Scripts

استفاده از hash برای inline scripts:

```http
Content-Security-Policy: script-src 'self' 'sha256-abc123...'
```

### 5. تست تدریجی CSP

1. شروع با CSP ساده
2. بررسی violations در console
3. اضافه کردن exceptions مورد نیاز
4. سخت‌گیرانه‌تر کردن تدریجی

### 6. Monitoring

- ثبت CSP violations در logs
- Alert در صورت violations زیاد
- بررسی منظم Security Headers

---

## مثال‌های CSP

### CSP ساده (Development)

```
default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';
```

### CSP متوسط (Staging)

```
default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://api.example.com;
```

### CSP سخت‌گیرانه (Production)

```
default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https://api.example.com wss://api.example.com; frame-ancestors 'none'; base-uri 'self'; form-action 'self'; upgrade-insecure-requests;
```

---

## عیب‌یابی

### مشکل: CSP block می‌کند منابع معتبر

**راه‌حل:**
1. بررسی console برای violation message
2. اضافه کردن source به directive مربوطه
3. یا استفاده از `report-only` mode برای تست

### مشکل: WebSocket connections block می‌شوند

**راه‌حل:**
```http
connect-src 'self' wss://api.yourdomain.com
```

### مشکل: Inline styles کار نمی‌کنند

**راه‌حل:**
```http
style-src 'self' 'unsafe-inline'
```

یا استفاده از nonce/hash برای inline styles.

---

## منابع بیشتر

- [MDN: Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [OWASP: Content Security Policy](https://cheatsheetseries.owasp.org/cheatsheets/Content_Security_Policy_Cheat_Sheet.html)
- [SecurityHeaders.com](https://securityheaders.com)
- [CSP Evaluator](https://csp-evaluator.withgoogle.com/)

---

**آخرین به‌روزرسانی:** 2024  
**نسخه:** 1.0

