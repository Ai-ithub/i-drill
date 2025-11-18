# 🔒 Security Checks در CI/CD Pipeline

این مستندات راهنمای کامل security checks در CI/CD pipeline پروژه i-Drill است.

---

## 📋 خلاصه

Pipeline امنیتی شامل بررسی‌های جامع امنیتی برای شناسایی آسیب‌پذیری‌ها در کد، وابستگی‌ها، و محیط deployment است.

---

## 🔧 ابزارهای امنیتی استفاده شده

### 1. Bandit - Python Code Security Scanner

**هدف**: شناسایی مشکلات امنیتی در کد Python

**پیکربندی**: `.bandit`

**اجرا**:
```bash
bandit -r src/ -f screen -ll --exclude tests
```

**خروجی**: JSON report و console output

---

### 2. pip-audit - Dependency Vulnerability Scanner

**هدف**: بررسی آسیب‌پذیری‌های وابستگی‌های Python

**اجرا**:
```bash
pip-audit --requirement requirements/backend.txt --desc
```

**خروجی**: لیست آسیب‌پذیری‌ها با توضیحات

---

### 3. Safety - Python Dependency Security Check

**هدف**: بررسی آسیب‌پذیری‌های وابستگی‌ها با استفاده از Safety DB

**اجرا**:
```bash
safety check --file requirements/backend.txt
```

**خروجی**: JSON report

---

### 4. TruffleHog - Secret Scanning

**هدف**: شناسایی secrets و credentials در repository

**اجرا**: به صورت خودکار در workflow

**خروجی**: گزارش secrets احتمالی

---

### 5. Hadolint - Dockerfile Security Linter

**هدف**: بررسی Dockerfile برای best practices و مشکلات امنیتی

**اجرا**:
```bash
hadolint Dockerfile
```

**خروجی**: JSON report

---

### 6. Trivy - Container & File System Scanner

**هدف**: 
- Scan فایل سیستم برای آسیب‌پذیری‌ها
- Scan Docker image برای آسیب‌پذیری‌ها

**اجرا**:
```bash
trivy fs .
trivy image i-drill:latest
```

**خروجی**: SARIF format (uploaded to GitHub Security tab)

---

### 7. Semgrep - Static Analysis

**هدف**: Static analysis با قوانین OWASP و security audit

**پیکربندی**: 
- `p/security-audit`
- `p/python`
- `p/owasp-top-ten`

**خروجی**: SARIF format (uploaded to GitHub Security tab)

---

## 📁 فایل‌های Workflow

### 1. `security.yml` - Security Checks کامل

**موقعیت**: `.github/workflows/security.yml`

**Triggers**:
- Push به `main`, `develop`, `master`
- Pull requests
- Weekly schedule (یکشنبه‌ها ساعت 2 صبح UTC)
- Manual dispatch

**Jobs**:
1. `bandit-scan` - Python code security
2. `dependency-scan` - Dependency vulnerabilities (matrix: backend, ml, dev)
3. `safety-check` - Safety DB check
4. `secret-scan` - TruffleHog secret detection
5. `dockerfile-lint` - Hadolint Dockerfile check
6. `trivy-fs-scan` - Trivy file system scan
7. `trivy-docker-scan` - Trivy Docker image scan
8. `semgrep-scan` - Semgrep static analysis
9. `security-summary` - Summary report

---

### 2. `ci.yml` - Quick Security Scan

**موقعیت**: `.github/workflows/ci.yml`

**Job**: `security-scan`

**شامل**:
- Bandit (quick scan)
- pip-audit
- Safety
- Trivy (file system)

**هدف**: اجرای سریع security checks در هر CI run

---

## 🔄 اجرای Security Checks

### به صورت خودکار

Security checks به صورت خودکار اجرا می‌شوند:

1. **در هر Push** به branches اصلی
2. **در هر Pull Request**
3. **هفتگی** (یکشنبه‌ها ساعت 2 صبح UTC)

### به صورت دستی

```bash
# Trigger workflow manually via GitHub CLI
gh workflow run security.yml

# یا از GitHub Actions UI:
# Actions → Security Checks → Run workflow
```

---

## 📊 مشاهده نتایج

### 1. GitHub Security Tab

نتایج Trivy و Semgrep به GitHub Security tab ارسال می‌شوند:

**مسیر**: Repository → Security → Code scanning alerts

### 2. Workflow Artifacts

گزارش‌های کامل در workflow artifacts ذخیره می‌شوند:

**مسیر**: Actions → [Workflow run] → Artifacts

**Artifacts**:
- `bandit-report.json`
- `pip-audit-*.json`
- `safety-report.json`
- `trufflehog-results.json`
- `hadolint-report.json`
- `trivy-*-results.sarif`
- `semgrep.sarif`

### 3. Workflow Summary

خلاصه نتایج در workflow summary نمایش داده می‌شود.

---

## 🛠️ اجرای محلی Security Checks

### نصب ابزارها

```bash
# فعال کردن virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# نصب security tools
pip install -r requirements/dev.txt
```

### اجرای Bandit

```bash
cd i-drill
bandit -r src/ -f screen -ll
```

### اجرای pip-audit

```bash
cd i-drill
pip-audit --requirement requirements/backend.txt --desc
```

### اجرای Safety

```bash
cd i-drill
safety check --file requirements/backend.txt
```

### اجرای Hadolint

```bash
# نصب Hadolint (Docker)
docker run --rm -i hadolint/hadolint < Dockerfile

# یا نصب local (بسته به OS)
# macOS: brew install hadolint
# Linux: wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64
# chmod +x /usr/local/bin/hadolint
```

---

## ⚙️ پیکربندی

### Bandit Configuration

فایل `.bandit` در root پروژه:

```ini
[bandit]
exclude_dirs = tests,test,__pycache__,venv
min_severity = medium
min_confidence = medium
skips = B101,B601
```

### Trivy Configuration

تنظیمات در workflow:

- Severity threshold: `CRITICAL,HIGH`
- Format: `SARIF`
- Exit code: `0` (برای جلوگیری از fail شدن pipeline)

---

## 🐛 عیب‌یابی

### مشکل: Bandit پیدا کردن false positives

**راه‌حل**: استفاده از `# nosec` comment یا اضافه کردن test ID به `.bandit` config

```python
# Example: Skip specific Bandit check
password = os.getenv("PASSWORD")  # nosec B105
```

### مشکل: pip-audit پیدا کردن آسیب‌پذیری‌های قدیمی

**راه‌حل**: به‌روزرسانی dependencies

```bash
pip-audit --requirement requirements/backend.txt --desc
# سپس به‌روزرسانی پکیج‌های آسیب‌پذیر
```

### مشکل: Safety نیاز به API key

**راه‌حل**: استفاده از API key (اختیاری) یا استفاده از offline mode

```bash
export SAFETY_API_KEY=your-api-key
safety check
```

---

## 📝 Best Practices

### 1. بررسی منظم

- ✅ بررسی Security tab حداقل هفتگی
- ✅ بررسی workflow artifacts بعد از هر PR
- ✅ اجرای security checks قبل از merge

### 2. رفع آسیب‌پذیری‌ها

- ✅ اولویت با Critical و High severity
- ✅ رفع آسیب‌پذیری‌ها در PR جداگانه
- ✅ تست پس از رفع آسیب‌پذیری

### 3. پایش مستمر

- ✅ فعال کردن Dependabot برای dependency updates
- ✅ بررسی weekly scheduled scans
- ✅ بررسی GitHub Security Advisories

---

## 🔗 منابع بیشتر

### مستندات رسمی

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pip-audit Documentation](https://github.com/pypa/pip-audit)
- [Safety Documentation](https://pyup.io/safety/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Semgrep Documentation](https://semgrep.dev/docs/)
- [Hadolint Documentation](https://github.com/hadolint/hadolint)

### منابع امنیتی

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security.html)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

---

## ✅ چک‌لیست

- [ ] Security checks در CI pipeline فعال هستند
- [ ] Security tab در GitHub بررسی می‌شود
- [ ] Bandit config تنظیم شده است
- [ ] Artifacts بررسی می‌شوند
- [ ] Dependabot فعال است
- [ ] Weekly scans اجرا می‌شوند
- [ ] Security alerts بررسی می‌شوند

---

**آخرین به‌روزرسانی:** نوامبر 2025  
**وضعیت:** ✅ فعال و در حال اجرا

