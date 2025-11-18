# ⚡ راهنمای سریع اجرای بررسی امنیتی هفتگی

راهنمای سریع برای اجرای بررسی امنیتی هفتگی در پروژه i-Drill.

---

## 🚀 روش‌های اجرا

### 1️⃣ اجرای خودکار (Scheduled) ⏰

**زمان**: هر یکشنبه ساعت 2:00 AM UTC (05:30 صبح تهران)

**وضعیت**: ✅ فعال - اجرا می‌شود به صورت خودکار

**بررسی**:
```
https://github.com/[owner]/[repo]/actions/workflows/security.yml
```

---

### 2️⃣ اجرای دستی از GitHub CLI 🔧

```bash
# اجرای workflow
gh workflow run security.yml

# بررسی وضعیت
gh run list --workflow=security.yml

# مشاهده نتایج
gh run view --web
```

---

### 3️⃣ اجرای دستی از GitHub UI 🖱️

1. بروید به: `Actions` → `Security Checks`
2. کلیک روی `Run workflow`
3. Branch را انتخاب کنید (`main`)
4. کلیک روی `Run workflow`

---

### 4️⃣ اجرای محلی 🖥️

#### Windows (PowerShell):

```powershell
cd i-drill
.\scripts\run-weekly-security-scan.ps1
```

#### Linux/Mac (Bash):

```bash
cd i-drill
chmod +x scripts/run-weekly-security-scan.sh
./scripts/run-weekly-security-scan.sh
```

#### Manual (هر OS):

```bash
# نصب ابزارها
pip install -r requirements/dev.txt

# اجرای Bandit
bandit -r src/ -f screen -ll

# اجرای pip-audit
pip-audit --requirement requirements/backend.txt --desc

# اجرای Safety
safety check --file requirements/backend.txt
```

---

## 📊 مشاهده نتایج

### GitHub Actions UI:
```
Actions → Security Checks → [Latest run]
```

### GitHub Security Tab:
```
Security → Code scanning
```

### Artifacts:
```
Actions → [Run] → Artifacts
```

---

## ✅ چک‌لیست سریع

- [ ] Workflow به درستی اجرا شده است
- [ ] هیچ Critical/High severity issue جدیدی وجود ندارد
- [ ] نتایج در Security tab بررسی شده است
- [ ] Artifacts دانلود و بررسی شده است

---

## 📝 اطلاعات بیشتر

برای اطلاعات کامل به مستندات زیر مراجعه کنید:
- 📚 [WEEKLY_SECURITY_SCAN.md](./WEEKLY_SECURITY_SCAN.md) - راهنمای کامل
- 📚 [SECURITY_CI_CD.md](./SECURITY_CI_CD.md) - مستندات CI/CD

---

**سوالات؟** Issues ایجاد کنید یا به مستندات کامل مراجعه کنید.

