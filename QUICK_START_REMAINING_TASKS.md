# 🚀 راهنمای سریع شروع کار روی موارد باقی‌مانده

**تاریخ:** 2025-01-27  
**وضعیت:** آماده برای شروع

---

## 📊 خلاصه سریع

**موارد باقی‌مانده:** 4 مورد  
**زمان تخمینی:** 44-60 ساعت (~5.5-7.5 روز کاری)  
**پیشرفت کلی:** ~85%

---

## 🎯 اولویت‌بندی برای شروع

### 1️⃣ راه‌اندازی E2E Tests (اولویت بالا)
**زمان:** 12-16 ساعت

**مراحل:**
```bash
# 1. نصب Playwright
cd frontend
npm install -D @playwright/test

# 2. راه‌اندازی اولیه
npx playwright install

# 3. ایجاد config
# ایجاد playwright.config.ts

# 4. نوشتن اولین تست
# tests/e2e/auth.spec.ts
```

**فایل‌های مورد نیاز:**
- `frontend/playwright.config.ts`
- `frontend/tests/e2e/auth.spec.ts`
- `frontend/tests/e2e/dashboard.spec.ts`
- به‌روزرسانی `.github/workflows/ci.yml`

---

### 2️⃣ ارتقا به ESLint 9 (اولویت متوسط)
**زمان:** 6-8 ساعت

**مراحل:**
```bash
# 1. به‌روزرسانی ESLint
cd frontend
npm install -D eslint@^9.39.1

# 2. تبدیل config
# تبدیل .eslintrc.cjs به eslint.config.js

# 3. به‌روزرسانی plugins
npm install -D @typescript-eslint/eslint-plugin@latest
npm install -D eslint-plugin-react-hooks@latest
npm install -D eslint-plugin-react-refresh@latest

# 4. تست
npm run lint
```

**فایل‌های تغییر یافته:**
- `frontend/.eslintrc.cjs` → حذف
- `frontend/eslint.config.js` → ایجاد
- `frontend/package.json` → به‌روزرسانی

---

### 3️⃣ به‌روزرسانی UI Libraries (اولویت پایین)
**زمان:** 18-24 ساعت

**ترتیب پیشنهادی:**
1. lucide-react (ساده‌ترین - 2-3 ساعت)
2. date-fns (متوسط - 4-6 ساعت)
3. recharts (پیچیده - 6-8 ساعت)
4. react-router-dom (متوسط - 6-8 ساعت)

**مراحل برای هر library:**
```bash
# 1. بررسی breaking changes
# مطالعه changelog

# 2. به‌روزرسانی
npm install library@latest

# 3. تست
npm run build
npm run test

# 4. رفع خطاها
# در صورت نیاز
```

---

### 4️⃣ به‌روزرسانی Testing Tools (اولویت پایین)
**زمان:** 8-12 ساعت

**مراحل:**
```bash
# 1. به‌روزرسانی @testing-library/react
npm install -D @testing-library/react@^16.3.0

# 2. به‌روزرسانی vitest
npm install -D vitest@^4.0.8

# 3. به‌روزرسانی config
# بررسی vitest.config.ts

# 4. تست
npm run test
```

---

## 📝 چک‌لیست سریع

### برای شروع هر کار:
- [ ] ایجاد branch جدید: `git checkout -b feature/task-name`
- [ ] Backup گرفتن از تغییرات مهم
- [ ] مطالعه مستندات مربوطه
- [ ] اجرای تست‌های موجود: `npm run test` / `pytest`
- [ ] شروع پیاده‌سازی
- [ ] تست کامل
- [ ] Commit و Push
- [ ] ایجاد Pull Request

---

## 🔗 منابع مفید

### E2E Testing:
- [Playwright Documentation](https://playwright.dev/)
- [Playwright React Guide](https://playwright.dev/docs/react)

### ESLint 9:
- [ESLint 9 Migration Guide](https://eslint.org/docs/latest/use/migrate-to-9.0.0)
- [Flat Config Format](https://eslint.org/docs/latest/use/configure/configuration-files-new)

### Library Updates:
- [date-fns v4 Migration](https://date-fns.org/docs/Upgrade-Guide)
- [recharts v3 Migration](https://recharts.org/en-US/migration-guide)
- [React Router v7 Migration](https://reactrouter.com/en/main/upgrading/v7)

---

## ⚠️ نکات مهم

1. **یک کار در یک زمان:** هر به‌روزرسانی را جداگانه انجام دهید
2. **تست کامل:** بعد از هر تغییر، تمام تست‌ها را اجرا کنید
3. **Backup:** قبل از تغییرات بزرگ backup بگیرید
4. **Branch:** هر کار را در branch جداگانه انجام دهید
5. **Documentation:** مستندات را به‌روز کنید

---

**آماده برای شروع!** 🚀

