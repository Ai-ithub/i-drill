# برنامه به‌روزرسانی 22 پکیج

## 📊 خلاصه وضعیت

- **تعداد پکیج‌های نیاز به آپدیت**: 22 پکیج
- **پکیج‌های Patch/Minor**: 5 پکیج (بی‌خطر)
- **پکیج‌های Major**: 17 پکیج (نیاز به بررسی)

---

## 📋 لیست پکیج‌های نیاز به آپدیت

### ✅ پکیج‌های Patch/Minor (بی‌خطر - آپدیت مستقیم)

1. **@tanstack/react-query**: `5.90.7` → `5.90.8` (Patch)
2. **axios**: `1.13.1` → `1.13.2` (Patch)
3. **autoprefixer**: `10.4.21` → `10.4.22` (Patch)
4. **@vitejs/plugin-react**: `4.7.0` → `5.1.1` (Major - اما سازگار با React 19)

### ⚠️ پکیج‌های Major (نیاز به بررسی Breaking Changes)

#### 1. React Ecosystem
- **react**: `18.3.1` → `19.2.0` (Major) ⚠️
- **react-dom**: `18.3.1` → `19.2.0` (Major) ⚠️
- **@types/react**: `18.3.26` → `19.2.4` (Major) ⚠️
- **@types/react-dom**: `18.3.7` → `19.2.3` (Major) ⚠️

#### 2. Styling
- **tailwindcss**: `3.4.18` → `4.1.17` (Major) ⚠️
- **postcss**: `8.4.32` → نیاز به بررسی آخرین نسخه

#### 3. Build Tools
- **vite**: `5.4.21` → `7.2.2` (Major - در package.json: 7.2.2) ⚠️
- **@vitejs/plugin-react**: `4.7.0` → `5.1.1` (Major) ⚠️

#### 4. Linting & TypeScript
- **eslint**: `8.57.1` → `9.39.1` (Major) ⚠️
- **@typescript-eslint/eslint-plugin**: `6.21.0` → `8.46.4` (Major) ⚠️
- **@typescript-eslint/parser**: `6.21.0` → `8.46.4` (Major) ⚠️
- **eslint-plugin-react-hooks**: `4.6.2` → `7.0.1` (Major) ⚠️

#### 5. Testing
- **@testing-library/react**: `14.3.1` → `16.3.0` (Major) ⚠️
- **vitest**: `1.6.1` → `4.0.8` (Major) ⚠️
- **jsdom**: `24.1.3` → `27.2.0` (Major) ⚠️

#### 6. UI Libraries & Utilities
- **date-fns**: `2.30.0` → `4.1.0` (Major) ⚠️
- **lucide-react**: `0.294.0` → `0.553.0` (Major) ⚠️
- **recharts**: `2.15.4` → `3.4.1` (Major) ⚠️
- **zustand**: `4.5.7` → `5.0.8` (Major) ⚠️
- **react-router-dom**: `6.30.1` → `7.9.5` (Major) ⚠️

#### 7. TypeScript
- **@types/node**: `20.19.24` → `24.10.1` (Major) ⚠️

---

## 📝 استراتژی به‌روزرسانی

### مرحله 1: آپدیت پکیج‌های Patch/Minor (بی‌خطر)

```bash
npm install --save-dev @tanstack/react-query@latest axios@latest autoprefixer@latest
```

### مرحله 2: آپدیت React Ecosystem (با احتیاط)

```bash
# آپدیت React به 19
npm install react@^19.2.0 react-dom@^19.2.0
npm install --save-dev @types/react@^19 @types/react-dom@^19

# آپدیت Vite React Plugin
npm install --save-dev @vitejs/plugin-react@^5.1.1
```

**نکته:** نیاز به بررسی breaking changes (راهنمای REACT_19_MIGRATION_GUIDE.md)

### مرحله 3: آپدیت Tailwind CSS (با احتیاط)

```bash
npm install --save-dev tailwindcss@^4.1.17
```

**نکته:** نیاز به تغییر syntax در `index.css` (راهنمای TAILWIND_CSS_4_MIGRATION_GUIDE.md)

### مرحله 4: آپدیت Build Tools

```bash
# Vite (احتمالاً قبلاً در package.json است)
npm install --save-dev vite@^7.2.2
```

### مرحله 5: آپدیت Linting Tools (با احتیاط)

```bash
# ESLint 9 (breaking changes)
npm install --save-dev eslint@^9.39.1

# TypeScript ESLint plugins
npm install --save-dev @typescript-eslint/eslint-plugin@^8.46.4 @typescript-eslint/parser@^8.46.4

# React Hooks ESLint plugin
npm install --save-dev eslint-plugin-react-hooks@^7.0.1
```

**نکته:** ESLint 9 نیاز به config جدید دارد (Flat Config)

### مرحله 6: آپدیت Testing Tools (با احتیاط)

```bash
# Testing Library
npm install --save-dev @testing-library/react@^16.3.0

# Vitest
npm install --save-dev vitest@^4.0.8

# JSDOM
npm install --save-dev jsdom@^27.2.0
```

**نکته:** @testing-library/react 16 نیاز به React 19 دارد

### مرحله 7: آپدیت UI Libraries (با احتیاط)

```bash
# Date utilities
npm install date-fns@^4.1.0

# Icons
npm install lucide-react@^0.553.0

# Charts
npm install recharts@^3.4.1

# State management
npm install zustand@^5.0.8

# Routing
npm install react-router-dom@^7.9.5

# Node types
npm install --save-dev @types/node@^24.10.1
```

---

## ⚠️ Breaking Changes مهم

### 1. React 19
- تغییر در StrictMode behavior
- تغییر در TypeScript types
- نیاز به @vitejs/plugin-react 5.x

### 2. Tailwind CSS 4
- تغییر syntax: `@tailwind` → `@import "tailwindcss"`
- حذف پشتیبانی از مرورگرهای قدیمی

### 3. ESLint 9
- Flat Config (eslint.config.js) به جای .eslintrc
- تغییرات در plugin system

### 4. Vite 7
- تغییرات در plugin system
- بهبود performance

### 5. date-fns 4
- تغییرات در API
- نیاز به بررسی مستندات

### 6. react-router-dom 7
- تغییرات در API
- نیاز به بررسی مستندات

---

## 📋 Checklist به‌روزرسانی

### قبل از شروع
- [ ] Backup از پروژه
- [ ] ایجاد branch جدید
- [ ] بررسی breaking changes
- [ ] مطالعه راهنماهای migration

### مراحل آپدیت
- [ ] آپدیت Patch/Minor packages
- [ ] آپدیت React Ecosystem
- [ ] آپدیت Tailwind CSS
- [ ] آپدیت Build Tools
- [ ] آپدیت Linting Tools
- [ ] آپدیت Testing Tools
- [ ] آپدیت UI Libraries

### بعد از آپدیت
- [ ] اجرای `npm install`
- [ ] بررسی خطاهای TypeScript: `npm run type-check`
- [ ] بررسی خطاهای Lint: `npm run lint`
- [ ] Build پروژه: `npm run build`
- [ ] اجرای تست‌ها: `npm test`
- [ ] تست دستی در dev mode: `npm run dev`
- [ ] بررسی تمام صفحات
- [ ] بررسی Dark Mode
- [ ] بررسی Responsive Design

---

## 🚀 دستورات سریع

### آپدیت همه (با احتیاط - پیشنهاد نمی‌شود)

```bash
# آپدیت همه به latest (خطرناک!)
npm update

# یا
npx npm-check-updates -u
npm install
```

**نکته:** این روش توصیه نمی‌شود چون ممکن است breaking changes داشته باشد.

### آپدیت تدریجی (توصیه می‌شود)

```bash
# ابتدا Patch/Minor
npm install axios@latest autoprefixer@latest

# سپس React (با بررسی breaking changes)
npm install react@^19.2.0 react-dom@^19.2.0

# و به همین ترتیب...
```

---

## 📚 منابع

- [React 19 Migration Guide](./REACT_19_MIGRATION_GUIDE.md)
- [Tailwind CSS 4 Migration Guide](./TAILWIND_CSS_4_MIGRATION_GUIDE.md)
- [ESLint 9 Migration Guide](https://eslint.org/docs/latest/use/migrate-to-9.0.0)
- [Vite 7 Migration Guide](https://vitejs.dev/guide/migration.html)

---

## ✅ پیشنهاد

**استراتژی پیشنهادی:**

1. **گام 1**: ابتدا پکیج‌های Patch/Minor را آپدیت کنید
2. **گام 2**: React 19 را با دقت آپدیت کنید (راهنمای موجود است)
3. **گام 3**: Tailwind CSS 4 را آپدیت کنید (راهنمای موجود است)
4. **گام 4**: بقیه پکیج‌ها را به تدریج آپدیت کنید
5. **گام 5**: تست کامل انجام دهید

**اولویت‌بندی:**
1. 🔴 مهم و لازم: React 19, Tailwind 4, Vite 7
2. 🟡 مهم: ESLint 9, TypeScript ESLint 8
3. 🟢 مفید: UI Libraries, Testing Tools

موفق باشید! 🚀

