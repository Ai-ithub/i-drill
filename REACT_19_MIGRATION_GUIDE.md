# راهنمای ارتقا به React 19.2.0

## 📊 وضعیت فعلی پروژه

- **React فعلی**: 18.2.0 (در package.json)
- **React هدف**: 19.2.0
- **React DOM فعلی**: 18.2.0
- **TypeScript**: 5.2.2
- **Vite**: 5.0.8
- **JSX Transform**: فعال (`"jsx": "react-jsx"` در tsconfig.json)

---

## 🔍 بررسی Breaking Changes در React 19

### ✅ موارد سازگار در پروژه

1. **JSX Transform جدید**: ✅ از قبل فعال است
   - `tsconfig.json` دارای `"jsx": "react-jsx"` است
   - نیاز به تغییر ندارد

2. **ReactDOM.createRoot**: ✅ استفاده می‌شود
   - `main.tsx` از `ReactDOM.createRoot` استفاده می‌کند
   - `ReactDOM.render` قدیمی استفاده نشده است

3. **استفاده از useMemo و useCallback**: ✅ استفاده شده
   - در 18 فایل استفاده شده است
   - **توجه**: در React 19 و StrictMode، رفتار متفاوت است (ممویزیشن در رندر دوم حفظ می‌شود)

---

## ⚠️ تغییرات مهم که نیاز به بررسی دارند

### 1. تغییرات TypeScript Types

**Types حذف شده:**
- `React.ReactChild` → جایگزین: `React.ReactElement | number | string`
- `React.ReactFragment` → جایگزین: `Iterable<React.ReactNode>`
- `React.ReactText` → جایگزین: `number | string`
- `VoidFunctionComponent` → جایگزین: `FunctionComponent`

**بررسی پروژه:**
```bash
# بررسی استفاده از types قدیمی
grep -r "ReactChild\|ReactFragment\|ReactText" src/
```

**نتیجه**: ✅ هیچ استفاده‌ای از types قدیمی یافت نشد

---

### 2. تغییر در رفتار StrictMode

**تغییرات:**
- `useMemo` و `useCallback`: در StrictMode، اولین رندر را در رندر دوم نیز استفاده می‌کنند
- **Ref callbacks**: در mount اولیه دو بار فراخوانی می‌شوند

**فایل‌های استفاده‌کننده از useMemo/useCallback:**
- ✅ 18 فایل استفاده می‌کنند
- نیاز به تست دقیق در StrictMode

**فایل‌های استفاده‌کننده از refs:**
- `src/hooks/useWebSocket.ts`: استفاده از `useRef` برای WebSocket
- `src/components/Notifications/NotificationBadge.tsx`: استفاده از ref

**اقدامات مورد نیاز:**
- تست دقیق رفتار `useMemo` و `useCallback` در StrictMode
- بررسی اینکه ref callbacks دو بار فراخوانی نمی‌شوند (در صورت وجود)

---

### 3. تغییر در مدیریت خطاها (Error Handling)

**تغییرات:**
- خطاهای کشف‌نشده → به `window.reportError` گزارش می‌شوند
- خطاهای کشف‌شده توسط Error Boundary → به `console.error` گزارش می‌شوند

**Error Boundary موجود:**
- ✅ `src/components/ErrorBoundary.tsx` وجود دارد
- ✅ استفاده می‌شود در `main.tsx`

**اقدامات مورد نیاز:**
- بررسی اینکه اگر از `window.reportError` استفاده می‌کنید، رفتار مورد انتظار را داشته باشد
- تست Error Boundary برای اطمینان از عملکرد صحیح

---

### 4. سازگاری با Dependencies

| Package | Version فعلی | سازگاری با React 19 | وضعیت |
|---------|-------------|---------------------|-------|
| `@tanstack/react-query` | ^5.62.7 | ✅ سازگار | نیاز به بررسی |
| `zustand` | ^4.4.7 | ✅ سازگار | نیاز به بررسی |
| `recharts` | ^2.10.3 | ✅ سازگار | نیاز به بررسی |
| `react-router-dom` | ^6.20.0 | ✅ سازگار | نیاز به بررسی |
| `lucide-react` | ^0.294.0 | ✅ سازگار | نیاز به بررسی |
| `@vitejs/plugin-react` | ^4.2.1 | ⚠️ نیاز به آپدیت | **مهم** |

**اقدامات مورد نیاز:**
- آپدیت `@vitejs/plugin-react` به آخرین نسخه برای پشتیبانی کامل React 19

---

## 📋 مراحل ارتقا

### مرحله 1: آپدیت Dependencies

```bash
cd i-drill/frontend

# 1. آپدیت @vitejs/plugin-react (مهم!)
npm install -D @vitejs/plugin-react@latest

# 2. آپدیت React و React DOM
npm install react@^19.2.0 react-dom@^19.2.0

# 3. آپدیت TypeScript types
npm install -D @types/react@^19 @types/react-dom@^19
```

### مرحله 2: اجرای Codemod برای TypeScript Types (اختیاری)

```bash
# اگر از types قدیمی استفاده می‌کردید، این را اجرا کنید
npx types-react-codemod@latest preset-19 ./src
```

**نتیجه:** احتمالاً نیاز نیست چون از types قدیمی استفاده نشده است.

### مرحله 3: بررسی و تست

```bash
# بررسی TypeScript errors
npm run type-check

# بررسی lint errors
npm run lint

# اجرای تست‌ها
npm test

# اجرای dev server
npm run dev
```

### مرحله 4: تست دستی موارد مهم

1. **تست StrictMode:**
   - بررسی عملکرد `useMemo` و `useCallback`
   - بررسی رفتار ref callbacks

2. **تست Error Boundaries:**
   - ایجاد خطا در کامپوننت‌ها
   - بررسی اینکه Error Boundary به درستی کار می‌کند

3. **تست WebSocket:**
   - بررسی اتصال WebSocket
   - بررسی `useRef` در `useWebSocket.ts`

4. **تست Real-time Updates:**
   - بررسی عملکرد Recharts با React 19
   - بررسی React Query hooks

---

## ⚡ تغییرات سریع پیشنهادی

### 1. آپدیت vite.config.ts (اگر لازم باشد)

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react({
      // React 19 از JSX runtime جدید استفاده می‌کند
      jsxRuntime: 'automatic',
    }),
  ],
  // ... rest of config
})
```

### 2. بررسی main.tsx

✅ از قبل از `ReactDOM.createRoot` استفاده می‌کند - نیاز به تغییر ندارد

---

## 🚨 موارد احتیاطی

1. **Backup**: قبل از ارتقا، از پروژه backup بگیرید
2. **Branch**: روی branch جداگانه کار کنید
3. **تست کامل**: تمام صفحات و قابلیت‌ها را تست کنید
4. **Dependencies**: بعد از ارتقا، dependencies را آپدیت کنید

---

## 📝 Checklist ارتقا

- [ ] Backup از پروژه
- [ ] ایجاد branch جدید
- [ ] آپدیت `@vitejs/plugin-react`
- [ ] آپدیت React و React DOM به 19.2.0
- [ ] آپدیت `@types/react` و `@types/react-dom`
- [ ] اجرای `npm run type-check`
- [ ] اجرای `npm run lint`
- [ ] تست StrictMode behavior
- [ ] تست Error Boundaries
- [ ] تست WebSocket connections
- [ ] تست تمام صفحات اصلی
- [ ] تست Real-time monitoring
- [ ] تست Charts و Recharts
- [ ] تست React Query hooks
- [ ] بررسی performance

---

## 📚 منابع

- [React 19 Upgrade Guide](https://react.dev/blog/2024/04/25/react-19-upgrade-guide)
- [React 19 Release Notes](https://react.dev/blog/2024/12/05/react-19)
- [Breaking Changes in React 19](https://github.com/facebook/react/blob/main/CHANGELOG.md)

---

## ✅ نتیجه‌گیری

**وضعیت کلی پروژه برای ارتقا: بسیار خوب** ✅

- اکثر تغییرات لازم از قبل انجام شده است
- JSX Transform فعال است
- ReactDOM.createRoot استفاده می‌شود
- Types قدیمی استفاده نشده‌اند
- Error Boundary موجود است

**اقدامات اصلی:**
1. آپدیت `@vitejs/plugin-react` به آخرین نسخه
2. آپدیت React و React DOM به 19.2.0
3. آپدیت TypeScript types
4. تست کامل تمام قابلیت‌ها

**ریسک ارتقا: پایین تا متوسط** ⚠️

موفق باشید! 🚀

