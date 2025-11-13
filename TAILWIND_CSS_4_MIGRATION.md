# Tailwind CSS 4.0 Migration Guide

راهنمای به‌روزرسانی به Tailwind CSS 4.0

## تغییرات اعمال شده

### 1. به‌روزرسانی Package

```json
{
  "devDependencies": {
    "tailwindcss": "^4.0.0"
  }
}
```

### 2. تغییر Syntax در CSS

**قبل (v3):**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**بعد (v4):**
```css
@import "tailwindcss";
```

### 3. پیکربندی

Tailwind CSS 4.0 از CSS-based configuration پشتیبانی می‌کند، اما `tailwind.config.js` همچنان پشتیبانی می‌شود.

برای استفاده از CSS-based configuration:

```css
@import "tailwindcss";

@theme {
  --color-primary-50: #eff6ff;
  --color-primary-100: #dbeafe;
  /* ... */
}
```

## ویژگی‌های جدید

### 1. موتور Oxide (Rust-based)
- **5x سریع‌تر** در زمان build
- بهبود عملکرد در پروژه‌های بزرگ

### 2. CSS Layers
- پشتیبانی از CSS Cascade Layers
- کنترل بهتر ترتیب استایل‌ها

### 3. CSS Variables
- استفاده از CSS custom properties
- Theme customization در CSS

### 4. بهبودهای دیگر
- پشتیبانی بهتر از modern CSS features
- بهبود tree-shaking
- کاهش bundle size

## Migration Steps

### Step 1: نصب

```bash
cd frontend
npm install tailwindcss@^4.0.0
```

### Step 2: به‌روزرسانی CSS

فایل `src/index.css` را به‌روزرسانی کنید:

```css
@import "tailwindcss";
```

### Step 3: بررسی Compatibility

- همه utility classes همچنان کار می‌کنند
- Plugins موجود نیاز به بررسی دارند
- Custom configurations باید تست شوند

## Breaking Changes

### 1. Syntax تغییر کرده
- `@tailwind` directives دیگر استفاده نمی‌شوند
- از `@import "tailwindcss"` استفاده کنید

### 2. برخی plugins ممکن است نیاز به به‌روزرسانی داشته باشند
- بررسی کنید که plugins شما با v4 سازگار هستند

### 3. Configuration
- CSS-based configuration جدید است
- `tailwind.config.js` همچنان پشتیبانی می‌شود

## Testing

بعد از migration:

```bash
# Build project
npm run build

# Run dev server
npm run dev

# Check for errors
npm run lint
```

## منابع

- [Tailwind CSS 4.0 Release Notes](https://tailwindcss.com/blog/tailwindcss-v4-beta)
- [Migration Guide](https://tailwindcss.com/docs/upgrade-guide)
- [CSS-based Configuration](https://tailwindcss.com/docs/configuration)

## Notes

- همه utility classes موجود همچنان کار می‌کنند
- نیازی به تغییر در کامپوننت‌ها نیست
- Performance بهبود یافته است

