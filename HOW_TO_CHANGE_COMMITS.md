# راهنمای تغییر پیام کامیت‌های فارسی به انگلیسی

## کامیت‌های مورد نظر:
1. `696ad60` - chore: Improve security configurations...
2. `55c0e0d` - feat: Add RL system management...
3. `8a934b1` - feat: Complete backend-dashboard integration...

## روش انجام:

### مرحله 1: شروع rebase
```bash
git rebase -i 280f21b^
```

### مرحله 2: در editor که باز می‌شود
محتوای فایل `rebase_todo.txt` را کپی و paste کنید:

```
reword 696ad6000c518b4699f22de2624cea803451675d
reword 55c0e0d298dbb9cb20cb534bcc1014ae0584c71c
reword 8a934b16a8fb02582a97458efe43aab8f67ba866
```

سپس فایل را ذخیره و ببندید.

### مرحله 3: برای هر کامیت
Git برای هر کامیت که `reword` شده، editor را باز می‌کند. در هر بار:

**برای کامیت 696ad60:**
محتوای فایل `msg_696ad60.txt` را کپی و paste کنید.

**برای کامیت 55c0e0d:**
محتوای فایل `msg_55c0e0d.txt` را کپی و paste کنید.

**برای کامیت 8a934b1:**
محتوای فایل `msg_8a934b1.txt` را کپی و paste کنید.

### مرحله 4: بررسی
```bash
git log --oneline | grep -E "55c0e0d|696ad60|8a934b1"
```

### مرحله 5: Push (اگر کامیت‌ها push شده‌اند)
```bash
git push --force origin Main
```

⚠️ **هشدار**: Force push تاریخچه را بازنویسی می‌کند. حتماً با تیم هماهنگ کنید!

## فایل‌های آماده شده:
- `rebase_todo.txt` - دستورات rebase
- `msg_696ad60.txt` - پیام جدید برای کامیت اول
- `msg_55c0e0d.txt` - پیام جدید برای کامیت دوم
- `msg_8a934b1.txt` - پیام جدید برای کامیت سوم

