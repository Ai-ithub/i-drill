# راهنمای تغییر پیام کامیت‌های فارسی به انگلیسی

## کامیت‌های مورد نظر:
- `55c0e0d` - feat: Add RL system management and autonomous mode
- `696ad60` - chore: Improve security configurations and phase zero maintenance infrastructure  
- `8a934b1` - feat: Complete backend-dashboard integration and automated CI

## روش 1: استفاده از git rebase (توصیه می‌شود)

### مرحله 1: پیدا کردن parent commit
```bash
git log --oneline --all | grep -A 5 "696ad60"
```

### مرحله 2: شروع rebase
```bash
git rebase -i 280f21b^
```

### مرحله 3: در editor که باز می‌شود:
- خطوط مربوط به سه کامیت را پیدا کنید
- `pick` را به `reword` تغییر دهید:

```
reword 696ad6000c518b4699f22de2624cea803451675d
reword 55c0e0d298dbb9cb20cb534bcc1014ae0584c71c
reword 8a934b16a8fb02582a97458efe43aab8f67ba866
```

### مرحله 4: برای هر کامیت، پیام جدید را وارد کنید:

**برای 696ad60:**
```
chore: Improve security configurations and phase zero maintenance infrastructure

- Add security headers and middleware improvements
- Enhance health check endpoints
- Update docker-compose configuration
- Improve Kafka service reliability
- Update setup guide documentation
```

**برای 55c0e0d:**
```
feat: Add RL system management and autonomous mode

- Implement RL control page with system management interface
- Add RL service with autonomous mode support
- Create RL API routes for system control
- Add comprehensive RL API tests
- Enhance RL service with advanced control features
```

**برای 8a934b1:**
```
feat: Complete backend-dashboard integration and automated CI

- Enhanced frontend pages (HistoricalData, Maintenance, Predictions)
- Improved backend services (maintenance, prediction, MLflow)
- Added comprehensive API tests for maintenance and predictions
- Updated CI/CD workflows
- WebSocket service improvements
```

## روش 2: استفاده از git commit --amend (فقط برای آخرین کامیت)

اگر کامیت آخرین کامیت است:
```bash
git commit --amend -m "New message"
```

## پس از تغییر پیام‌ها:

### بررسی تغییرات:
```bash
git log --oneline | grep -E "55c0e0d|696ad60|8a934b1"
```

### Push کردن (اگر کامیت‌ها push شده‌اند):
```bash
git push --force origin Main
```

⚠️ **هشدار**: Force push تاریخچه را بازنویسی می‌کند. حتماً با تیم هماهنگ کنید!

