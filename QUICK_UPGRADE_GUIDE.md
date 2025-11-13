# 🚀 Quick Upgrade Guide

راهنمای سریع به‌روزرسانی به آخرین نسخه‌ها

## ✅ تغییرات اعمال شده

### Tailwind CSS 4.0
- ✅ Package به‌روزرسانی شد
- ✅ CSS syntax تغییر کرد
- ✅ Config به‌روزرسانی شد

### FastAPI 0.115+
- ✅ Requirements به‌روزرسانی شد
- ✅ Version range استفاده می‌شود

### Docker Images
- ✅ همه images به آخرین نسخه‌ها به‌روزرسانی شدند
- ✅ Python base image به 3.12 به‌روزرسانی شد

## 📋 مراحل نصب

### 1. Backend Dependencies

```bash
cd src/backend
pip install -r requirements/backend.txt --upgrade
```

### 2. Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Docker Services

```bash
# Pull latest images
docker-compose pull

# Rebuild if needed
docker-compose build

# Start services
docker-compose up -d
```

## 🧪 تست کردن

### Backend

```bash
# Test API
curl http://localhost:8001/health

# Check FastAPI version
python -c "import fastapi; print(fastapi.__version__)"
```

### Frontend

```bash
# Build test
npm run build

# Dev server
npm run dev
```

### Docker

```bash
# Check running containers
docker-compose ps

# Check logs
docker-compose logs -f
```

## ⚠️ نکات مهم

1. **Tailwind CSS 4**: Syntax تغییر کرده اما همه classes کار می‌کنند
2. **PostgreSQL 16**: اگر داده موجود دارید، ممکن است نیاز به migration باشد
3. **Python 3.12**: مطمئن شوید که همه dependencies سازگار هستند

## 🔄 Rollback (در صورت نیاز)

### Tailwind CSS

```bash
cd frontend
npm install tailwindcss@^3.3.6
# Revert index.css to @tailwind directives
```

### FastAPI

```bash
cd src/backend
pip install fastapi==0.116.1
```

### Docker

```bash
# Revert docker-compose.yml
git checkout docker-compose.yml
docker-compose pull
docker-compose up -d
```

## 📚 مستندات بیشتر

- [Tailwind CSS 4 Migration](TAILWIND_CSS_4_MIGRATION.md)
- [Upgrade Summary](UPGRADE_SUMMARY.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

---

**آماده استفاده!** 🎉

