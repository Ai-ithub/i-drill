# 🔧 Troubleshooting Guide

راهنمای عیب‌یابی مشکلات رایج در i-Drill

## 📋 فهرست مطالب

1. [مشکلات نصب و راه‌اندازی](#مشکلات-نصب-و-راه-اندازی)
2. [مشکلات Backend](#مشکلات-backend)
3. [مشکلات Frontend](#مشکلات-frontend)
4. [مشکلات Database](#مشکلات-database)
5. [مشکلات Kafka](#مشکلات-kafka)
6. [مشکلات MLflow](#مشکلات-mlflow)
7. [مشکلات Docker](#مشکلات-docker)
8. [مشکلات Performance](#مشکلات-performance)
9. [مشکلات Authentication](#مشکلات-authentication)

## 🚀 مشکلات نصب و راه‌اندازی

### مشکل: Python version mismatch

**خطا:**
```
Python version 3.12 is required but older version is installed
```

**راه‌حل:**
```bash
# بررسی نسخه Python
python --version

# نصب Python 3.12+
# Windows: از python.org دانلود کنید
# Linux: sudo apt install python3.12
# macOS: brew install python@3.12
```

### مشکل: npm install fails

**خطا:**
```
npm ERR! code ELIFECYCLE
npm ERR! errno 1
```

**راه‌حل:**
```bash
# پاک کردن cache
npm cache clean --force

# حذف node_modules و package-lock.json
rm -rf node_modules package-lock.json

# نصب مجدد
npm install
```

### مشکل: Docker Compose fails

**خطا:**
```
ERROR: Couldn't connect to Docker daemon
```

**راه‌حل:**
```bash
# بررسی وضعیت Docker
docker ps

# راه‌اندازی Docker service
# Linux: sudo systemctl start docker
# Windows/Mac: Start Docker Desktop

# بررسی Docker Compose
docker-compose --version
```

## 🔧 مشکلات Backend

### مشکل: Module not found

**خطا:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**راه‌حل:**
```bash
# فعال کردن virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# نصب dependencies
pip install -r requirements/backend.txt

# بررسی نصب
pip list | grep fastapi
```

### مشکل: Port already in use

**خطا:**
```
ERROR: [Errno 48] Address already in use
```

**راه‌حل:**
```bash
# پیدا کردن process
# Linux/Mac:
lsof -i :8001
kill -9 <PID>

# Windows:
netstat -ano | findstr :8001
taskkill /PID <PID> /F

# یا استفاده از port دیگر
uvicorn app:app --reload --port 8002
```

### مشکل: Database connection error

**خطا:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**راه‌حل:**
```bash
# بررسی وضعیت PostgreSQL
# Docker:
docker ps | grep postgres

# Local:
sudo systemctl status postgresql

# بررسی connection string در .env
DATABASE_URL=postgresql://user:password@localhost:5432/idrill

# تست connection
psql -h localhost -U user -d idrill
```

### مشکل: Import errors

**خطا:**
```
ImportError: cannot import name 'X' from 'Y'
```

**راه‌حل:**
```bash
# بررسی PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/backend"

# یا اجرا از root directory
cd src/backend
python -m uvicorn app:app --reload
```

## 🎨 مشکلات Frontend

### مشکل: Build fails

**خطا:**
```
Error: Cannot find module 'X'
```

**راه‌حل:**
```bash
# نصب dependencies
npm install

# بررسی package.json
cat package.json

# حذف و نصب مجدد
rm -rf node_modules package-lock.json
npm install
```

### مشکل: Hot reload not working

**راه‌حل:**
```bash
# بررسی Vite config
# frontend/vite.config.ts

# راه‌اندازی مجدد dev server
npm run dev

# پاک کردن cache
rm -rf node_modules/.vite
```

### مشکل: TypeScript errors

**خطا:**
```
TS2307: Cannot find module '@/components/UI'
```

**راه‌حل:**
```bash
# بررسی tsconfig.json
# اطمینان از وجود paths:
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}

# Restart TypeScript server در VS Code
# Cmd/Ctrl + Shift + P -> "TypeScript: Restart TS Server"
```

### مشکل: CORS errors

**خطا:**
```
Access to fetch at 'http://localhost:8001' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**راه‌حل:**
```python
# src/backend/app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🗄️ مشکلات Database

### مشکل: Migration fails

**خطا:**
```
alembic.util.exc.CommandError: Target database is not up to date
```

**راه‌حل:**
```bash
# اجرای migrations
alembic upgrade head

# یا از ابتدا
alembic downgrade base
alembic upgrade head
```

### مشکل: Table already exists

**خطا:**
```
sqlalchemy.exc.ProgrammingError: relation "X" already exists
```

**راه‌حل:**
```bash
# حذف table
psql -d idrill -c "DROP TABLE IF EXISTS X;"

# یا استفاده از Alembic
alembic downgrade -1
alembic upgrade head
```

### مشکل: Slow queries

**راه‌حل:**
```sql
-- ایجاد index
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_sensor_data_rig_id ON sensor_data(rig_id);

-- بررسی query plan
EXPLAIN ANALYZE SELECT * FROM sensor_data WHERE rig_id = 'RIG_01';
```

## 📨 مشکلات Kafka

### مشکل: Kafka connection error

**خطا:**
```
kafka.errors.KafkaError: Unable to bootstrap brokers
```

**راه‌حل:**
```bash
# بررسی وضعیت Kafka
docker ps | grep kafka

# بررسی logs
docker logs kafka

# راه‌اندازی مجدد
docker-compose restart kafka

# بررسی KAFKA_BOOTSTRAP_SERVERS در .env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### مشکل: Consumer lag

**راه‌حل:**
```bash
# بررسی consumer groups
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list

# بررسی lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group --describe
```

## 🤖 مشکلات MLflow

### مشکل: MLflow not accessible

**خطا:**
```
ConnectionError: Could not connect to MLflow server
```

**راه‌حل:**
```bash
# بررسی MLflow server
mlflow ui --port 5000

# بررسی MLFLOW_TRACKING_URI در .env
MLFLOW_TRACKING_URI=http://localhost:5000
```

### مشکل: Model not found

**خطا:**
```
mlflow.exceptions.MlflowException: Model version not found
```

**راه‌حل:**
```python
# بررسی models در registry
from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()
print(models)

# بررسی model versions
versions = client.get_latest_versions("model_name")
print(versions)
```

## 🐳 مشکلات Docker

### مشکل: Container won't start

**خطا:**
```
Error response from daemon: driver failed programming external connectivity
```

**راه‌حل:**
```bash
# بررسی ports
docker ps -a

# پاک کردن containers
docker-compose down

# راه‌اندازی مجدد
docker-compose up -d
```

### مشکل: Out of memory

**خطا:**
```
ERROR: failed to start container: OOMKilled
```

**راه‌حل:**
```yaml
# docker-compose.yml
services:
  service:
    mem_limit: 2g
    memswap_limit: 2g
```

### مشکل: Volume permissions

**خطا:**
```
Permission denied: /var/lib/postgresql/data
```

**راه‌حل:**
```bash
# تغییر ownership
sudo chown -R 999:999 ./postgres-data

# یا در docker-compose.yml
volumes:
  postgres-data:
    driver: local
```

## ⚡ مشکلات Performance

### مشکل: Slow API responses

**راه‌حل:**
```python
# اضافه کردن caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# استفاده از async
@router.get("/")
async def endpoint():
    # async operations
    pass

# اضافه کردن indexes در database
```

### مشکل: Frontend slow loading

**راه‌حل:**
```typescript
// Code splitting
const LazyComponent = React.lazy(() => import('./Component'));

// Memoization
const MemoizedComponent = React.memo(Component);

// Virtual scrolling برای lists
```

### مشکل: Memory leaks

**راه‌حل:**
```typescript
// Cleanup در useEffect
useEffect(() => {
  const subscription = subscribe();
  return () => subscription.unsubscribe();
}, []);

// Cleanup در Python
try:
    # operations
finally:
    # cleanup
    pass
```

## 🔐 مشکلات Authentication

### مشکل: Token expired

**خطا:**
```
401 Unauthorized: Token expired
```

**راه‌حل:**
```typescript
// Refresh token
const refreshToken = async () => {
  const response = await fetch('/api/v1/auth/refresh', {
    method: 'POST',
    body: JSON.stringify({ refresh_token }),
  });
  return response.json();
};
```

### مشکل: Invalid credentials

**راه‌حل:**
```bash
# Reset password
python scripts/reset_password.py username new_password

# یا از طریق API
POST /api/v1/auth/password/reset/request
```

## 📞 دریافت کمک

اگر مشکل شما حل نشد:

1. **بررسی Logs:**
   ```bash
   # Backend
   tail -f logs/app.log
   
   # Frontend
   npm run dev -- --debug
   
   # Docker
   docker-compose logs -f
   ```

2. **بررسی Issues:**
   - GitHub Issues: https://github.com/Ai-ithub/i-drill/issues

3. **مستندات:**
   - [API Documentation](src/backend/API_DOCUMENTATION.md)
   - [Architecture Guide](docs/ARCHITECTURE.md)
   - [Developer Guide](docs/DEVELOPER_GUIDE.md)

4. **Community:**
   - Discord: [لینک Discord]
   - Email: support@idrill.example.com

## 📝 گزارش مشکل

هنگام گزارش مشکل، لطفاً شامل موارد زیر باشید:

1. **شرح مشکل:** چه اتفاقی افتاد؟
2. **خطاها:** پیام‌های خطا
3. **مراحل بازتولید:** چگونه می‌توان مشکل را بازتولید کرد؟
4. **Environment:**
   - OS و نسخه
   - Python/Node.js version
   - Docker version (اگر استفاده می‌کنید)
5. **Logs:** لاگ‌های مربوطه

