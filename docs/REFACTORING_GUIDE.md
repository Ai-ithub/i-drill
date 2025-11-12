# 🔧 Refactoring Guide

راهنمای refactoring و بهبود ساختار کد

## 📋 اصول Refactoring

### 1. Single Responsibility Principle (SRP)
هر کلاس یا تابع باید یک مسئولیت داشته باشد.

**قبل:**
```python
class DataService:
    def get_data(self):
        # Get data
        pass
    
    def process_data(self):
        # Process data
        pass
    
    def save_to_database(self):
        # Save to database
        pass
```

**بعد:**
```python
class DataService:
    def get_data(self):
        # Get data only
        pass

class DataProcessor:
    def process_data(self):
        # Process data only
        pass

class DatabaseService:
    def save(self):
        # Save only
        pass
```

### 2. DRY (Don't Repeat Yourself)
از تکرار کد خودداری کنید.

**قبل:**
```python
def get_sensor_data_1():
    conn = create_connection()
    # ... code
    conn.close()

def get_sensor_data_2():
    conn = create_connection()
    # ... code
    conn.close()
```

**بعد:**
```python
@contextmanager
def get_db_connection():
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()

def get_sensor_data_1():
    with get_db_connection() as conn:
        # ... code
        pass
```

### 3. Naming Conventions
استفاده از نام‌های واضح و توصیفی.

**قبل:**
```python
def proc(d):
    # ...
    pass
```

**بعد:**
```python
def process_sensor_data(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    # ...
    pass
```

## 🏗️ ساختار پیشنهادی

### Backend Structure
```
src/backend/
├── api/
│   ├── routes/          # API endpoints
│   ├── models/          # Pydantic schemas
│   ├── dependencies.py  # FastAPI dependencies
│   └── exceptions.py    # Custom exceptions
├── services/             # Business logic
├── utils/               # Utility functions
├── database/             # Database models and managers
└── config/              # Configuration
```

### Frontend Structure
```
frontend/src/
├── components/
│   ├── UI/              # Reusable UI components
│   ├── Layout/          # Layout components
│   └── Features/        # Feature-specific components
├── pages/               # Page components
├── hooks/               # Custom React hooks
├── services/            # API services
├── utils/               # Utility functions
├── i18n/                # Internationalization
└── types/               # TypeScript types
```

## 🔄 Refactoring Patterns

### 1. Extract Method
استخراج منطق تکراری به متد جداگانه.

### 2. Extract Class
استخراج منطق مرتبط به کلاس جداگانه.

### 3. Replace Magic Numbers
جایگزینی اعداد ثابت با constants.

**قبل:**
```python
if depth > 10000:
    # ...
```

**بعد:**
```python
MAX_SAFE_DEPTH = 10000
if depth > MAX_SAFE_DEPTH:
    # ...
```

### 4. Introduce Parameter Object
گروه‌بندی پارامترهای مرتبط.

**قبل:**
```python
def create_sensor_data(rig_id, depth, wob, rpm, torque):
    # ...
```

**بعد:**
```python
@dataclass
class SensorDataParams:
    rig_id: str
    depth: float
    wob: float
    rpm: float
    torque: float

def create_sensor_data(params: SensorDataParams):
    # ...
```

## ✅ Checklist Refactoring

- [ ] کد تکراری حذف شده
- [ ] نام‌های واضح استفاده شده
- [ ] توابع کوچک و focused هستند
- [ ] کلاس‌ها Single Responsibility دارند
- [ ] Type hints اضافه شده
- [ ] Docstrings کامل هستند
- [ ] Error handling مناسب است
- [ ] Tests برای کد refactored شده نوشته شده

## 🧪 Testing After Refactoring

بعد از refactoring، حتماً تست‌ها را اجرا کنید:

```bash
pytest tests/ -v
```

## 📚 منابع

- Clean Code by Robert C. Martin
- Refactoring by Martin Fowler
- Python Best Practices

