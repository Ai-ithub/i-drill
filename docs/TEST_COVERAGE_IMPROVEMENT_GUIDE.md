# 📊 راهنمای بهبود Test Coverage

این سند راهنمای بهبود Test Coverage در پروژه i-Drill است.

---

## 📋 خلاصه بهبودها

### تست‌های جدید اضافه شده

| فایل تست | Coverage | توضیحات |
|----------|----------|---------|
| `test_websocket_manager.py` | ~90% | تست‌های کامل برای WebSocket Manager |
| `test_backup_service.py` | ~85% | تست‌های Backup Service |
| `test_security_headers.py` | ~90% | تست‌های Security Headers و CSP |
| `test_integration_service.py` | ~80% | تست‌های Integration Service |

### Coverage Goals

- **Current Target**: 70%+ (افزایش از 60%)
- **Critical Components**: 85%+
- **Services**: 80%+
- **Utilities**: 90%+

---

## 🎯 بخش‌های پوشش داده شده

### ✅ Services

- ✅ `websocket_manager.py` - تست‌های کامل
- ✅ `backup_service.py` - تست‌های کامل
- ✅ `integration_service.py` - تست‌های کامل
- ✅ `auth_service.py` - تست‌های موجود
- ✅ `data_service.py` - تست‌های موجود
- ✅ `control_service.py` - تست‌های موجود
- ✅ `email_service.py` - تست‌های موجود

### ✅ Utilities

- ✅ `security.py` - تست‌های کامل (CSP, Security Headers)
- ✅ `validators.py` - تست‌های موجود
- ✅ `prometheus_metrics.py` - نیاز به تست

### ⚠️ بخش‌های نیازمند تست بیشتر

- ⚠️ `cache_service.py` - نیاز به تست
- ⚠️ `ml_retraining_service.py` - نیاز به تست
- ⚠️ `model_validation_service.py` - نیاز به تست
- ⚠️ `prometheus_metrics.py` - نیاز به تست

---

## 🚀 اجرای تست‌ها

### روش 1: استفاده از Script

```bash
# Linux/Mac
./scripts/run_coverage.sh

# Windows PowerShell
.\scripts\run_coverage.ps1
```

### روش 2: دستی

```bash
cd src/backend
pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
```

### روش 3: با Coverage Report

```bash
pytest tests/ \
    --cov=src/backend \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-branch \
    --cov-fail-under=70
```

---

## 📊 مشاهده Coverage Report

### HTML Report

```bash
# بعد از اجرای تست‌ها
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Terminal Report

Coverage report در terminal نمایش داده می‌شود:

```
Name                          Stmts   Miss  Cover   Missing
------------------------------------------------------------
services/websocket_manager.py     45      5    89%   12-15, 20-22
services/backup_service.py        120     18    85%   45-50, 100-105
utils/security.py                 150     15    90%   200-205
------------------------------------------------------------
TOTAL                           1500    300    80%
```

---

## ✍️ نوشتن تست‌های جدید

### ساختار تست

```python
"""
Unit tests for [Component Name]
"""
import pytest
from unittest.mock import Mock, patch
from services.component_name import ComponentName


class TestComponentName:
    """Tests for ComponentName"""
    
    @pytest.fixture
    def component(self):
        """Create component instance"""
        return ComponentName()
    
    def test_feature_name(self, component):
        """Test description"""
        # Arrange
        input_data = "test"
        
        # Act
        result = component.method(input_data)
        
        # Assert
        assert result == expected_value
```

### استفاده از Fixtures

```python
@pytest.fixture
def mock_database():
    """Mock database"""
    return Mock()

def test_with_mock(self, mock_database):
    """Test with mocked dependency"""
    # Use mock_database
    pass
```

### تست Async Functions

```python
@pytest.mark.asyncio
async def test_async_function(self):
    """Test async function"""
    result = await async_function()
    assert result is not None
```

---

## 🎯 Coverage Targets

### Minimum Coverage

- **Overall**: 70%
- **Services**: 80%
- **Utilities**: 90%
- **API Routes**: 75%

### Critical Components

- **Authentication**: 90%+
- **Security**: 95%+
- **Database**: 85%+
- **WebSocket**: 85%+

---

## 🔍 بررسی Coverage

### شناسایی بخش‌های تست نشده

```bash
# اجرای تست با report
pytest --cov=. --cov-report=term-missing

# بخش‌های Missing را بررسی کنید
```

### Coverage Report Analysis

1. **HTML Report**: برای بررسی دقیق
2. **Terminal Report**: برای quick check
3. **XML Report**: برای CI/CD integration

---

## 📈 بهبود Coverage

### مراحل

1. **شناسایی**: بخش‌های تست نشده را شناسایی کنید
2. **اولویت‌بندی**: بخش‌های critical را اولویت دهید
3. **نوشتن تست**: تست‌های unit و integration بنویسید
4. **اجرا و بررسی**: تست‌ها را اجرا و coverage را بررسی کنید
5. **تکرار**: تا رسیدن به target coverage

### Best Practices

- ✅ تست edge cases
- ✅ تست error handling
- ✅ تست boundary conditions
- ✅ تست async functions
- ✅ استفاده از mocks برای dependencies
- ✅ تست integration بین components

---

## 🐛 عیب‌یابی

### مشکل: Coverage کم است

```bash
# بررسی بخش‌های missing
pytest --cov=. --cov-report=term-missing | grep "Missing"

# اضافه کردن تست‌های بیشتر
```

### مشکل: تست‌ها fail می‌شوند

```bash
# اجرای تست‌ها با verbose
pytest -v

# اجرای یک تست خاص
pytest tests/test_specific.py::TestClass::test_method -v
```

### مشکل: Coverage report تولید نمی‌شود

```bash
# بررسی نصب pytest-cov
pip install pytest-cov

# اجرای مجدد
pytest --cov=. --cov-report=html
```

---

## 📚 منابع بیشتر

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pytest-cov Documentation](https://pytest-cov.readthedocs.io/)

---

**آخرین به‌روزرسانی:** ژانویه 2025  
**نسخه:** 1.0  
**Coverage Target:** 70%+

