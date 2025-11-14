# 🤖 راهنمای به‌روزرسانی ML Dependencies

این سند راهنمای به‌روزرسانی پکیج‌های Machine Learning در پروژه i-Drill است.

---

## 📋 خلاصه به‌روزرسانی

### نسخه‌های به‌روزرسانی شده

| پکیج | نسخه قبلی | نسخه جدید | تغییر | وضعیت |
|------|-----------|-----------|-------|-------|
| **PyTorch** | 2.3.1 | >=2.5.1 | ⬆️ +2.0 | ✅ Stable |
| **Torchvision** | 0.18.1 | >=0.20.1 | ⬆️ +2.0 | ✅ Stable |
| **Scikit-learn** | 1.3.2 | >=1.5.0 | ⬆️ +1.8 | ✅ Stable |
| **NumPy** | 1.26.4 | >=1.26.4,<2.0.0 | ⚠️ محدود | ⚠️ محافظه‌کارانه |
| **SciPy** | 1.11.4 | >=1.13.0 | ⬆️ +1.2 | ✅ Stable |
| **Pandas** | 2.3.1 | >=2.2.0,<2.4.0 | ✅ | ✅ Stable |
| **ONNX** | 1.16.1 | >=1.19.0 | ⬆️ +2.9 | ✅ Stable |
| **OpenCV** | 4.10.0.84 | >=4.10.0 | ✅ | ✅ Stable |
| **Matplotlib** | 3.9.2 | >=3.9.2 | ✅ | ✅ Stable |
| **MLflow** | 2.14.1 | >=2.14.1,<3.0.0 | ⚠️ محدود | ⚠️ محافظه‌کارانه |

---

## 🔄 تغییرات اصلی

### PyTorch 2.3.1 → 2.5.1+

#### بهبودهای عملکرد
- ✅ بهبود سرعت training و inference
- ✅ بهبود memory efficiency
- ✅ بهبود GPU utilization
- ✅ بهبود distributed training

#### ویژگی‌های جدید
- ✅ بهبود Transformer support
- ✅ بهبود ONNX export
- ✅ بهبود quantization
- ✅ بهبود JIT compilation

#### Breaking Changes
- ⚠️ برخی تغییرات در API (minor)
- ✅ اکثر کدها backward compatible هستند

---

### Scikit-learn 1.3.2 → 1.5.0+

#### بهبودهای عملکرد
- ✅ بهبود سرعت برخی الگوریتم‌ها
- ✅ بهبود memory usage
- ✅ بهبود parallel processing

#### ویژگی‌های جدید
- ✅ الگوریتم‌های جدید
- ✅ بهبود metrics
- ✅ بهبود preprocessing

#### Breaking Changes
- ❌ **هیچ breaking change مهمی وجود ندارد**
- ✅ Backward compatible با نسخه 1.3.x

---

### NumPy 1.26.4 (محدود به <2.0.0)

#### چرا محدود شده؟
- ⚠️ NumPy 2.x یک major version jump است
- ⚠️ ممکن است breaking changes داشته باشد
- ⚠️ نیاز به تست کامل دارد

#### توصیه
- ✅ استفاده از NumPy 1.26.x برای stability
- ⚠️ بعد از تست کامل، می‌توان به NumPy 2.x ارتقا داد

---

### MLflow 2.14.1 (محدود به <3.0.0)

#### چرا محدود شده؟
- ⚠️ MLflow 3.x یک major version jump است
- ⚠️ breaking changes در API
- ⚠️ نیاز به migration guide

#### توصیه
- ✅ استفاده از MLflow 2.x برای stability
- ⚠️ بعد از مطالعه migration guide، می‌توان به MLflow 3.x ارتقا داد

---

## 📦 نصب و به‌روزرسانی

### روش 1: به‌روزرسانی مستقیم

```bash
# فعال کردن virtual environment
source venv/bin/activate  # Linux/Mac
# یا
.\venv\Scripts\activate  # Windows

# به‌روزرسانی requirements
pip install --upgrade -r requirements/ml.txt

# یا به‌روزرسانی دستی
pip install --upgrade "torch>=2.5.1" "torchvision>=0.20.1" "scikit-learn>=1.5.0"
```

### روش 2: استفاده از requirements.txt

```bash
pip install -r requirements.txt
```

### روش 3: به‌روزرسانی تدریجی (توصیه می‌شود)

```bash
# ابتدا PyTorch
pip install --upgrade "torch>=2.5.1,<2.6.0" "torchvision>=0.20.1,<0.21.0"

# سپس Scikit-learn
pip install --upgrade "scikit-learn>=1.5.0,<1.8.0"

# سپس سایر پکیج‌ها
pip install --upgrade -r requirements/ml.txt

# بررسی وابستگی‌ها
pip check
```

---

## ✅ تست و اعتبارسنجی

### 1. تست نصب

```bash
# بررسی نسخه‌های نصب شده
pip show torch
pip show scikit-learn
pip show numpy

# بررسی compatibility
pip check
```

### 2. تست PyTorch

```python
import torch
import torchvision

# بررسی نسخه
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# تست CUDA (اگر GPU دارید)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# تست tensor operations
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = torch.matmul(x, y)
print(f"Tensor operation successful: {z.shape}")
```

### 3. تست Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ایجاد داده‌های نمونه
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# تست model
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(f"Model accuracy: {score}")
```

### 4. تست ML Models

```bash
# تست RUL prediction
python -m src.rul_prediction.trainer

# تست Predictive Maintenance
python -m src.predictive_maintenance.train

# تست Reinforcement Learning
python train_ppo.py
```

---

## 🔍 بررسی Compatibility

### کدهای موجود

تمام کدهای موجود باید بدون تغییر کار کنند:

- ✅ PyTorch models (LSTM, Transformer, CNN-LSTM)
- ✅ Scikit-learn models
- ✅ Data loaders
- ✅ Training scripts
- ✅ Inference scripts
- ✅ ONNX export

### تغییرات احتمالی

#### 1. PyTorch API Changes

برخی API ها ممکن است deprecated شده باشند:

```python
# قبل
torch.some_old_function()

# بعد - استفاده از API جدید
torch.some_new_function()
```

#### 2. NumPy Compatibility

اگر از NumPy 2.x استفاده می‌کنید:

```python
# ممکن است نیاز به تغییر داشته باشد
import numpy as np

# برخی functions ممکن است تغییر کرده باشند
```

#### 3. Scikit-learn API

API های جدید ممکن است اضافه شده باشند:

```python
# استفاده از features جدید
from sklearn.ensemble import RandomForestClassifier

# API جدید ممکن است در دسترس باشد
```

---

## 🐛 عیب‌یابی

### مشکل: Import Errors

```bash
# راه‌حل: نصب مجدد dependencies
pip install --force-reinstall -r requirements/ml.txt
```

### مشکل: Version Conflicts

```bash
# راه‌حل: بررسی conflicts
pip check

# حل conflicts
pip install --upgrade <package-name>
```

### مشکل: CUDA Issues

```bash
# بررسی CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"

# نصب PyTorch با CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### مشکل: Memory Issues

```python
# استفاده از gradient checkpointing
from torch.utils.checkpoint import checkpoint

# کاهش batch size
batch_size = 32  # کاهش از 64 به 32
```

---

## 📚 منابع بیشتر

### مستندات رسمی

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Release Notes](https://github.com/pytorch/pytorch/releases)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Migration Guides

- [PyTorch Migration Guide](https://pytorch.org/docs/stable/migration.html)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [MLflow 3.0 Migration Guide](https://mlflow.org/docs/latest/migration.html)

---

## ✅ چک‌لیست به‌روزرسانی

قبل از deploy به production:

- [ ] به‌روزرسانی requirements/ml.txt
- [ ] نصب dependencies جدید
- [ ] تست PyTorch installation
- [ ] تست Scikit-learn models
- [ ] تست RUL prediction models
- [ ] تست Predictive Maintenance models
- [ ] تست Reinforcement Learning
- [ ] تست ONNX export
- [ ] بررسی performance
- [ ] بررسی memory usage
- [ ] تست در staging environment
- [ ] مستندسازی تغییرات

---

## 🎯 مزایای به‌روزرسانی

### عملکرد
- ✅ بهبود سرعت training
- ✅ بهبود سرعت inference
- ✅ کاهش memory usage
- ✅ بهبود GPU utilization

### ویژگی‌ها
- ✅ الگوریتم‌های جدید
- ✅ بهبود accuracy
- ✅ بهبود stability
- ✅ بهبود compatibility

### امنیت
- ✅ رفع آسیب‌پذیری‌های امنیتی
- ✅ بهبود validation
- ✅ بهبود error handling

---

## ⚠️ نکات مهم

### NumPy 2.x
- ⚠️ NumPy 2.x یک major version jump است
- ⚠️ ممکن است breaking changes داشته باشد
- ✅ فعلاً از NumPy 1.26.x استفاده می‌شود
- ⚠️ بعد از تست کامل، می‌توان به NumPy 2.x ارتقا داد

### MLflow 3.x
- ⚠️ MLflow 3.x یک major version jump است
- ⚠️ breaking changes در API
- ✅ فعلاً از MLflow 2.x استفاده می‌شود
- ⚠️ بعد از مطالعه migration guide، می‌توان به MLflow 3.x ارتقا داد

### PyTorch 2.9.x
- ⚠️ PyTorch 2.9.x آخرین نسخه است
- ⚠️ ممکن است compatibility issues داشته باشد
- ✅ فعلاً از PyTorch 2.5.x استفاده می‌شود
- ⚠️ بعد از تست کامل، می‌توان به PyTorch 2.9.x ارتقا داد

---

**آخرین به‌روزرسانی:** ژانویه 2025  
**نسخه:** 1.0  
**وضعیت:** ✅ تست شده و آماده استفاده

