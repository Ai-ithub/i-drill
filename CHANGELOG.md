# 📝 Changelog

تمام تغییرات مهم این پروژه در این فایل مستند شده‌اند.

فرمت بر اساس [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) است و این پروژه از [Semantic Versioning](https://semver.org/spec/v2.0.0.html) پیروی می‌کند.

---

## [Unreleased]

### Added
- تست‌های جدید برای WebSocket Manager
- تست‌های جدید برای Backup Service
- تست‌های جدید برای Security Headers
- تست‌های جدید برای Integration Service
- تست‌های جدید برای Prometheus Metrics
- تست‌های جدید برای Cache Service
- مستندات بهبود یافته
- API Reference Guide
- Documentation Index

### Changed
- بهبود Test Coverage از 60% به 70%+
- به‌روزرسانی FastAPI از 0.115.0 به 0.121.2
- به‌روزرسانی Uvicorn از 0.32.1 به 0.38.0
- به‌روزرسانی Pydantic از 2.11.7 به 2.12.4
- به‌روزرسانی PyTorch از 2.3.1 به 2.5.1+
- به‌روزرسانی Scikit-learn از 1.3.2 به 1.5.0+
- بهبود Security Headers و CSP
- بهبود pytest.ini configuration

### Security
- حذف رمز عبور پیش‌فرض از dvr.py
- اضافه کردن validation برای rig_id در WebSocket
- بهبود Security Headers (CSP, HSTS, Permissions Policy)
- بهبود CORS configuration

---

## [1.0.0] - 2025-01-XX

### Added
- سیستم کامل Real-time Monitoring
- Dashboard با React + TypeScript
- WebSocket integration برای live data
- JWT Authentication با RBAC
- Predictive Maintenance با ML models
- Reinforcement Learning برای optimization
- Data Validation & Reconciliation (DVR)
- MLOps pipeline با MLflow
- Docker Compose setup
- CI/CD با GitHub Actions
- Comprehensive test suite
- Documentation کامل

### Features
- Real-time sensor data visualization
- Historical data analysis
- RUL prediction (LSTM, Transformer, CNN-LSTM)
- Anomaly detection
- Maintenance scheduling
- Automated parameter optimization
- Multi-rig support
- Role-based access control
- Internationalization (i18n)
- Dark mode support
- Responsive design

---

## [0.9.0] - 2024-12-XX

### Added
- Initial release
- Basic dashboard
- Sensor data API
- ML model integration

---

## Breaking Changes

### [1.0.0]
- هیچ breaking change مهمی وجود ندارد

---

## Migration Guides

### FastAPI Upgrade
برای به‌روزرسانی FastAPI، [FASTAPI_UPGRADE_GUIDE.md](docs/FASTAPI_UPGRADE_GUIDE.md) را ببینید.

### ML Dependencies Upgrade
برای به‌روزرسانی ML Dependencies، [ML_DEPENDENCIES_UPGRADE_GUIDE.md](docs/ML_DEPENDENCIES_UPGRADE_GUIDE.md) را ببینید.

---

## Deprecated

هیچ feature ای در حال حاضر deprecated نشده است.

---

## Removed

هیچ feature ای حذف نشده است.

---

## Security

### [1.0.0]
- بهبود Security Headers
- بهبود CSP policies
- بهبود CORS configuration
- حذف hardcoded passwords
- بهبود token storage

---

**نکته:** برای جزئیات بیشتر تغییرات، commit history را ببینید.

