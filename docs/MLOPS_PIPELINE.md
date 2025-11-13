# MLOps Pipeline Documentation

## Overview

این مستندات پیکربندی کامل pipeline MLOps برای پروژه i-drill را شرح می‌دهد. این pipeline شامل آموزش خودکار مدل‌ها، اعتبارسنجی، استقرار و نظارت است.

## Components

### 1. Automated Model Training (`mlops-train.yml`)

این workflow آموزش خودکار مدل‌ها را مدیریت می‌کند.

**Triggers:**
- Manual dispatch (با انتخاب نوع مدل)
- Scheduled (هفتگی، یکشنبه‌ها ساعت 2 صبح UTC)
- Push به branches اصلی یا تغییر در فایل‌های آموزش

**Features:**
- پشتیبانی از انواع مدل‌ها: PPO, SAC, LSTM, Transformer, CNN-LSTM
- یکپارچه‌سازی با MLflow برای tracking
- ارزیابی خودکار مدل‌های آموزش‌دیده
- ثبت خودکار مدل‌ها در MLflow Model Registry

**Usage:**
```bash
# Trigger manually via GitHub Actions UI
# یا از command line:
gh workflow run mlops-train.yml \
  -f model_type=ppo \
  -f experiment_name=i-drill-training
```

### 2. Model Validation (`mlops-validate.yml`)

این workflow مدل‌ها را قبل از استقرار اعتبارسنجی می‌کند.

**Triggers:**
- Manual dispatch (با نام مدل)
- پس از تکمیل workflow آموزش

**Validation Checks:**
- Data quality validation
- Model performance metrics
- Drift detection
- Prediction consistency tests
- Model bias checks

**Promotion:**
- پس از اعتبارسنجی موفق، مدل به stage "Staging" ارتقا می‌یابد

### 3. Model Deployment (`mlops-deploy.yml`)

این workflow استقرار مدل‌ها را مدیریت می‌کند.

**Deployment Strategies:**
- **Canary**: استقرار تدریجی با درصد ترافیک مشخص
- **Blue-Green**: استقرار بدون downtime
- **Rolling**: استقرار incremental

**Environments:**
- Staging: برای تست قبل از production
- Production: استقرار نهایی

**Features:**
- Build Docker images برای model serving
- Health checks و smoke tests
- Automatic rollback در صورت failure
- Monitoring post-deployment

### 4. Automated Retraining (`mlops-retrain.yml`)

این workflow شرایط لازم برای ریترینینگ را بررسی می‌کند.

**Retraining Conditions:**
- **Time-based**: ریترینینگ دوره‌ای (به‌صورت پیش‌فرض هر 30 روز)
- **Performance-based**: ریترینینگ در صورت کاهش عملکرد
- **Drift-based**: ریترینینگ در صورت تشخیص drift در داده‌ها
- **Data availability**: ریترینینگ در صورت دسترسی به داده‌های جدید

**Schedule:**
- بررسی روزانه ساعت 2 صبح UTC

## Configuration

### MLOps Config (`config/mlops_config.yaml`)

فایل پیکربندی اصلی که شامل تنظیمات برای:
- Training parameters
- Validation thresholds
- Deployment strategies
- Monitoring settings
- Retraining conditions

## Scripts

### Training Scripts

#### `scripts/log_training_to_mlflow.py`
ثبت نتایج آموزش در MLflow

```bash
python scripts/log_training_to_mlflow.py \
  --model_type ppo \
  --model_name ppo_drilling_env \
  --model_path ./models/ppo \
  --experiment i-drill-training
```

#### `scripts/evaluate_model.py`
ارزیابی مدل‌های آموزش‌دیده

```bash
python scripts/evaluate_model.py \
  --model_dir ./models \
  --experiment i-drill-training \
  --num_episodes 10
```

### Validation Scripts

#### `scripts/check_model_quality.py`
بررسی کیفیت مدل قبل از استقرار

```bash
python scripts/check_model_quality.py \
  --metrics_file ./evaluation_results/metrics.json \
  --thresholds_file ./config/mlops_config.yaml
```

### Monitoring Scripts

#### `scripts/monitor_model_performance.py`
نظارت بر عملکرد مدل در production

```bash
# One-time check
python scripts/monitor_model_performance.py \
  --model_name ppo_drilling_env \
  --config ./config/mlops_config.yaml

# Continuous monitoring
python scripts/monitor_model_performance.py \
  --model_name ppo_drilling_env \
  --config ./config/mlops_config.yaml \
  --continuous \
  --interval 3600
```

### Retraining Scripts

#### `scripts/trigger_retraining.py`
بررسی و trigger کردن ریترینینگ

```bash
# Check only
python scripts/trigger_retraining.py \
  --model_name ppo_drilling_env \
  --config ./config/mlops_config.yaml \
  --check-only

# Auto-trigger if conditions met
python scripts/trigger_retraining.py \
  --model_name ppo_drilling_env \
  --config ./config/mlops_config.yaml \
  --auto-trigger
```

## Model Validation Service

سرویس `ModelValidationService` در `src/backend/services/model_validation_service.py` ارائه می‌دهد:

- Validation برای regression models
- Validation برای classification models
- Data drift detection
- Prediction consistency checks
- Model metrics validation
- Generation of validation reports

## MLflow Integration

تمام workflow‌ها با MLflow یکپارچه شده‌اند:

- **Tracking**: تمام metrics و parameters ثبت می‌شوند
- **Model Registry**: مدل‌ها در registry ثبت می‌شوند
- **Stages**: مدل‌ها بین stages (None -> Staging -> Production -> Archived) منتقل می‌شوند
- **Experiments**: سازمان‌دهی runs در experiments

## Monitoring

### Prometheus Metrics

مدل‌ها metrics زیر را export می‌کنند:
- `model_predictions_total`: تعداد کل predictions
- `model_latency_ms`: زمان پاسخ model
- `model_errors_total`: تعداد خطاها
- `model_prediction_score`: score predictions

### Grafana Dashboards

Dashboard‌های آماده برای نمایش:
- Model performance trends
- Latency distribution
- Error rates
- Prediction distributions

## Environment Variables

### Required

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=i-drill-models
```

### Optional

```bash
TOTAL_TIMESTEPS=100000
LEARNING_RATE=0.0003
GAMMA=0.99
```

## Workflow Dependencies

```
mlops-train.yml
  ↓
mlops-validate.yml
  ↓
mlops-deploy.yml
  ↓
monitoring (continuous)
  ↓
mlops-retrain.yml (when conditions met)
```

## Best Practices

1. **Always validate before deployment**: استفاده از `mlops-validate.yml` قبل از استقرار
2. **Monitor continuously**: نظارت مداوم بر عملکرد در production
3. **Set appropriate thresholds**: تنظیم thresholds واقع‌بینانه در `mlops_config.yaml`
4. **Version control models**: استفاده از MLflow Model Registry برای versioning
5. **Test in staging first**: همیشه ابتدا در staging تست کنید
6. **Automate retraining**: استفاده از automated retraining برای حفظ کیفیت

## Troubleshooting

### Training Fails

- بررسی logs در GitHub Actions
- بررسی دسترسی به MLflow
- بررسی resources (GPU, memory)

### Validation Fails

- بررسی thresholds در config
- بررسی کیفیت داده‌های validation
- بررسی metrics model

### Deployment Fails

- بررسی health checks
- بررسی دسترسی به container registry
- بررسی network connectivity

## Future Improvements

- [ ] Integration با Feature Store (Feast)
- [ ] A/B Testing framework
- [ ] Model explainability checks
- [ ] Automated hyperparameter tuning
- [ ] Distributed training support
- [ ] Multi-cloud deployment

## Support

برای سوالات و issues، لطفاً:
1. بررسی مستندات
2. بررسی GitHub Issues
3. ایجاد issue جدید در صورت نیاز

