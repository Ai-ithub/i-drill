# 🔍 تحلیل مشکلات اتصال داشبورد

## مشکلات شناسایی شده:

### 1. ⚠️ سرورها در حال اجرا نیستند
- **Backend (پورت 8001)**: در حال اجرا نیست
- **Frontend (پورت 3001)**: در حال اجرا نیست

**راه حل**: اجرای `start_dashboard.ps1` یا راه‌اندازی دستی سرورها

---

### 2. 🔗 WebSocket URLs به صورت Hardcoded هستند

**مکان‌های مشکل‌دار**:

#### `frontend/src/pages/RealTimeMonitoring/RealTimeMonitoring.tsx`
```typescript
const { data: wsData, isConnected } = useWebSocket(
  `ws://localhost:8001/api/v1/sensor-data/ws/${rigId}`
)
```

#### `frontend/src/pages/Data/tabs/RealTimeDataTab.tsx`
```typescript
const wsUrl = `ws://localhost:8001/api/v1/sensor-data/ws/${rigId}`
```

**مشکل**: 
- این URLها به صورت سخت‌کد شده‌اند و از متغیرهای محیطی استفاده نمی‌کنند
- در محیط production یا زمانی که پورت‌ها تغییر می‌کنند، کار نمی‌کنند

**راه حل پیشنهادی**: استفاده از متغیر محیطی `VITE_WS_URL`

---

### 3. ✅ تنظیمات CORS صحیح است
- Backend شامل `http://localhost:3001` در `DEFAULT_ALLOWED_ORIGINS` است
- CORS برای development به درستی تنظیم شده است

---

### 4. ✅ API Base URL صحیح است
- Frontend از `http://localhost:8001/api/v1` استفاده می‌کند
- Backend تمام routeها را با prefix `/api/v1` ارائه می‌دهد

---

### 5. ⚠️ Vite Proxy Configuration
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8001',
    changeOrigin: true,
  },
}
```

**نکته**: Proxy برای `/api` تنظیم شده، اما frontend مستقیماً از `http://localhost:8001/api/v1` استفاده می‌کند.

---

## 📋 راه‌حل‌های پیشنهادی:

### راه حل 1: اجرای دستی سرورها

#### Backend:
```powershell
cd src\backend
python start_server.py
```

#### Frontend:
```powershell
cd frontend
npm run dev
```

---

### راه حل 2: استفاده از اسکریپت PowerShell

```powershell
.\start_dashboard.ps1
```

---

### راه حل 3: رفع Hardcoded URLs (پیشنهادی برای بهبود)

1. افزودن `VITE_WS_URL` به `.env` فایل frontend
2. استفاده از متغیر محیطی در کدها

---

## 🔧 مراحل رفع مشکلات:

1. ✅ بررسی CORS - بدون مشکل
2. ✅ بررسی API Routes - بدون مشکل  
3. ✅ رفع Hardcoded WebSocket URLs - **انجام شد**
   - استفاده از `VITE_WS_URL` در RealTimeMonitoring.tsx
   - استفاده از `VITE_WS_URL` در RealTimeDataTab.tsx
   - بهبود استفاده از `VITE_API_URL` در AuthContext.tsx
4. ⚠️ راه‌اندازی سرورها - نیاز به اجرای دستی
5. ⚠️ تست اتصال - بعد از راه‌اندازی سرورها

---

## 📝 نکات مهم:

- Backend روی پورت **8001** اجرا می‌شود
- Frontend روی پورت **3001** اجرا می‌شود
- تمام API endpoints با prefix `/api/v1` هستند
- WebSocket endpoint: `ws://localhost:8001/api/v1/sensor-data/ws/{rigId}`

