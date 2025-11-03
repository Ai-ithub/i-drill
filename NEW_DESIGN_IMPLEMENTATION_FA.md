# 🎨 پیاده‌سازی طراحی جدید داشبورد i-Drill

**تاریخ:** 2025-11-03  
**الگو:** TURBIN Generator Dashboard  
**وضعیت:** ✅ **طراحی پایه تکمیل شد**

---

## 📊 خلاصه اجرایی

طراحی جدید داشبورد بر اساس نمونه تصاویر TURBIN Generator ایجاد شد با ویژگی‌های:

✅ **Sidebar در سمت راست** با رنگ سبز  
✅ **Layout مشابه نمونه** با پس‌زمینه مشکی  
✅ **Gauge های دایره‌ای** (Circular Gauges)  
✅ **Gauge های خطی** (Linear Gauges)  
✅ **منوی چند سطحی** با submenu ها  
✅ **صفحه Gauge کامل** با تمام پارامترها  

---

## 🎯 فایل‌های ایجاد شده

### 1️⃣ **Layout جدید** (`NewLayout.tsx`)

**مسیر:** `frontend/src/components/Layout/NewLayout.tsx`

**ویژگی‌ها:**
```typescript
✅ Sidebar در سمت راست
✅ رنگ‌بندی سبز (gradient) مشابه TURBIN
✅ منوی چند سطحی با submenu
✅ 12 آیتم منو:
   - display (با submenu)
   - check list
   - Alarm Systems
   - Control
   - Graph_Analysis
   - 3D_Analysis_OP
   - REAL_TIME_OP
   - Reporting
   - Connection
   - Data Loggers
   - Databases
   - PDM
```

**منوی display:**
- Gauge
- SENSOR
- REAL_TIME_M

**طراحی:**
- پس‌زمینه: Gradient سبز (#8BC34A to #689F38)
- فونت: سفید روی سبز
- Active menu: سبز تیره (#558B2F) با border سفید
- Submenu: سبز کمرنگ‌تر (#7CB342)

---

### 2️⃣ **صفحه Gauge** (`GaugePage.tsx`)

**مسیر:** `frontend/src/pages/Gauge/GaugePage.tsx`

**بخش‌های صفحه:**

#### A. **Frequency Section**
```
- amplitude (Circular Gauge: 0-100)
- frequence (Circular Gauge: 0-1000)
```

#### B. **Pressure Sections** (3 ستون)
```
Absolute Pressure:
- psi-compers (Circular Gauge: 0-1000)
- psi-turbin (Circular Gauge: 0-1000)

Static Pressure:
- psi-compers_s (Circular Gauge: 0-1000)
- psi-turbin_2 (Circular Gauge: 0-1000)

Dynamic Pressure:
- psi-comper.s (Circular Gauge: 0-1000)
- psi-turbin (Circular Gauge: 0-1000)
```

#### C. **Pressure Linear Gauges**
```
- P_C (Linear Gauge: -100 to 100)
- P_T (Linear Gauge: -100 to 100)
```

#### D. **Temperature Section** (7 gauge خطی)
```
- Relative Temp
- Surface Temp
- Internal Temp
- Point Temp
- Fluctuating Temp
- Freezing Point
- Dew Point
```

#### E. **Viscosity Section** (3 gauge خطی)
```
- Temp_vis
- Flash Point
- TBN
```

**قابلیت‌ها:**
✅ داده‌های real-time (شبیه‌سازی شده)
✅ آپدیت هر 1 ثانیه
✅ 3 dropdown برای System, Gauge_parameter, sensor_parameter

---

### 3️⃣ **Circular Gauge Component**

**مسیر:** `frontend/src/components/Gauges/CircularGauge.tsx`

**ویژگی‌ها:**
```typescript
✅ Gauge دایره‌ای با عقربه
✅ محدوده: -135° تا +135° (270° کل)
✅ Background arc سیاه
✅ Value arc سبز (#4CAF50)
✅ Tick marks (9 نشانه)
✅ عقربه سفید
✅ نمایش عددی مقدار
✅ Label و واحد در پایین
```

**Props:**
- `label`: string
- `value`: number
- `min`: number
- `max`: number
- `unit`: string
- `size`: number (default: 200)

**مثال استفاده:**
```tsx
<CircularGauge
  label="amplitude"
  value={850}
  min={0}
  max={1000}
  unit="Hz"
/>
```

---

### 4️⃣ **Linear Gauge Component**

**مسیر:** `frontend/src/components/Gauges/LinearGauge.tsx`

**ویژگی‌ها:**
```typescript
✅ Gauge عمودی (Linear/Bar)
✅ Fill gradient سبز (از پایین)
✅ Background سیاه
✅ Border خاکستری
✅ Tick marks (5 نشانه)
✅ مقادیر در سمت راست
✅ نمایش عددی در وسط
✅ انیمیشن smooth
```

**Props:**
- `label`: string
- `value`: number
- `min`: number
- `max`: number
- `unit`: string
- `height`: number (default: 200)
- `width`: number (default: 60)

**مثال استفاده:**
```tsx
<LinearGauge
  label="Relative Temp"
  value={75}
  min={0}
  max={100}
  unit="°C"
/>
```

---

## 🎨 رنگ‌بندی و طراحی

### Sidebar:
```css
Background: linear-gradient(to bottom, #8BC34A, #689F38)
Header: linear-gradient(to right, #7CB342, #8BC34A)
Menu hover: #689F38
Active menu: #558B2F
Submenu: #7CB342
Border: #689F38
Text: white
```

### Gauges:
```css
Background: #1a1a1a (dark gray)
Border: #666 (gray)
Fill/Value: #4CAF50 (green)
Tick marks: #999 (light gray)
Needle: #fff (white)
Text: #fff (white)
```

### Main Content:
```css
Background: #000 (black)
Text: #fff (white)
Dropdowns: #gray-800
Section titles: #green-400 (italic)
```

---

## 📱 Routes جدید

```typescript
/                      → RealTimeMonitoring (صفحه اصلی)
/dashboard             → Dashboard
/realtime              → RealTimeMonitoring
/historical            → HistoricalData
/predictions           → Predictions
/maintenance           → Maintenance
/display/gauge         → GaugePage ✨ جدید
/display/sensor        → RealTimeMonitoring
```

---

## 🚀 راه‌اندازی

### نصب و اجرا:
```powershell
cd frontend
npm install
npm run dev
```

### باز کردن صفحات:
```
صفحه اصلی:           http://localhost:3000/
صفحه Gauge:          http://localhost:3000/display/gauge
```

---

## 📐 ساختار فایل‌ها

```
frontend/src/
├── components/
│   ├── Layout/
│   │   ├── Layout.tsx              (قدیمی)
│   │   ├── NewLayout.tsx           ✨ جدید - TURBIN style
│   │   ├── Header.tsx
│   │   └── Sidebar.tsx
│   │
│   └── Gauges/
│       ├── CircularGauge.tsx       ✨ جدید
│       ├── LinearGauge.tsx         ✨ جدید
│       └── index.ts                ✨ جدید
│
├── pages/
│   ├── Gauge/
│   │   └── GaugePage.tsx           ✨ جدید
│   │
│   ├── Dashboard/
│   │   └── Dashboard.tsx
│   │
│   ├── RealTimeMonitoring/
│   │   └── RealTimeMonitoring.tsx  (آپدیت شد)
│   │
│   ├── HistoricalData/
│   ├── Predictions/
│   └── Maintenance/
│
└── App.tsx                          (آپدیت شد)
```

---

## ✨ ویژگی‌های کلیدی

### 1. **Sidebar پویا**
```typescript
- منوی چند سطحی
- Expand/Collapse برای submenu ها
- Active state highlighting
- Smooth transitions
- Scroll برای منوهای زیاد
```

### 2. **Circular Gauge**
```typescript
- عقربه متحرک
- Arc رنگی
- Tick marks
- Responsive
- قابل تنظیم (min, max, size)
```

### 3. **Linear Gauge**
```typescript
- Fill animation
- Gradient رنگی
- Tick marks با مقادیر
- نمایش عددی
- قابل تنظیم (height, width)
```

### 4. **Real-time Updates**
```typescript
- داده‌های شبیه‌سازی شده
- آپدیت هر 1 ثانیه
- Smooth transitions
- بدون lag
```

---

## 🎯 تفاوت‌ها با طراحی قبلی

| ویژگی | طراحی قدیم | طراحی جدید TURBIN |
|-------|------------|-------------------|
| Sidebar | چپ | راست ✨ |
| رنگ Sidebar | آبی-خاکستری | سبز gradient ✨ |
| Gauge ها | نمودار خطی | دایره‌ای + خطی ✨ |
| پس‌زمینه | خاکستری تیره | مشکی کامل ✨ |
| منو | ساده | چند سطحی ✨ |
| استایل | مدرن | صنعتی/فنی ✨ |

---

## 📊 نمونه کدها

### استفاده از NewLayout:
```tsx
import NewLayout from '@/components/Layout/NewLayout'

function App() {
  return (
    <NewLayout>
      {/* محتوای صفحه */}
    </NewLayout>
  )
}
```

### استفاده از Gauges:
```tsx
import { CircularGauge, LinearGauge } from '@/components/Gauges'

function MyPage() {
  return (
    <div>
      <CircularGauge
        label="Frequency"
        value={50}
        min={0}
        max={100}
        unit="Hz"
      />
      
      <LinearGauge
        label="Temperature"
        value={75}
        min={0}
        max={100}
        unit="°C"
      />
    </div>
  )
}
```

---

## 🔜 کارهای آینده

### صفحات باقیمانده:
1. ⏳ **SENSOR Page** - نمودار noise signal و histogram
2. ⏳ **Control Page** - دکمه‌های RUN و threshold meters
3. ⏳ **RPM Page** - gauge های RPM, torque, pressure
4. ⏳ **Graph Analysis** - نمودارهای optimization
5. ⏳ **3D Analysis** - تحلیل سه‌بعدی

### بهبودها:
- [ ] اتصال به داده‌های واقعی API
- [ ] ذخیره تنظیمات کاربر
- [ ] Export به PDF/Image
- [ ] Theme customization
- [ ] Responsive برای موبایل

---

## 📚 مستندات مرتبط

- `HIGH_PRIORITY_IMPLEMENTATION_SUMMARY_FA.md` - پیاده‌سازی authentication و MLOps
- `CRITICAL_IMPLEMENTATION_SUMMARY_FA.md` - پیاده‌سازی backend
- `START_HERE_FA.md` - راهنمای شروع سریع

---

## ✅ Checklist تکمیل

### Layout:
- [x] NewLayout component با sidebar راست
- [x] منوی چند سطحی
- [x] رنگ‌بندی TURBIN
- [x] Active state handling
- [x] Responsive sidebar

### Components:
- [x] CircularGauge component
- [x] LinearGauge component
- [x] Real-time data simulation
- [x] Smooth animations

### Pages:
- [x] GaugePage با تمام بخش‌ها
- [x] Integration با Layout
- [x] Routing setup
- [ ] SENSOR page
- [ ] Control page
- [ ] RPM page

---

## 🎉 نتیجه‌گیری

**طراحی جدید TURBIN-style برای i-Drill با موفقیت پیاده‌سازی شد!**

### آماده برای:
✅ نمایش داده‌های real-time  
✅ Gauge های دایره‌ای و خطی  
✅ منوی چند سطحی  
✅ ظاهر صنعتی و حرفه‌ای  
✅ توسعه صفحات بیشتر  

**موفق باشید! 🚀**

---

**نویسنده:** AI Assistant  
**تاریخ:** 2025-11-03  
**ورژن:** 1.0.0  
**الگو:** TURBIN Generator Dashboard

