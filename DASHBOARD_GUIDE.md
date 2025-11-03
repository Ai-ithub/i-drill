# 🎨 راهنمای داشبورد i-Drill

## 🚀 دسترسی به داشبورد

داشبورد i-Drill در حال حاضر **در حال اجرا** است!

### 📍 آدرس‌های دسترسی

- **🌐 داشبورد اصلی**: [http://localhost:3000](http://localhost:3000)
- **⚡ Backend API**: [http://localhost:8001](http://localhost:8001)
- **📚 مستندات API**: [http://localhost:8001/docs](http://localhost:8001/docs)

---

## 🎯 صفحات داشبورد

### 📊 منوی Display (در Sidebar راست)

برای دسترسی به صفحات مختلف، روی **"display"** در sidebar سبز سمت راست کلیک کنید:

#### 1️⃣ **صفحه Gauge** 
```
http://localhost:3000/display/gauge
```
**ویژگی‌ها:**
- 🎯 Circular Gauges برای WOB, RPM, Torque
- 📏 Linear Gauges برای Pressure و Flow Rate
- 🎨 طراحی TURBIN با رنگ‌های آبی و سبز
- 📊 نمایش real-time مقادیر

**ظاهر:**
- Background تیره (Gray-900)
- Gauges رنگی با animation
- مقادیر به صورت font-mono
- Grid layout responsive

---

#### 2️⃣ **صفحه RPM**
```
http://localhost:3000/display/rpm
```
**ویژگی‌ها:**
- 🔵 RPM Gauge (0-200 rpm) - آبی
- 🟢 Torque Gauge (0-40 kft-lbs) - سبز
- 🟡 Pressure Gauge (0-5000 psi) - زرد
- 🟣 WOB Linear Gauge - بنفش
- 🔴 ROP Linear Gauge - صورتی
- 🌡️ Temperature Display با gradient
- 📊 Performance Indicators
- 🚨 Alarms & Warnings Panel

**کنترل‌ها:**
- ▶️ دکمه "LIVE DATA" برای شروع به‌روزرسانی
- ⏸ دکمه "PAUSE" برای توقف

**ظاهر:**
- 3 gauge دایره‌ای بزرگ در ردیف اول
- 2 gauge خطی در ردیف دوم
- Temperature bar با gradient قرمز-زرد
- Performance indicators با progress bars

---

#### 3️⃣ **صفحه SENSOR**
```
http://localhost:3000/display/sensor
```
**ویژگی‌ها:**
- 📈 Noise Signal Chart - نمایش سیگنال و نویز
- 📊 Distribution Histogram - توزیع دامنه
- 📉 FFT Spectrum - آنالیز فرکانس
- 📊 Statistics Panel:
  - Mean Value
  - Std Deviation
  - Peak-to-Peak
  - RMS Value
  - SNR (Signal-to-Noise Ratio)
  - Noise Floor

**کنترل‌ها:**
- ▶️ دکمه "START" برای شروع ضبط
- ⏸ دکمه "PAUSE" برای توقف
- نمایش Sample Rate: 1000 Hz
- نمایش Buffer: 100 pts

**ظاهر:**
- 2 chart بزرگ (Line و Bar)
- Statistics grid با 6 metric
- FFT spectrum در پایین
- رنگ‌ها: آبی (سیگنال)، قرمز (نویز)، سبز (FFT)

---

#### 4️⃣ **صفحه Control**
```
http://localhost:3000/display/control
```
**ویژگی‌ها:**

**بخش RUN CONTROLS:**
- ▶️ START - شروع عملیات
- ⏸ PAUSE - توقف موقت
- ⏹ STOP - توقف کامل
- 🚨 EMERGENCY STOP - توقف اضطراری
- نمایش Status (RUNNING/STOPPED)
- Runtime Info (Runtime, Depth, ROP)

**بخش THRESHOLD SETTINGS:**
- تنظیمات آستانه برای 6 پارامتر:
  1. WOB (0-50 klbs)
  2. RPM (0-200 rpm)
  3. Torque (0-40 kft-lbs)
  4. Pressure (0-5000 psi)
  5. Flow Rate (0-1000 gpm)
  6. Temperature (0-300 °F) - CRITICAL

**ویژگی‌های Threshold:**
- Progress bar برای هر پارامتر
- نمایش مقدار فعلی
- Input برای Min و Max
- Warning indicator (زرد) برای مقادیر خارج از محدوده
- Critical badge (قرمز) برای پارامترهای بحرانی

**ظاهر:**
- Layout 1:2 (کنترل‌ها در چپ، thresholds در راست)
- دکمه‌های بزرگ با icon
- Progress bars رنگی
- Input fields برای تنظیم آستانه

---

## 🎨 طراحی کلی داشبورد

### رنگ‌بندی (TURBIN Style):
- **Background**: Gray-900 (#111827) - تیره
- **Cards**: Gray-800 (#1F2937)
- **Primary**: Cyan-400/500 (#06B6D4) - آبی روشن
- **Success**: Green-400/500 (#10B981)
- **Warning**: Yellow-400/500 (#F59E0B)
- **Danger**: Red-600/700 (#DC2626)
- **Borders**: Cyan-500/30 (شفاف)

### Sidebar راست (سبز):
- رنگ: Gradient از #8BC34A به #689F38
- Logo: "TURBIN Generator"
- منوهای تو در تو
- Active state: Border سفید در سمت راست
- Hover effect: تیره‌تر شدن

### Typography:
- **Headings**: Font-bold, text-cyan-400
- **Values/Numbers**: Font-mono (Monospace)
- **Labels**: text-gray-400
- **Units**: text-sm text-gray-500

---

## 🔄 نحوه استفاده

### 1. ورود به داشبورد:
1. مرورگر خود را باز کنید
2. به آدرس http://localhost:3000 بروید
3. داشبورد به صورت خودکار لود می‌شود

### 2. ناوبری:
- از **Sidebar راست** برای حرکت بین صفحات استفاده کنید
- روی **"display"** کلیک کنید تا submenu باز شود
- هر صفحه را انتخاب کنید

### 3. تعامل با Gauges:
- دکمه‌های **START/LIVE DATA** برای فعال‌سازی real-time
- مشاهده تغییرات لحظه‌ای در gauges
- دکمه **PAUSE** برای توقف

### 4. تنظیم Thresholds (در صفحه Control):
- مقادیر Min و Max را وارد کنید
- دکمه **"Apply Changes"** برای ذخیره
- دکمه **"Reset to Defaults"** برای بازگشت

---

## 📱 Responsive Design

داشبورد به صورت کامل responsive است:

- **Desktop** (lg): 3 ستون برای gauges
- **Tablet** (md): 2 ستون
- **Mobile** (sm): 1 ستون
- Sidebar همیشه در سمت راست

---

## 🎯 ویژگی‌های ویژه

### ✨ Animations:
- Gauge needles با transition نرم
- Progress bars با animation
- Pulse effect برای status indicators
- Hover effects روی دکمه‌ها

### 📊 Charts (Recharts):
- Interactive tooltips
- Responsive sizing
- Custom styling
- Real-time updates

### 🎨 Visual Effects:
- Gradient backgrounds
- Shadow effects (hover)
- Border glows (active states)
- Smooth transitions

---

## 🔍 تست داشبورد

### چک‌لیست:
- ✅ Sidebar راست به درستی نمایش داده می‌شود؟
- ✅ منوی Display باز می‌شود؟
- ✅ هر 4 صفحه لود می‌شوند؟
- ✅ Gauges به درستی نمایش داده می‌شوند؟
- ✅ دکمه START/PAUSE کار می‌کند؟
- ✅ Charts داده نمایش می‌دهند؟
- ✅ Thresholds قابل تغییر هستند؟

---

## 🐛 عیب‌یابی

### مشکلات رایج:

**1. صفحه لود نمی‌شود:**
- بررسی کنید که frontend server در حال اجرا باشد
- در ترمینال خطایی نمایش داده می‌شود؟
- مرورگر را refresh کنید (Ctrl+Shift+R)

**2. Gauges/Charts نمایش داده نمی‌شوند:**
- Console browser را باز کنید (F12)
- خطاهای JavaScript را بررسی کنید
- مطمئن شوید که `recharts` نصب است

**3. Styling خراب است:**
- Clear کردن cache مرورگر
- بررسی فایل‌های CSS/Tailwind
- Rebuild کردن frontend: `npm run build`

---

## 📸 نکات مهم

1. **Performance**: 
   - Gauges در هر ثانیه update می‌شوند
   - از `React.memo` برای بهینه‌سازی استفاده شده
   - Debouncing برای threshold inputs

2. **Accessibility**:
   - Keyboard navigation support
   - ARIA labels برای screen readers
   - High contrast colors

3. **Browser Support**:
   - Chrome 90+ ✅
   - Firefox 88+ ✅
   - Safari 14+ ✅
   - Edge 90+ ✅

---

## 🎉 لذت ببرید!

داشبورد i-Drill آماده استفاده است! 

**برای دیدن بهترین نتیجه:**
1. صفحه را Full Screen کنید
2. از مرورگر Chrome استفاده کنید
3. Theme تیره (Dark Mode) فعال است

---

**© 2025 i-Drill Dashboard - Designed by Parsa** 🚀

