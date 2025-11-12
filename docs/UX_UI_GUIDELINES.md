# 🎨 UX/UI Guidelines

راهنمای طراحی و تجربه کاربری برای i-Drill

## 🎯 Design Principles

### 1. Clarity
- استفاده از زبان واضح و ساده
- نمایش اطلاعات به صورت منطقی
- استفاده از آیکون‌های واضح

### 2. Consistency
- استفاده از design system یکپارچه
- رنگ‌ها و فونت‌های ثابت
- الگوهای تعاملی یکسان

### 3. Feedback
- نمایش وضعیت عملیات
- پیام‌های خطای واضح
- Loading states مناسب

### 4. Efficiency
- دسترسی سریع به عملکردهای مهم
- Keyboard shortcuts
- Bulk operations

## 🎨 Design System

### Colors

#### Primary Colors
- **Blue**: `#3b82f6` - Actions, links
- **Green**: `#10b981` - Success, positive states
- **Red**: `#ef4444` - Errors, warnings
- **Yellow**: `#f59e0b` - Warnings

#### Neutral Colors
- **Gray-50**: `#f9fafb` - Backgrounds
- **Gray-900**: `#111827` - Text primary
- **Gray-600**: `#4b5563` - Text secondary

### Typography

#### Font Families
- **Sans-serif**: System fonts (Inter, Roboto)
- **Mono**: For numbers and codes

#### Font Sizes
- **Heading 1**: 2.25rem (36px)
- **Heading 2**: 1.875rem (30px)
- **Body**: 1rem (16px)
- **Small**: 0.875rem (14px)

### Spacing

استفاده از spacing scale:
- **xs**: 0.25rem (4px)
- **sm**: 0.5rem (8px)
- **md**: 1rem (16px)
- **lg**: 1.5rem (24px)
- **xl**: 2rem (32px)

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Mobile-First Approach
همیشه از mobile شروع کنید و به desktop برسید.

## ♿ Accessibility

### WCAG 2.1 AA Compliance

#### Color Contrast
- Normal text: حداقل 4.5:1
- Large text: حداقل 3:1
- Interactive elements: حداقل 3:1

#### Keyboard Navigation
- تمام عملکردها با keyboard قابل دسترسی
- Focus indicators واضح
- Tab order منطقی

#### Screen Readers
- استفاده از ARIA labels
- Semantic HTML
- Alt text برای images

### Best Practices

1. **Skip Links**: لینک برای skip به main content
2. **Focus Management**: مدیریت focus در modals
3. **Error Messages**: پیام‌های خطای واضح
4. **Loading States**: نمایش وضعیت loading

## 🎭 Component Patterns

### Buttons

```tsx
// Primary button
<button className="btn btn-primary">Save</button>

// Secondary button
<button className="btn btn-secondary">Cancel</button>

// Destructive button
<button className="btn btn-destructive">Delete</button>
```

### Forms

```tsx
<form>
  <label htmlFor="rig-id">Rig ID</label>
  <input
    id="rig-id"
    type="text"
    required
    aria-describedby="rig-id-error"
  />
  <span id="rig-id-error" className="error-message" role="alert">
    Rig ID is required
  </span>
</form>
```

### Cards

```tsx
<div className="card">
  <div className="card-header">
    <h3>Title</h3>
  </div>
  <div className="card-body">
    Content
  </div>
</div>
```

## 📊 Data Visualization

### Charts
- استفاده از رنگ‌های متمایز
- Legend واضح
- Tooltips informative
- Responsive design

### Tables
- Sortable columns
- Pagination
- Filtering
- Responsive (scroll on mobile)

## 🚀 Performance

### Loading States
- Skeleton screens برای content
- Progress indicators برای operations
- Optimistic updates

### Optimization
- Lazy loading برای images
- Code splitting
- Memoization برای expensive computations

## 🌐 Internationalization

### RTL Support
- پشتیبانی کامل از RTL برای فارسی
- Mirror کردن layout
- Text alignment مناسب

### Localization
- تاریخ و زمان محلی
- فرمت اعداد محلی
- واحدهای اندازه‌گیری

## ✅ UX Checklist

- [ ] Navigation واضح است
- [ ] Error messages مفید هستند
- [ ] Loading states نمایش داده می‌شوند
- [ ] Mobile responsive است
- [ ] Keyboard navigation کار می‌کند
- [ ] Screen reader compatible است
- [ ] Color contrast مناسب است
- [ ] Touch targets به اندازه کافی بزرگ هستند (44x44px)
- [ ] RTL support برای فارسی
- [ ] Performance بهینه است

