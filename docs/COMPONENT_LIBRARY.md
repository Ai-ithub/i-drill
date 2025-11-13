# 🎨 Component Library

راهنمای کامل کامپوننت‌های UI در i-Drill

## 📋 فهرست مطالب

1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [Usage Examples](#usage-examples)
5. [Styling](#styling)
6. [Accessibility](#accessibility)

## 🎯 Overview

Component Library شامل کامپوننت‌های قابل استفاده مجدد برای ساخت رابط کاربری i-Drill است. همه کامپوننت‌ها:
- ✅ TypeScript support
- ✅ Responsive design
- ✅ Accessibility (WCAG 2.1 AA)
- ✅ Dark mode support
- ✅ Customizable styling

## 📦 Installation

کامپوننت‌ها در `frontend/src/components/UI/` قرار دارند:

```typescript
import { Button, Card, Input, Loading } from '@/components/UI';
```

## 🧩 Components

### Button

دکمه قابل استفاده مجدد با variants و sizes مختلف.

```typescript
import { Button } from '@/components/UI';

// Basic usage
<Button>Click me</Button>

// With variant
<Button variant="primary">Primary</Button>
<Button variant="secondary">Secondary</Button>
<Button variant="destructive">Delete</Button>
<Button variant="outline">Outline</Button>
<Button variant="ghost">Ghost</Button>

// With size
<Button size="sm">Small</Button>
<Button size="md">Medium</Button>
<Button size="lg">Large</Button>

// With loading state
<Button isLoading>Loading...</Button>

// With icons
<Button leftIcon={<Icon />}>With Icon</Button>
<Button rightIcon={<Icon />}>With Icon</Button>

// Disabled
<Button disabled>Disabled</Button>

// Full width
<Button fullWidth>Full Width</Button>
```

**Props:**
- `variant`: `"primary" | "secondary" | "destructive" | "outline" | "ghost"`
- `size`: `"sm" | "md" | "lg"`
- `isLoading`: `boolean`
- `disabled`: `boolean`
- `fullWidth`: `boolean`
- `leftIcon`: `ReactNode`
- `rightIcon`: `ReactNode`

### Card

کارت برای نمایش محتوای گروه‌بندی شده.

```typescript
import { Card } from '@/components/UI';

<Card>
  <Card.Header>
    <h2>Title</h2>
  </Card.Header>
  <Card.Content>
    <p>Content goes here</p>
  </Card.Content>
  <Card.Footer>
    <Button>Action</Button>
  </Card.Footer>
</Card>

// With variant
<Card variant="elevated">Elevated Card</Card>
<Card variant="outlined">Outlined Card</Card>
```

**Props:**
- `variant`: `"default" | "elevated" | "outlined"`
- `className`: `string`

### Input

فیلد ورودی با label، error و icon support.

```typescript
import { Input } from '@/components/UI';

// Basic usage
<Input
  label="Email"
  type="email"
  placeholder="Enter your email"
/>

// With error
<Input
  label="Email"
  error="Email is required"
/>

// With helper text
<Input
  label="Password"
  helperText="Must be at least 8 characters"
/>

// With icon
<Input
  label="Search"
  leftIcon={<SearchIcon />}
/>

// Disabled
<Input
  label="Disabled"
  disabled
/>
```

**Props:**
- `label`: `string`
- `error`: `string`
- `helperText`: `string`
- `leftIcon`: `ReactNode`
- `rightIcon`: `ReactNode`
- `disabled`: `boolean`
- `required`: `boolean`

### Loading

کامپوننت‌های loading state.

```typescript
import { Loading, Skeleton, SkeletonText } from '@/components/UI';

// Spinner
<Loading />

// Skeleton
<Skeleton width={200} height={100} />

// Skeleton text
<SkeletonText lines={3} />
```

**Props:**
- `Loading`: No props
- `Skeleton`: `width`, `height`, `className`
- `SkeletonText`: `lines`, `className`

### Toast

سیستم اعلان‌ها.

```typescript
import { toast } from '@/components/UI/Toast';

// Success
toast.success('Operation successful!');

// Error
toast.error('Something went wrong');

// Warning
toast.warning('Please check your input');

// Info
toast.info('New update available');
```

**Usage in component:**

```typescript
import { ToastContainer } from '@/components/UI';

function App() {
  return (
    <>
      <YourApp />
      <ToastContainer />
    </>
  );
}
```

### EmptyState

نمایش پیام زمانی که داده‌ای وجود ندارد.

```typescript
import { EmptyState } from '@/components/UI';

<EmptyState
  title="No data found"
  description="There is no data to display"
  icon={<Icon />}
  action={
    <Button onClick={handleAction}>Add Data</Button>
  }
/>
```

**Props:**
- `title`: `string`
- `description`: `string`
- `icon`: `ReactNode`
- `action`: `ReactNode`

### ErrorDisplay

نمایش خطا با گزینه retry.

```typescript
import { ErrorDisplay } from '@/components/UI';

<ErrorDisplay
  error={error}
  onRetry={() => refetch()}
  onGoHome={() => navigate('/')}
/>
```

**Props:**
- `error`: `Error | string`
- `onRetry`: `() => void`
- `onGoHome`: `() => void`
- `variant`: `"default" | "minimal" | "detailed"`

## 💡 Usage Examples

### Form Example

```typescript
import { Card, Input, Button } from '@/components/UI';
import { useState } from 'react';

function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  return (
    <Card>
      <Card.Header>
        <h2>Login</h2>
      </Card.Header>
      <Card.Content>
        <Input
          label="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Input
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </Card.Content>
      <Card.Footer>
        <Button fullWidth>Login</Button>
      </Card.Footer>
    </Card>
  );
}
```

### Data Display Example

```typescript
import { Card, Loading, ErrorDisplay, EmptyState } from '@/components/UI';
import { useQuery } from '@tanstack/react-query';

function DataDisplay() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['data'],
    queryFn: fetchData,
  });

  if (isLoading) return <Loading />;
  if (error) return <ErrorDisplay error={error} />;
  if (!data) return <EmptyState title="No data" />;

  return (
    <Card>
      <Card.Content>
        {/* Display data */}
      </Card.Content>
    </Card>
  );
}
```

### Table with Actions

```typescript
import { Card, Button } from '@/components/UI';

function DataTable() {
  return (
    <Card>
      <Card.Header>
        <h2>Data Table</h2>
        <Button variant="primary">Add New</Button>
      </Card.Header>
      <Card.Content>
        <table>
          {/* Table content */}
        </table>
      </Card.Content>
    </Card>
  );
}
```

## 🎨 Styling

### Custom Styling

کامپوننت‌ها از Tailwind CSS استفاده می‌کنند. می‌توانید با `className` استایل‌های سفارشی اضافه کنید:

```typescript
<Button className="custom-class">Button</Button>
```

### Theme Customization

برای تغییر theme، فایل `tailwind.config.js` را ویرایش کنید:

```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0891b2', // cyan-600
          // ... other shades
        },
      },
    },
  },
};
```

## ♿ Accessibility

همه کامپوننت‌ها با استانداردهای WCAG 2.1 AA سازگار هستند:

- ✅ Keyboard navigation
- ✅ Screen reader support
- ✅ Focus management
- ✅ ARIA labels
- ✅ Color contrast
- ✅ Touch target sizes (44x44px minimum)

### Keyboard Navigation

- `Tab`: Navigate between interactive elements
- `Enter/Space`: Activate buttons
- `Escape`: Close modals/dropdowns

### Screen Reader Support

کامپوننت‌ها به صورت خودکار ARIA attributes را اضافه می‌کنند:

```typescript
<Button aria-label="Close dialog">×</Button>
```

## 📚 API Reference

برای جزئیات کامل API هر کامپوننت، به فایل‌های source مراجعه کنید:

- `frontend/src/components/UI/Button.tsx`
- `frontend/src/components/UI/Card.tsx`
- `frontend/src/components/UI/Input.tsx`
- `frontend/src/components/UI/Loading.tsx`
- `frontend/src/components/UI/Toast.tsx`
- `frontend/src/components/UI/EmptyState.tsx`
- `frontend/src/components/UI/ErrorDisplay.tsx`

## 🧪 Testing

همه کامپوننت‌ها تست شده‌اند. برای مشاهده تست‌ها:

```bash
npm test -- Button.test.tsx
```

## 🔄 Changelog

### v1.0.0
- Initial release
- Button, Card, Input, Loading, Toast, EmptyState, ErrorDisplay components

## 📖 منابع بیشتر

- [React Documentation](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Accessibility Guide](docs/UX_UI_GUIDELINES.md)

