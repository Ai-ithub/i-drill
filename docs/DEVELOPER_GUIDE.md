# 🛠️ Developer Guide

راهنمای کامل برای توسعه‌دهندگان i-Drill

## 📋 فهرست مطالب

1. [شروع کار](#شروع-کار)
2. [ساختار پروژه](#ساختار-پروژه)
3. [راه‌اندازی محیط توسعه](#راه-اندازی-محیط-توسعه)
4. [معماری و الگوها](#معماری-و-الگوها)
5. [توسعه Backend](#توسعه-backend)
6. [توسعه Frontend](#توسعه-frontend)
7. [تست‌نویسی](#تست-نویسی)
8. [Debugging](#debugging)
9. [Best Practices](#best-practices)
10. [Contribution Guidelines](#contribution-guidelines)

## 🚀 شروع کار

### پیش‌نیازها

- **Python 3.12+**
- **Node.js 18+**
- **PostgreSQL 15+**
- **Docker & Docker Compose** (اختیاری)
- **Git**

### نصب اولیه

```bash
# Clone repository
git clone https://github.com/Ai-ithub/i-drill.git
cd i-drill

# Backend setup
cd src/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements/backend.txt
pip install -r requirements/dev.txt

# Frontend setup
cd ../../frontend
npm install
```

### راه‌اندازی محیط توسعه

```bash
# Start services with Docker Compose
docker-compose up -d postgres kafka zookeeper

# Run backend
cd src/backend
uvicorn app:app --reload --port 8001

# Run frontend (in another terminal)
cd frontend
npm run dev
```

## 📁 ساختار پروژه

```
i-drill/
├── src/
│   ├── backend/              # Backend API
│   │   ├── api/              # API routes
│   │   │   └── routes/       # Route handlers
│   │   ├── services/         # Business logic
│   │   ├── models/           # Database models
│   │   ├── database/         # DB configuration
│   │   └── app.py            # FastAPI app
│   ├── drilling_env/         # RL environment
│   ├── rul_prediction/       # RUL models
│   └── predictive_maintenance/ # Maintenance models
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API clients
│   │   └── utils/           # Utilities
│   └── package.json
├── tests/                    # Backend tests
├── docs/                     # Documentation
└── docker-compose.yml        # Docker configuration
```

## 🏗️ معماری و الگوها

### Backend Architecture

```
┌─────────────┐
│   Routes    │  ← API endpoints
└──────┬──────┘
       │
┌──────▼──────┐
│  Services   │  ← Business logic
└──────┬──────┘
       │
┌──────▼──────┐
│   Models    │  ← Database models
└──────┬──────┘
       │
┌──────▼──────┐
│  Database   │  ← PostgreSQL
└─────────────┘
```

### Frontend Architecture

```
┌─────────────┐
│   Pages     │  ← Page components
└──────┬──────┘
       │
┌──────▼──────┐
│ Components  │  ← Reusable components
└──────┬──────┘
       │
┌──────▼──────┐
│   Hooks     │  ← Custom hooks (React Query)
└──────┬──────┘
       │
┌──────▼──────┐
│  Services   │  ← API clients
└─────────────┘
```

### الگوهای طراحی

- **Repository Pattern**: برای دسترسی به داده
- **Service Layer**: برای منطق کسب‌وکار
- **Dependency Injection**: برای وابستگی‌ها
- **Factory Pattern**: برای ایجاد سرویس‌ها

## 🔧 توسعه Backend

### ایجاد Route جدید

```python
# src/backend/api/routes/example.py
from fastapi import APIRouter, HTTPException
from api.models.schemas import ExampleRequest, ExampleResponse
from services.example_service import ExampleService

router = APIRouter(prefix="/example", tags=["example"])
service = ExampleService()

@router.post("/", response_model=ExampleResponse)
async def create_example(request: ExampleRequest):
    """Create a new example"""
    try:
        result = service.create(request)
        return ExampleResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### ایجاد Service جدید

```python
# src/backend/services/example_service.py
from database import get_db
from models.example import Example
import logging

logger = logging.getLogger(__name__)

class ExampleService:
    def __init__(self):
        self.db = next(get_db())
    
    def create(self, request):
        """Create example"""
        example = Example(**request.dict())
        self.db.add(example)
        self.db.commit()
        return {"id": example.id, "message": "Created"}
```

### ایجاد Model جدید

```python
# src/backend/models/example.py
from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from datetime import datetime

class Example(Base):
    __tablename__ = "examples"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### ثبت Route در App

```python
# src/backend/app.py
from api.routes import example

app.include_router(example.router, prefix="/api/v1")
```

## 🎨 توسعه Frontend

### ایجاد Component جدید

```typescript
// frontend/src/components/Example/Example.tsx
import React from 'react';
import { Card } from '@/components/UI';

interface ExampleProps {
  title: string;
  data: any[];
}

export const Example: React.FC<ExampleProps> = ({ title, data }) => {
  return (
    <Card>
      <Card.Header>
        <h2>{title}</h2>
      </Card.Header>
      <Card.Content>
        {/* Component content */}
      </Card.Content>
    </Card>
  );
};
```

### ایجاد Page جدید

```typescript
// frontend/src/pages/Example/Example.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { exampleService } from '@/services';
import { Loading, ErrorDisplay } from '@/components/UI';

export const ExamplePage: React.FC = () => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['example'],
    queryFn: () => exampleService.getExample(),
  });

  if (isLoading) return <Loading />;
  if (error) return <ErrorDisplay error={error} />;

  return (
    <div>
      {/* Page content */}
    </div>
  );
};
```

### ایجاد Custom Hook

```typescript
// frontend/src/hooks/useExample.ts
import { useQuery } from '@tanstack/react-query';
import { exampleService } from '@/services';

export const useExample = (id: string) => {
  return useQuery({
    queryKey: ['example', id],
    queryFn: () => exampleService.getExampleById(id),
    enabled: !!id,
  });
};
```

### ایجاد API Service

```typescript
// frontend/src/services/exampleService.ts
import { apiClient } from './apiClient';

export const exampleService = {
  getExample: async () => {
    const response = await apiClient.get('/example');
    return response.data;
  },
  
  getExampleById: async (id: string) => {
    const response = await apiClient.get(`/example/${id}`);
    return response.data;
  },
  
  createExample: async (data: any) => {
    const response = await apiClient.post('/example', data);
    return response.data;
  },
};
```

## 🧪 تست‌نویسی

### Backend Tests

```python
# tests/test_example.py
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_create_example():
    response = client.post(
        "/api/v1/example/",
        json={"name": "Test Example"}
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test Example"
```

### Frontend Tests

```typescript
// frontend/src/components/Example/__tests__/Example.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Example } from '../Example';

describe('Example', () => {
  it('renders with title', () => {
    render(<Example title="Test" data={[]} />);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
```

### اجرای تست‌ها

```bash
# Backend
pytest tests/ -v --cov=src/backend

# Frontend
npm test
npm test -- --coverage
```

## 🐛 Debugging

### Backend Debugging

```python
# استفاده از logging
import logging
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Frontend Debugging

```typescript
// استفاده از console
console.log('Debug:', data);
console.error('Error:', error);

// React DevTools
// Chrome DevTools
```

### VS Code Debugging

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["app:app", "--reload"],
      "jinja": true
    }
  ]
}
```

## ✅ Best Practices

### Backend

1. **Type Hints**: همیشه از type hints استفاده کنید
2. **Error Handling**: خطاها را به درستی handle کنید
3. **Logging**: از logging برای debugging استفاده کنید
4. **Documentation**: Docstrings را کامل بنویسید
5. **Testing**: برای هر feature تست بنویسید

### Frontend

1. **TypeScript**: از TypeScript استفاده کنید
2. **Component Structure**: کامپوننت‌ها را کوچک نگه دارید
3. **Reusability**: کامپوننت‌های قابل استفاده مجدد بسازید
4. **Error Boundaries**: از Error Boundaries استفاده کنید
5. **Performance**: از React.memo و useMemo استفاده کنید

### Code Style

- **Python**: PEP 8
- **TypeScript**: ESLint + Prettier
- **Git**: Conventional Commits

## 🤝 Contribution Guidelines

### فرآیند Contribution

1. **Fork** repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### Commit Message Format

```
type(scope): subject

body

footer
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Example:**
```
feat(api): add example endpoint

Add new example endpoint for creating examples.
Includes validation and error handling.

Closes #123
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance considered
- [ ] Security reviewed

## 📚 منابع بیشتر

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/)
- [Testing Guide](tests/README.md)
- [API Documentation](src/backend/API_DOCUMENTATION.md)
- [Architecture Guide](docs/ARCHITECTURE.md)

## ❓ سوالات متداول

### چگونه یک feature جدید اضافه کنم؟

1. Issue ایجاد کنید
2. Branch جدید بسازید
3. کد را بنویسید و تست کنید
4. Pull Request ایجاد کنید

### چگونه با database کار کنم؟

از SQLAlchemy models استفاده کنید. برای migrations از Alembic استفاده کنید.

### چگونه API را test کنم؟

از FastAPI TestClient استفاده کنید یا از pytest با httpx استفاده کنید.

### چگونه frontend را optimize کنم؟

- از React.memo استفاده کنید
- از useMemo و useCallback استفاده کنید
- Code splitting انجام دهید
- Lazy loading استفاده کنید

