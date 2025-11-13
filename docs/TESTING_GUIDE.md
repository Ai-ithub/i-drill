# 🧪 Testing Guide

راهنمای کامل تست‌نویسی در i-Drill

## 📋 فهرست مطالب

1. [Overview](#overview)
2. [Backend Testing](#backend-testing)
3. [Frontend Testing](#frontend-testing)
4. [Integration Testing](#integration-testing)
5. [E2E Testing](#e2e-testing)
6. [Performance Testing](#performance-testing)
7. [Test Coverage](#test-coverage)
8. [CI/CD Integration](#cicd-integration)

## 🎯 Overview

i-Drill از چندین نوع تست استفاده می‌کند:

- **Unit Tests**: تست واحدهای کوچک کد
- **Integration Tests**: تست تعامل بین کامپوننت‌ها
- **E2E Tests**: تست end-to-end
- **Performance Tests**: تست عملکرد

### Test Stack

**Backend:**
- pytest
- pytest-asyncio
- pytest-cov
- httpx (برای API testing)

**Frontend:**
- Vitest
- React Testing Library
- @testing-library/user-event

## 🔧 Backend Testing

### Setup

```bash
# نصب dependencies
pip install -r requirements/dev.txt

# اجرای تست‌ها
pytest tests/ -v

# با coverage
pytest tests/ -v --cov=src/backend --cov-report=html
```

### Unit Tests

```python
# tests/test_example_service.py
import pytest
from services.example_service import ExampleService

def test_example_service_create():
    """Test creating an example"""
    service = ExampleService()
    result = service.create({"name": "Test"})
    assert result["name"] == "Test"
    assert "id" in result
```

### API Tests

```python
# tests/test_example_api.py
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_create_example():
    """Test POST /api/v1/example/"""
    response = client.post(
        "/api/v1/example/",
        json={"name": "Test Example"}
    )
    assert response.status_code == 201
    assert response.json()["name"] == "Test Example"

def test_get_example():
    """Test GET /api/v1/example/{id}"""
    # Create first
    create_response = client.post(
        "/api/v1/example/",
        json={"name": "Test"}
    )
    example_id = create_response.json()["id"]
    
    # Get
    response = client.get(f"/api/v1/example/{example_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test"
```

### Async Tests

```python
# tests/test_async_service.py
import pytest
from services.async_service import AsyncService

@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation"""
    service = AsyncService()
    result = await service.async_operation()
    assert result is not None
```

### Database Tests

```python
# tests/test_database.py
import pytest
from database import get_db, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session():
    """Create test database session"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_record(db_session):
    """Test database operation"""
    from models.example import Example
    example = Example(name="Test")
    db_session.add(example)
    db_session.commit()
    assert example.id is not None
```

### Mocking

```python
# tests/test_with_mock.py
from unittest.mock import Mock, patch
import pytest

@patch('services.external_service.ExternalService.call')
def test_with_mock(mock_call):
    """Test with mocked external service"""
    mock_call.return_value = {"status": "success"}
    
    from services.example_service import ExampleService
    service = ExampleService()
    result = service.use_external()
    
    assert result["status"] == "success"
    mock_call.assert_called_once()
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
def auth_headers(client):
    """Authentication headers fixture"""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "test", "password": "test"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```

## 🎨 Frontend Testing

### Setup

```bash
# نصب dependencies (در package.json موجود است)
npm install

# اجرای تست‌ها
npm test

# با coverage
npm test -- --coverage

# Watch mode
npm test -- --watch
```

### Component Tests

```typescript
// frontend/src/components/Example/__tests__/Example.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Example } from '../Example';

describe('Example', () => {
  it('renders correctly', () => {
    render(<Example title="Test" />);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });

  it('handles click events', () => {
    const handleClick = vi.fn();
    render(<Example onClick={handleClick} />);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
```

### Hook Tests

```typescript
// frontend/src/hooks/__tests__/useExample.test.ts
import { describe, it, expect } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useExample } from '../useExample';

describe('useExample', () => {
  it('fetches data', async () => {
    const { result } = renderHook(() => useExample('123'));
    
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });
    
    expect(result.current.data).toBeDefined();
  });
});
```

### API Mocking

```typescript
// frontend/src/services/__tests__/exampleService.test.ts
import { describe, it, expect, vi } from 'vitest';
import { exampleService } from '../exampleService';

// Mock fetch
global.fetch = vi.fn();

describe('exampleService', () => {
  it('fetches example data', async () => {
    const mockData = { id: 1, name: 'Test' };
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });

    const result = await exampleService.getExample(1);
    expect(result).toEqual(mockData);
  });
});
```

### Snapshot Tests

```typescript
// frontend/src/components/Example/__tests__/Example.test.tsx
import { render } from '@testing-library/react';
import { Example } from '../Example';

it('matches snapshot', () => {
  const { container } = render(<Example title="Test" />);
  expect(container).toMatchSnapshot();
});
```

## 🔗 Integration Testing

### API Integration Tests

```python
# tests/integration/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_full_workflow(client):
    """Test complete workflow"""
    # 1. Register user
    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123"
        }
    )
    assert register_response.status_code == 201
    
    # 2. Login
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": "testuser", "password": "password123"}
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Create sensor data
    sensor_response = client.post(
        "/api/v1/sensor-data/",
        json={"rig_id": "RIG_01", "depth": 1000.0},
        headers=headers
    )
    assert sensor_response.status_code == 201
    
    # 4. Get sensor data
    get_response = client.get(
        "/api/v1/sensor-data/realtime",
        headers=headers
    )
    assert get_response.status_code == 200
    assert len(get_response.json()["data"]) > 0
```

## 🎭 E2E Testing

### Playwright Setup

```bash
# نصب Playwright
npm install -D @playwright/test
npx playwright install
```

### E2E Test Example

```typescript
// e2e/example.spec.ts
import { test, expect } from '@playwright/test';

test('user can login and view dashboard', async ({ page }) => {
  // Navigate to login
  await page.goto('http://localhost:3000/login');
  
  // Fill login form
  await page.fill('input[name="username"]', 'testuser');
  await page.fill('input[name="password"]', 'password123');
  await page.click('button[type="submit"]');
  
  // Wait for redirect
  await page.waitForURL('**/dashboard');
  
  // Verify dashboard content
  await expect(page.locator('h1')).toContainText('Dashboard');
});
```

## ⚡ Performance Testing

### Load Testing

```python
# tests/performance/test_load.py
import pytest
import asyncio
from httpx import AsyncClient
from app import app

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent API requests"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        tasks = [
            client.get("/api/v1/sensor-data/realtime")
            for _ in range(100)
        ]
        responses = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in responses)
```

### Frontend Performance

```typescript
// frontend/src/__tests__/performance.test.tsx
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { Example } from '@/components/Example';

describe('Performance', () => {
  it('renders quickly', () => {
    const start = performance.now();
    render(<Example />);
    const end = performance.now();
    
    expect(end - start).toBeLessThan(100); // < 100ms
  });
});
```

## 📊 Test Coverage

### Backend Coverage

```bash
# Generate coverage report
pytest --cov=src/backend --cov-report=html

# View report
open htmlcov/index.html
```

### Frontend Coverage

```bash
# Generate coverage report
npm test -- --coverage

# View report
open coverage/index.html
```

### Coverage Goals

- **Backend**: 75%+ overall, 80%+ for new code
- **Frontend**: 70%+ overall, 80%+ for components

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements/dev.txt
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      - run: npm ci
      - run: npm test -- --coverage
```

## 📝 Best Practices

### Test Organization

1. **Structure:**
   ```
   tests/
   ├── unit/
   ├── integration/
   ├── e2e/
   └── fixtures/
   ```

2. **Naming:**
   - `test_*.py` for Python
   - `*.test.tsx` for React components
   - `*.spec.ts` for E2E tests

3. **Isolation:**
   - هر تست باید مستقل باشد
   - از shared state اجتناب کنید
   - از fixtures برای setup استفاده کنید

### Test Quality

1. **AAA Pattern:**
   ```python
   def test_example():
       # Arrange
       service = ExampleService()
       
       # Act
       result = service.create({"name": "Test"})
       
       # Assert
       assert result["name"] == "Test"
   ```

2. **Descriptive Names:**
   ```python
   # Good
   def test_create_user_with_valid_data_returns_201():
       pass
   
   # Bad
   def test_user():
       pass
   ```

3. **One Assertion Per Test:**
   ```python
   # Good
   def test_user_has_id():
       assert user.id is not None
   
   def test_user_has_name():
       assert user.name == "Test"
   ```

## 📚 منابع بیشتر

- [pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/)

