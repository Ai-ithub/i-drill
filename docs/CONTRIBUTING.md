# 🤝 Contributing to i-Drill

راهنمای مشارکت در پروژه i-Drill

## 🙏 تشکر

از علاقه شما به مشارکت در i-Drill متشکریم! هر مشارکتی، چه کوچک و چه بزرگ، ارزشمند است.

## 📋 فهرست مطالب

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Coding Standards](#coding-standards)
5. [Commit Guidelines](#commit-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Testing Requirements](#testing-requirements)
8. [Documentation](#documentation)

## 📜 Code of Conduct

### رفتارهای مورد انتظار

- استفاده از زبان محترمانه و فراگیر
- احترام به نظرات و تجربیات مختلف
- پذیرش انتقاد سازنده
- تمرکز بر آنچه برای جامعه بهتر است
- نشان دادن همدلی با سایر اعضای جامعه

### رفتارهای غیرقابل قبول

- استفاده از زبان یا تصاویر جنسی
- توهین‌های شخصی، نظرات سیاسی یا حمله
- آزار و اذیت عمومی یا خصوصی
- انتشار اطلاعات خصوصی دیگران
- سایر رفتارهایی که در یک محیط حرفه‌ای نامناسب هستند

## 🚀 Getting Started

### 1. Fork و Clone

```bash
# Fork repository در GitHub
# سپس clone کنید
git clone https://github.com/YOUR_USERNAME/i-drill.git
cd i-drill
```

### 2. Setup Environment

```bash
# Backend
cd src/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements/backend.txt
pip install -r requirements/dev.txt

# Frontend
cd ../../frontend
npm install
```

### 3. Create Branch

```bash
git checkout -b feature/your-feature-name
# یا
git checkout -b fix/your-bug-fix
```

## 🔄 Development Process

### Workflow

1. **Issue ایجاد کنید** (برای features بزرگ)
2. **Branch بسازید** از `main`
3. **کد بنویسید** و تست کنید
4. **Commit کنید** با پیام مناسب
5. **Push کنید** به fork شما
6. **Pull Request ایجاد کنید**

### Branch Naming

- `feature/description` - برای features جدید
- `fix/description` - برای bug fixes
- `docs/description` - برای مستندات
- `refactor/description` - برای refactoring
- `test/description` - برای تست‌ها

## 📝 Coding Standards

### Python (Backend)

```python
# ✅ Good
def calculate_average(values: List[float]) -> float:
    """Calculate average of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Average value
    """
    if not values:
        raise ValueError("Values cannot be empty")
    return sum(values) / len(values)

# ❌ Bad
def calc_avg(vals):
    return sum(vals)/len(vals)
```

**Standards:**
- PEP 8 style guide
- Type hints برای همه functions
- Docstrings برای همه functions/classes
- Maximum line length: 100 characters
- Use `black` برای formatting

### TypeScript (Frontend)

```typescript
// ✅ Good
interface UserProps {
  id: string;
  name: string;
  email: string;
}

export const User: React.FC<UserProps> = ({ id, name, email }) => {
  return (
    <div>
      <h2>{name}</h2>
      <p>{email}</p>
    </div>
  );
};

// ❌ Bad
export const User = (props) => {
  return <div>{props.name}</div>;
};
```

**Standards:**
- ESLint + Prettier
- TypeScript strict mode
- Functional components
- Props interfaces
- Meaningful variable names

## 📝 Commit Guidelines

### Format

```
type(scope): subject

body

footer
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

### Examples

```bash
# Feature
feat(api): add sensor data aggregation endpoint

Add new endpoint for aggregating sensor data by time range.
Includes validation and error handling.

Closes #123

# Bug fix
fix(auth): resolve token expiration issue

Token expiration was not being checked correctly.
Now properly validates token before each request.

Fixes #456

# Documentation
docs(readme): update installation instructions

Update README with new Docker Compose setup steps.
```

## 🔀 Pull Request Process

### قبل از ارسال PR

- [ ] کد با style guidelines سازگار است
- [ ] تست‌ها pass می‌شوند
- [ ] Coverage کاهش نیافته است
- [ ] مستندات به‌روزرسانی شده است
- [ ] Breaking changes مستند شده‌اند
- [ ] Commit messages واضح هستند

### PR Template

```markdown
## Description
توضیح مختصر تغییرات

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
چگونه تست شده است؟

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. **Automated Checks**: CI/CD باید pass شود
2. **Code Review**: حداقل یک approver
3. **Testing**: تست‌های manual (در صورت نیاز)
4. **Merge**: بعد از approval

## 🧪 Testing Requirements

### Coverage Requirements

- **Backend**: 75%+ overall, 80%+ for new code
- **Frontend**: 70%+ overall, 80%+ for components

### Test Types

1. **Unit Tests**: برای همه functions/methods
2. **Integration Tests**: برای API endpoints
3. **Component Tests**: برای React components
4. **E2E Tests**: برای critical flows (اختیاری)

### Running Tests

```bash
# Backend
pytest tests/ -v --cov=src/backend

# Frontend
npm test
npm test -- --coverage
```

## 📚 Documentation

### مستندات مورد نیاز

- **Code Comments**: برای logic پیچیده
- **Docstrings**: برای همه functions/classes
- **README Updates**: اگر setup تغییر کرد
- **API Documentation**: برای endpoints جدید
- **User Guide**: برای features جدید کاربری

### Documentation Standards

- فارسی برای user-facing docs
- انگلیسی برای code comments
- Examples در همه docs
- Screenshots برای UI changes

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
## Description
توضیح مختصر bug

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
چه انتظاری داشتید؟

## Actual Behavior
چه اتفاقی افتاد؟

## Environment
- OS: [e.g., Windows 10]
- Browser: [e.g., Chrome 120]
- Version: [e.g., 1.0.0]

## Screenshots
اگر قابل اعمال است

## Additional Context
هر اطلاعات اضافی
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
توضیح feature

## Use Case
چرا این feature مفید است؟

## Proposed Solution
راه‌حل پیشنهادی

## Alternatives Considered
راه‌حل‌های جایگزین

## Additional Context
هر اطلاعات اضافی
```

## 🎯 Areas for Contribution

### High Priority

- 🐛 Bug fixes
- 📚 Documentation improvements
- 🧪 Test coverage
- ♿ Accessibility improvements
- 🌐 Internationalization

### Medium Priority

- 🎨 UI/UX improvements
- ⚡ Performance optimizations
- 🔧 Code refactoring
- 📊 Analytics features

### Low Priority

- 🎨 Design improvements
- 📝 Code comments
- 🔍 Code organization

## 📞 Questions?

اگر سوالی دارید:

- **GitHub Discussions**: [Discussions](https://github.com/Ai-ithub/i-drill/discussions)
- **GitHub Issues**: [Issues](https://github.com/Ai-ithub/i-drill/issues)
- **Email**: support@idrill.example.com

## 🙌 Recognition

همه contributors در [AUTHORS.md](../AUTHORS.md) ذکر می‌شوند.

---

**متشکریم از مشارکت شما!** 🎉

