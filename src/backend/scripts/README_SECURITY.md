# 🔐 Security Scripts Documentation

This directory contains security-related scripts for the i-Drill backend.

## Available Scripts

### 1. `generate_secret_key.py`

Generates cryptographically secure secret keys for JWT tokens and encryption.

**Usage:**
```bash
# Generate 32-byte key (recommended)
python scripts/generate_secret_key.py

# Generate 64-byte key (extra secure)
python scripts/generate_secret_key.py 64
```

**Output:**
- Prints a secure `SECRET_KEY` value
- Provides instructions for adding to `.env` file

---

### 2. `check_security.py`

Comprehensive security audit script that checks for:
- Known vulnerabilities in dependencies
- Insecure configuration patterns
- Hardcoded secrets in code
- Missing `.gitignore` patterns

**Usage:**
```bash
# Run full security check
python scripts/check_security.py

# Verbose output
python scripts/check_security.py --verbose
```

**What it checks:**
1. ✅ Dependencies for known vulnerabilities (using `safety`)
2. ✅ Environment file security (`.env` or `config.env.example`)
3. ✅ Hardcoded secrets in Python files
4. ✅ `.gitignore` configuration

**Requirements:**
- `safety` package (installed automatically if missing)

---

### 3. `update_dependencies.py`

Helps update dependencies to latest secure versions.

**Usage:**
```bash
# Check for outdated packages
python scripts/update_dependencies.py --check

# Update requirements file
python scripts/update_dependencies.py --update

# Preview changes (dry run)
python scripts/update_dependencies.py --update --dry-run
```

**Features:**
- Checks installed vs required versions
- Finds latest available versions
- Updates `requirements/backend.txt` file
- Dry-run mode for safe testing

---

## Quick Start

### Initial Security Setup

1. **Generate Secret Key:**
   ```bash
   python scripts/generate_secret_key.py
   ```

2. **Create `.env` file:**
   ```bash
   cp config.env.example .env
   # Edit .env and add the generated SECRET_KEY
   ```

3. **Run Security Check:**
   ```bash
   python scripts/check_security.py
   ```

4. **Check Dependencies:**
   ```bash
   python scripts/update_dependencies.py --check
   ```

---

## Security Best Practices

### Before Deployment

1. ✅ Run `check_security.py` and fix all issues
2. ✅ Generate new `SECRET_KEY` for production
3. ✅ Change all default passwords
4. ✅ Update dependencies to latest secure versions
5. ✅ Review `.env` file for insecure values
6. ✅ Ensure `.gitignore` excludes `.env` files

### Regular Maintenance

- **Weekly**: Run `check_security.py`
- **Monthly**: Update dependencies with `update_dependencies.py`
- **Quarterly**: Rotate `SECRET_KEY` and passwords

---

## Troubleshooting

### `safety` not found

The script will automatically install `safety` if it's missing. If installation fails:

```bash
pip install safety
```

### Dependency update fails

If `update_dependencies.py` fails to check versions:

1. Ensure you have internet connection
2. Check that `pip` is up to date: `pip install --upgrade pip`
3. Some packages may not be available on PyPI

### Security check shows false positives

Some warnings may be false positives:
- Development-only configurations (acceptable if `APP_ENV=development`)
- Test files with example credentials
- Documentation files

Review each warning and fix actual security issues.

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Security Check

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements/backend.txt
      - run: pip install safety
      - run: python src/backend/scripts/check_security.py
```

---

## Additional Resources

- [SECURITY.md](../SECURITY.md) - Complete security guide
- [config.env.example](../config.env.example) - Environment configuration template
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Security best practices

---

**Last Updated**: 2025-01-XX

