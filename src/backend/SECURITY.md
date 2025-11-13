# 🔐 Security Guide for i-Drill Backend

This document outlines security best practices and requirements for the i-Drill backend application.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Secret Key Management](#secret-key-management)
3. [Database Security](#database-security)
4. [Authentication & Authorization](#authentication--authorization)
5. [Dependency Security](#dependency-security)
6. [Environment Configuration](#environment-configuration)
7. [Production Deployment](#production-deployment)
8. [Security Checklist](#security-checklist)

---

## 🚀 Quick Start

### 1. Generate Secure Secret Key

```bash
cd src/backend
python scripts/generate_secret_key.py
```

Copy the generated `SECRET_KEY` to your `.env` file.

### 2. Create Secure Environment File

```bash
cp config.env.example .env
# Edit .env and set all required values
```

### 3. Run Security Check

```bash
python scripts/check_security.py
```

---

## 🔑 Secret Key Management

### Generating a Secret Key

**CRITICAL**: Never use default or placeholder secret keys in production!

```bash
# Generate a 32-byte key (recommended)
python scripts/generate_secret_key.py

# Generate a 64-byte key (extra secure)
python scripts/generate_secret_key.py 64
```

### Setting the Secret Key

Add to your `.env` file:

```env
SECRET_KEY=your-generated-secret-key-here
```

### Security Requirements

- **Minimum Length**: 32 characters
- **Character Set**: URL-safe base64 (letters, numbers, `-`, `_`)
- **Generation**: Use cryptographically secure random generator
- **Storage**: Never commit to version control
- **Rotation**: Rotate keys periodically (recommended: every 90 days)

### Validation

The application automatically validates secret keys on startup:

- ✅ Checks for minimum length (32 chars)
- ✅ Warns about insecure patterns
- ✅ Blocks startup in production if insecure key detected

---

## 🗄️ Database Security

### Connection String Format

```env
DATABASE_URL=postgresql://username:password@host:port/database
```

### Security Best Practices

1. **Use Strong Passwords**
   - Minimum 16 characters
   - Mix of uppercase, lowercase, numbers, special characters
   - Avoid dictionary words

2. **Limit Database Access**
   - Use dedicated database user (not `postgres` superuser)
   - Grant only necessary permissions
   - Use connection pooling

3. **Encrypt Connections**
   - Use SSL/TLS for database connections in production
   - Set `sslmode=require` in connection string

4. **Mask Passwords in Logs**
   - Database URLs are automatically masked in logs
   - Never log plain passwords

### Example Secure Configuration

```env
# Development
DATABASE_URL=postgresql://drill_user:StrongP@ssw0rd123@localhost:5432/drilling_db

# Production (with SSL)
DATABASE_URL=postgresql://drill_user:StrongP@ssw0rd123@db.example.com:5432/drilling_db?sslmode=require
```

---

## 🔐 Authentication & Authorization

### Default Admin Account

The application creates a default admin account on first startup:

**⚠️ WARNING**: Change default credentials immediately in production!

```env
# Development (acceptable)
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_PASSWORD=admin123

# Production (REQUIRED to change)
DEFAULT_ADMIN_USERNAME=your_admin_username
DEFAULT_ADMIN_PASSWORD=YourStrongPassword123!
```

### Password Requirements

- **Minimum Length**: 8 characters (12+ recommended for production)
- **Hashing**: Passwords are hashed using bcrypt
- **Storage**: Only hashed passwords are stored in database

### JWT Token Configuration

```env
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours
```

**Security Recommendations**:
- Use shorter expiration times in production (e.g., 60 minutes)
- Implement token refresh mechanism
- Use HTTPS only in production

---

## 📦 Dependency Security

### Checking for Vulnerabilities

```bash
# Install safety (vulnerability scanner)
pip install safety

# Scan requirements
safety check --file requirements/backend.txt

# Or use our security check script
python scripts/check_security.py
```

### Updating Dependencies

1. **Regular Updates**: Update dependencies monthly
2. **Security Patches**: Apply security patches immediately
3. **Version Pinning**: Pin exact versions in production
4. **Audit Logs**: Keep audit logs of dependency updates

### Critical Dependencies

These packages handle security-sensitive operations:

- `python-jose[cryptography]` - JWT token handling
- `passlib[bcrypt]` - Password hashing
- `bcrypt` - Cryptographic hashing
- `fastapi` - Web framework security
- `sqlalchemy` - Database security

**Always keep these updated!**

---

## ⚙️ Environment Configuration

### Required Environment Variables

```env
# Application Environment
APP_ENV=production  # or 'development'

# Secret Key (REQUIRED)
SECRET_KEY=<generated-secure-key>

# Database
DATABASE_URL=<secure-connection-string>

# Admin Account (change in production!)
DEFAULT_ADMIN_USERNAME=<username>
DEFAULT_ADMIN_PASSWORD=<strong-password>

# CORS (restrict in production)
CORS_ORIGINS=https://yourdomain.com
```

### Optional Security Settings

```env
# Rate Limiting
ENABLE_RATE_LIMIT=true
RATE_LIMIT_DEFAULT=100/minute
RATE_LIMIT_AUTH=5/minute

# Redis Password (if enabled)
REDIS_PASSWORD=<secure-password>

# Kafka Authentication (if enabled)
KAFKA_USERNAME=<username>
KAFKA_PASSWORD=<secure-password>
```

### Environment File Security

1. **Never commit `.env` files** to version control
2. **Use `.env.example`** as a template
3. **Restrict file permissions**: `chmod 600 .env`
4. **Use secrets management** in production (AWS Secrets Manager, HashiCorp Vault, etc.)

---

## 🚀 Production Deployment

### Pre-Deployment Checklist

- [ ] Generate and set secure `SECRET_KEY`
- [ ] Change default admin credentials
- [ ] Use strong database passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS for specific domains
- [ ] Set `APP_ENV=production`
- [ ] Disable debug mode (`API_DEBUG=false`)
- [ ] Set up rate limiting
- [ ] Configure secure database connection (SSL)
- [ ] Review and update all dependencies
- [ ] Run security check script
- [ ] Set up monitoring and alerting
- [ ] Configure backup and recovery

### Security Headers

The application includes security headers via FastAPI middleware:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security` (when using HTTPS)

### Network Security

1. **Firewall Rules**: Only expose necessary ports
2. **Reverse Proxy**: Use nginx/traefik with SSL termination
3. **Internal Services**: Keep database, Redis, Kafka on private network
4. **VPN Access**: Require VPN for administrative access

### Monitoring

Monitor for:
- Failed authentication attempts
- Unusual API usage patterns
- Database connection errors
- High error rates
- Resource exhaustion

---

## ✅ Security Checklist

### Development

- [ ] `.env` file is in `.gitignore`
- [ ] No hardcoded secrets in code
- [ ] Using secure secret key (not placeholder)
- [ ] Default admin password changed
- [ ] Dependencies are up to date
- [ ] Security check script passes

### Production

- [ ] All secrets stored securely (not in code)
- [ ] HTTPS/TLS enabled
- [ ] Strong passwords for all services
- [ ] Database access restricted
- [ ] CORS configured for specific domains
- [ ] Rate limiting enabled
- [ ] Logging configured (no sensitive data)
- [ ] Backup and recovery tested
- [ ] Security monitoring enabled
- [ ] Incident response plan documented

---

## 🛠️ Security Tools

### Built-in Scripts

1. **Generate Secret Key**
   ```bash
   python scripts/generate_secret_key.py
   ```

2. **Security Check**
   ```bash
   python scripts/check_security.py
   ```

### External Tools

1. **Safety** - Dependency vulnerability scanner
   ```bash
   pip install safety
   safety check
   ```

2. **Bandit** - Python security linter
   ```bash
   pip install bandit
   bandit -r src/backend
   ```

3. **OWASP ZAP** - Web application security scanner

---

## 📞 Security Incident Response

If you discover a security vulnerability:

1. **DO NOT** create a public issue
2. **DO** report to: [security@example.com]
3. **DO** include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

---

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/secrets.html)

---

**Last Updated**: 2025-01-XX
**Maintained By**: i-Drill Security Team

