"""
Security utilities for i-Drill API
"""
import os
import secrets
import hashlib
import re
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def generate_secret_key(length: int = 32) -> str:
    """
    Generate a secure random secret key
    
    Args:
        length: Length of the secret key in bytes
        
    Returns:
        Hex-encoded secret key
    """
    return secrets.token_urlsafe(length)


def get_or_generate_secret_key() -> str:
    """
    Get SECRET_KEY from environment or generate a new one
    
    In production, this should always come from environment variables.
    In development, if not set, generates a temporary key with warning.
    
    Returns:
        Secret key string
    """
    secret_key = os.getenv("SECRET_KEY")
    
    if not secret_key:
        if os.getenv("APP_ENV", "development").lower() == "production":
            raise RuntimeError(
                "SECRET_KEY environment variable must be set in production. "
                "Generate one using: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        else:
            # Development mode - generate temporary key
            secret_key = generate_secret_key()
            logger.warning(
                f"⚠️ SECRET_KEY not set. Generated temporary key: {secret_key[:10]}... "
                "This key will change on restart. Set SECRET_KEY in .env for persistence."
            )
    
    # Validate secret key strength
    if len(secret_key) < 32:
        logger.warning(
            f"⚠️ SECRET_KEY is too short ({len(secret_key)} chars). "
            "Recommended minimum: 32 characters for production."
        )
    
    # Check for common insecure patterns
    insecure_patterns = [
        "your-secret-key",
        "change-in-production",
        "change_this",
        "change_this_to_a_secure",
        "secret",
        "password",
        "12345",
        "admin",
        "default",
        "test",
        "demo",
        "example"
    ]
    
    secret_key_lower = secret_key.lower()
    for pattern in insecure_patterns:
        if pattern in secret_key_lower:
            if os.getenv("APP_ENV", "development").lower() == "production":
                raise RuntimeError(
                    f"SECRET_KEY contains insecure pattern '{pattern}'. "
                    "Please use a secure random key in production."
                )
            else:
                logger.warning(
                    f"⚠️ SECRET_KEY contains potentially insecure pattern '{pattern}'. "
                    "Consider using a more secure key."
                )
    
    return secret_key


def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256 (for non-sensitive hashing)
    Note: For actual password hashing, use bcrypt via passlib
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def validate_cors_origins(origins: list[str]) -> list[str]:
    """
    Validate and sanitize CORS origins
    
    Args:
        origins: List of origin URLs
        
    Returns:
        Validated and sanitized list of origins
    """
    validated = []
    for origin in origins:
        origin = origin.strip()
        if not origin:
            continue
        
        # Basic validation
        if not (origin.startswith("http://") or origin.startswith("https://")):
            logger.warning(f"Invalid CORS origin format: {origin}")
            continue
        
        validated.append(origin)
    
    return validated


def get_rate_limit_config() -> dict:
    """
    Get rate limiting configuration from environment
    
    Returns:
        Dictionary with rate limit settings
    """
    default_limit = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
    enable_rate_limit = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"
    
    # Per-endpoint limits
    limits = {
        "default": default_limit,
        "auth": os.getenv("RATE_LIMIT_AUTH", "5/minute"),  # Stricter for auth
        "predictions": os.getenv("RATE_LIMIT_PREDICTIONS", "20/minute"),  # ML endpoints
        "sensor_data": os.getenv("RATE_LIMIT_SENSOR_DATA", "200/minute"),  # High volume
    }
    
    return {
        "enabled": enable_rate_limit,
        "limits": limits,
        "storage_url": os.getenv("RATE_LIMIT_STORAGE_URL", "memory://"),  # Use Redis in production
    }


def mask_sensitive_data(data: str, mask_char: str = "*") -> str:
    """
    Mask sensitive data in strings (passwords, tokens, etc.)
    
    Args:
        data: String containing potentially sensitive data
        mask_char: Character to use for masking
        
    Returns:
        Masked string
    """
    if not data or len(data) < 4:
        return mask_char * 4
    
    # Show first 2 and last 2 characters, mask the rest
    if len(data) <= 6:
        return mask_char * len(data)
    
    return data[:2] + mask_char * (len(data) - 4) + data[-2:]


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password strength
    
    Args:
        password: Password to validate
        
    Returns:
        (is_valid, list of issues)
    """
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if len(password) < 12:
        issues.append("Password should be at least 12 characters for production")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Password should contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        issues.append("Password should contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        issues.append("Password should contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password should contain at least one special character")
    
    # Check for common weak passwords
    common_passwords = [
        "password", "12345678", "admin123", "qwerty",
        "password123", "admin", "letmein", "welcome"
    ]
    if password.lower() in common_passwords:
        issues.append("Password is too common and easily guessable")
    
    return len(issues) == 0, issues


def get_csp_policy(is_production: bool = False, api_url: str = None) -> str:
    """
    Generate Content Security Policy (CSP) header value
    
    Args:
        is_production: Whether running in production mode
        api_url: API base URL for connect-src directive
        
    Returns:
        CSP policy string
    """
    # Get custom CSP from environment or use default
    custom_csp = os.getenv("CSP_POLICY")
    if custom_csp:
        return custom_csp
    
    # Default CSP policy
    if is_production:
        # Strict CSP for production
        connect_src = ["'self'"]
        
        # Add API URL and WebSocket support if provided
        if api_url:
            connect_src.append(api_url)
            ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://")
            connect_src.append(ws_url)
        
        csp_directives = [
            "default-src 'self'",
            "script-src 'self'",  # No unsafe-inline or unsafe-eval in production
            "style-src 'self' 'unsafe-inline'",  # Allow inline styles for React
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            f"connect-src {' '.join(connect_src)}",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "upgrade-insecure-requests",
        ]
    else:
        # More permissive CSP for development
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Allow for HMR
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https: http:",
            "font-src 'self' data:",
            "connect-src 'self' ws: wss: http: https:",
            "frame-ancestors 'self'",
        ]
    
    return "; ".join(csp_directives)


def get_security_headers(is_production: bool = False, api_url: str = None) -> dict:
    """
    Get security headers dictionary
    
    Args:
        is_production: Whether running in production mode
        api_url: API base URL for CSP
        
    Returns:
        Dictionary of security headers
    """
    headers = {
        # Prevent MIME type sniffing
        "X-Content-Type-Options": "nosniff",
        
        # Prevent clickjacking
        "X-Frame-Options": "DENY",
        
        # XSS Protection (legacy but still useful)
        "X-XSS-Protection": "1; mode=block",
        
        # Referrer policy
        "Referrer-Policy": "strict-origin-when-cross-origin",
        
        # Content Security Policy
        "Content-Security-Policy": get_csp_policy(is_production, api_url),
        
        # Permissions Policy (formerly Feature-Policy)
        "Permissions-Policy": (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        ),
    }
    
    # HSTS only in production with HTTPS
    if is_production and os.getenv("FORCE_HTTPS", "false").lower() == "true":
        max_age = int(os.getenv("HSTS_MAX_AGE", "31536000"))  # 1 year default
        include_subdomains = os.getenv("HSTS_INCLUDE_SUBDOMAINS", "true").lower() == "true"
        
        hsts_value = f"max-age={max_age}"
        if include_subdomains:
            hsts_value += "; includeSubDomains"
        
        preload = os.getenv("HSTS_PRELOAD", "false").lower() == "true"
        if preload:
            hsts_value += "; preload"
        
        headers["Strict-Transport-Security"] = hsts_value
    
    return headers