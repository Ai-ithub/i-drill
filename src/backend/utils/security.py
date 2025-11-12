"""
Security utilities for i-Drill API
"""
import os
import secrets
import hashlib
from typing import Optional
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
        "secret",
        "password",
        "12345",
        "admin"
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

