#!/usr/bin/env python3
"""
Generate a secure secret key for i-Drill API
"""
import secrets
import sys

def generate_secret_key(length: int = 32) -> str:
    """Generate a secure random secret key"""
    return secrets.token_urlsafe(length)

if __name__ == "__main__":
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    key = generate_secret_key(length)
    print(f"SECRET_KEY={key}")
    print(f"\nAdd this to your .env file:")
    print(f"SECRET_KEY={key}")

