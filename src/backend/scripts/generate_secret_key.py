#!/usr/bin/env python3
"""
Generate a secure secret key for i-Drill API

Usage:
    python scripts/generate_secret_key.py [length]

Examples:
    python scripts/generate_secret_key.py        # Generate 32-byte key (recommended)
    python scripts/generate_secret_key.py 64     # Generate 64-byte key (extra secure)
"""
import secrets
import sys
import os

def generate_secret_key(length: int = 32) -> str:
    """
    Generate a secure random secret key using cryptographically secure random generator
    
    Args:
        length: Length of the secret key in bytes (default: 32, minimum: 32)
        
    Returns:
        URL-safe base64 encoded secret key
    """
    if length < 32:
        print("⚠️  Warning: Key length should be at least 32 bytes for security.", file=sys.stderr)
        length = 32
    
    return secrets.token_urlsafe(length)

def main():
    """Main function to generate and display secret key"""
    try:
        length = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    except ValueError:
        print("❌ Error: Length must be a number", file=sys.stderr)
        sys.exit(1)
    
    if length < 32:
        print("⚠️  Warning: Using minimum recommended length of 32 bytes", file=sys.stderr)
        length = 32
    
    key = generate_secret_key(length)
    
    print("=" * 70)
    print("🔐 SECURE SECRET KEY GENERATED")
    print("=" * 70)
    print(f"\nGenerated {length}-byte secret key:")
    print(f"\nSECRET_KEY={key}\n")
    print("=" * 70)
    print("\n📝 Next steps:")
    print("1. Copy the SECRET_KEY line above")
    print("2. Add it to your .env file (or config.env)")
    print("3. NEVER commit .env files to version control!")
    print("4. Keep this key secret and secure\n")
    
    # Check if .env exists and offer to update
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        print(f"💡 Tip: Your .env file exists at: {env_path}")
        print("   You can manually add the SECRET_KEY to it.\n")
    else:
        config_example = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env.example')
        if os.path.exists(config_example):
            print(f"💡 Tip: Copy config.env.example to .env and add the SECRET_KEY\n")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

