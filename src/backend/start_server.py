#!/usr/bin/env python3
"""
Simple server starter with error handling
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    import uvicorn
    from app import app
    
    print("=" * 60)
    print("Starting i-Drill Backend Server...")
    print("=" * 60)
    print("Port: 8001")
    print("API Docs: http://localhost:8001/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    
except Exception as e:
    print(f"ERROR: Failed to start server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

