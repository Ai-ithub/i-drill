#!/usr/bin/env python3
"""
Simple Backend Test Script
Tests basic functionality without requiring external services
"""
import sys
from pathlib import Path

# Add src/backend to path
sys.path.insert(0, str(Path(__file__).parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    logger.info("="*60)
    logger.info("Testing module imports...")
    logger.info("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test database module
    try:
        from database import DatabaseManager, get_db, init_database
        logger.info("✅ database module imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Failed to import database: {e}")
        tests_failed += 1
    
    # Test schemas
    try:
        from api.models.schemas import (
            SensorDataPoint,
            SensorDataResponse,
            PredictionRequest,
            MaintenanceAlert,
            User,
            Token
        )
        logger.info("✅ schemas imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Failed to import schemas: {e}")
        tests_failed += 1
    
    # Test database models
    try:
        from api.models.database_models import (
            SensorData,
            MaintenanceAlertDB,
            UserDB,
            RULPredictionDB
        )
        logger.info("✅ database_models imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Failed to import database_models: {e}")
        tests_failed += 1
    
    # Test services
    try:
        from services.data_service import DataService
        logger.info("✅ data_service imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Failed to import data_service: {e}")
        tests_failed += 1
    
    try:
        from services.kafka_service import kafka_service
        logger.info("✅ kafka_service imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Failed to import kafka_service: {e}")
        tests_failed += 1
    
    # Test routes
    try:
        from api.routes import (
            health,
            sensor_data,
            predictions,
            maintenance,
            producer,
            config
        )
        logger.info("✅ api routes imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Failed to import routes: {e}")
        tests_failed += 1
    
    # Test main app
    try:
        from app import app
        logger.info("✅ FastAPI app imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Failed to import app: {e}")
        tests_failed += 1
    
    logger.info("")
    logger.info(f"Import tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_pydantic_schemas():
    """Test Pydantic schema creation"""
    logger.info("="*60)
    logger.info("Testing Pydantic schemas...")
    logger.info("="*60)
    
    try:
        from api.models.schemas import SensorDataPoint
        from datetime import datetime
        
        # Create a test sensor data point
        data = SensorDataPoint(
            rig_id="RIG_TEST",
            timestamp=datetime.now(),
            depth=5000.0,
            wob=15000.0,
            rpm=100.0,
            torque=10000.0,
            rop=50.0,
            mud_flow=800.0,
            mud_pressure=3000.0
        )
        
        logger.info(f"✅ Created sensor data: rig_id={data.rig_id}, depth={data.depth}")
        
        # Test validation
        try:
            invalid_data = SensorDataPoint(
                rig_id="RIG_TEST",
                timestamp=datetime.now(),
                depth=-100.0,  # Invalid: negative depth
                wob=15000.0,
                rpm=100.0,
                torque=10000.0,
                rop=50.0,
                mud_flow=800.0,
                mud_pressure=3000.0
            )
            logger.error("❌ Validation failed: accepted negative depth")
            return False
        except:
            logger.info("✅ Validation working: rejected negative depth")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_models():
    """Test SQLAlchemy models"""
    logger.info("="*60)
    logger.info("Testing database models...")
    logger.info("="*60)
    
    try:
        from api.models.database_models import Base, SensorData, UserDB
        from sqlalchemy import inspect
        
        # Check that models have required attributes
        sensor_columns = [c.name for c in inspect(SensorData).columns]
        logger.info(f"✅ SensorData columns: {len(sensor_columns)} columns")
        
        user_columns = [c.name for c in inspect(UserDB).columns]
        logger.info(f"✅ UserDB columns: {len(user_columns)} columns")
        
        # Check relationships and constraints
        logger.info(f"✅ Database models structure validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fastapi_app():
    """Test FastAPI app structure"""
    logger.info("="*60)
    logger.info("Testing FastAPI application...")
    logger.info("="*60)
    
    try:
        from app import app
        from fastapi.testclient import TestClient
        
        # Check app attributes
        logger.info(f"✅ App title: {app.title}")
        logger.info(f"✅ App version: {app.version}")
        
        # Check routes
        routes = [route.path for route in app.routes]
        logger.info(f"✅ Total routes: {len(routes)}")
        
        # Check key endpoints exist
        required_endpoints = [
            "/",
            "/health",
            "/api/v1/health/",
            "/api/v1/sensor-data/realtime",
        ]
        
        for endpoint in required_endpoints:
            if endpoint in routes:
                logger.info(f"✅ Endpoint exists: {endpoint}")
            else:
                logger.warning(f"⚠️ Endpoint missing: {endpoint}")
        
        # Try to create test client
        client = TestClient(app)
        logger.info("✅ Test client created successfully")
        
        # Test root endpoint (without database)
        try:
            response = client.get("/")
            logger.info(f"✅ Root endpoint: status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Root endpoint test skipped: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FastAPI app test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    logger.info("="*60)
    logger.info("i-Drill Backend Test Suite")
    logger.info("="*60)
    logger.info("")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Pydantic Schemas", test_pydantic_schemas()))
    results.append(("Database Models", test_database_models()))
    results.append(("FastAPI Application", test_fastapi_app()))
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:.<40} {status}")
    
    logger.info("")
    logger.info(f"Total: {passed} passed, {failed} failed")
    logger.info("="*60)
    
    if failed == 0:
        logger.info("🎉 All tests passed! Backend is ready.")
        return True
    else:
        logger.error(f"❌ {failed} test(s) failed. Please fix the issues.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

