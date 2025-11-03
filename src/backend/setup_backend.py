#!/usr/bin/env python3
"""
Backend Setup Script
Initializes database, creates tables, and prepares the backend for first run
"""
import sys
import logging
from pathlib import Path

# Add src/backend to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database():
    """Initialize database and create tables"""
    try:
        logger.info("="*60)
        logger.info("Setting up database...")
        logger.info("="*60)
        
        from database import init_database, db_manager
        from api.models.database_models import Base
        
        # Initialize database connection
        logger.info("Connecting to database...")
        success = init_database(
            database_url=None,  # Uses environment variable or default
            create_tables=False,  # We'll create them manually for better control
            echo=True,  # Show SQL statements
        )
        
        if not success:
            logger.error("❌ Failed to initialize database connection")
            return False
        
        logger.info("✅ Database connection established")
        
        # Create all tables
        logger.info("Creating database tables...")
        try:
            Base.metadata.create_all(bind=db_manager.engine)
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Table creation warning: {e}")
            logger.info("(This is normal if tables already exist)")
        
        # Verify tables
        logger.info("Verifying tables...")
        inspector = db_manager.engine.dialect.get_table_names
        logger.info("✅ Database setup complete")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_data():
    """Create sample data for testing"""
    try:
        logger.info("="*60)
        logger.info("Creating sample data...")
        logger.info("="*60)
        
        from database import db_manager
        from api.models.database_models import UserDB, WellProfileDB
        from datetime import datetime
        
        with db_manager.session_scope() as session:
            # Check if sample user exists
            existing_user = session.query(UserDB).filter(
                UserDB.username == "admin"
            ).first()
            
            if not existing_user:
                # Create admin user
                admin_user = UserDB(
                    username="admin",
                    email="admin@idrill.com",
                    hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5ztJ.Yj5q.G7e",  # "admin123"
                    full_name="System Administrator",
                    role="admin",
                    is_active=True
                )
                session.add(admin_user)
                logger.info("✅ Created admin user (username: admin, password: admin123)")
            else:
                logger.info("ℹ️ Admin user already exists")
            
            # Check if sample well profile exists
            existing_well = session.query(WellProfileDB).filter(
                WellProfileDB.well_id == "WELL_01"
            ).first()
            
            if not existing_well:
                # Create sample well profile
                well_profile = WellProfileDB(
                    well_id="WELL_01",
                    rig_id="RIG_01",
                    total_depth=12000.0,
                    kick_off_point=2000.0,
                    build_rate=2.5,
                    max_inclination=45.0,
                    target_zone_start=8000.0,
                    target_zone_end=12000.0
                )
                session.add(well_profile)
                logger.info("✅ Created sample well profile (WELL_01)")
            else:
                logger.info("ℹ️ Sample well profile already exists")
        
        logger.info("✅ Sample data creation complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample data creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_services():
    """Verify that required services are accessible"""
    logger.info("="*60)
    logger.info("Verifying services...")
    logger.info("="*60)
    
    # Check Kafka
    try:
        from services.kafka_service import kafka_service
        kafka_healthy = kafka_service.check_connection()
        if kafka_healthy:
            logger.info("✅ Kafka connection: OK")
        else:
            logger.warning("⚠️ Kafka connection: FAILED (optional service)")
    except Exception as e:
        logger.warning(f"⚠️ Kafka check failed: {e} (optional service)")
    
    # Check Database
    try:
        from database import check_database_health
        db_health = check_database_health()
        if db_health.get("database") == "healthy":
            logger.info("✅ Database connection: OK")
        else:
            logger.error("❌ Database connection: FAILED")
            return False
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False
    
    logger.info("✅ Service verification complete")
    return True


def main():
    """Main setup function"""
    logger.info("="*60)
    logger.info("i-Drill Backend Setup")
    logger.info("="*60)
    
    # Step 1: Setup database
    if not setup_database():
        logger.error("Setup failed at database initialization")
        return False
    
    # Step 2: Create sample data
    try:
        if not create_sample_data():
            logger.warning("Sample data creation failed (optional)")
    except Exception as e:
        logger.warning(f"Sample data creation failed: {e} (optional)")
    
    # Step 3: Verify services
    if not verify_services():
        logger.warning("Some services are not available")
    
    logger.info("="*60)
    logger.info("✨ Backend setup complete!")
    logger.info("="*60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start the backend server:")
    logger.info("   python app.py")
    logger.info("   or")
    logger.info("   uvicorn app:app --host 0.0.0.0 --port 8001 --reload")
    logger.info("")
    logger.info("2. Access API documentation:")
    logger.info("   http://localhost:8001/docs")
    logger.info("")
    logger.info("3. Test health endpoint:")
    logger.info("   http://localhost:8001/health")
    logger.info("")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

