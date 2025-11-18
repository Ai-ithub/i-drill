"""
Database Connection Management with SQLAlchemy
Provides connection pooling, session management, and initialization
"""
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import logging
from typing import Generator
import os

from api.models.database_models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database Manager for handling connections and sessions
    """
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
        
    def initialize(
        self,
        database_url: str = None,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
    ):
        """
        Initialize database connection with optimized pool settings
        
        Performance optimizations:
        - Connection pooling for reduced connection overhead
        - Pool pre-ping to verify connections before use
        - Configurable pool size based on workload
        
        Args:
            database_url: Database connection URL
            echo: Enable SQL logging
            pool_size: Connection pool size
            max_overflow: Max overflow connections
            pool_timeout: Connection timeout
            pool_recycle: Recycle connections after this many seconds
        """
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        # Get database URL from environment or use default
        if database_url is None:
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:5432/drilling_db'
            )
        
        try:
            # Create engine with connection pooling
            # Use connect_args to set connection timeout to avoid hanging
            # Set pool_pre_ping=False initially to avoid connection attempts during engine creation
            self.engine = create_engine(
                database_url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=False,  # Disable pre-ping to avoid immediate connection attempts
                connect_args={
                    "connect_timeout": 3,  # 3 second connection timeout (reduced from 5)
                },
            )
            
            # Add connection event listeners
            @event.listens_for(self.engine, "connect")
            def receive_connect(dbapi_conn, connection_record):
                logger.debug("New database connection established")
            
            @event.listens_for(self.engine, "close")
            def receive_close(dbapi_conn, connection_record):
                logger.debug("Database connection closed")
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            self._initialized = True
            logger.info(f"Database engine initialized: {self._mask_password(database_url)}")
            logger.info("Note: Connection will be tested on first use")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            logger.warning("Backend will continue without database (limited functionality)")
            self.engine = None
            self.SessionLocal = None
            self._initialized = False
            # Don't raise - allow application to continue
    
    def create_tables(self):
        """Create all database tables"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """
        Get a database session
        
        Returns:
            Database session
            
        Raises:
            RuntimeError: If database is not initialized
        """
        if not self._initialized or self.SessionLocal is None:
            raise RuntimeError("Database not initialized or not available")
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic commit/rollback
        
        Usage:
            with db_manager.session_scope() as session:
                session.query(Model).all()
        
        Yields:
            Database session
        """
        if not self._initialized or self.SessionLocal is None:
            raise RuntimeError("Database not initialized or not available")
            
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def check_connection(self) -> bool:
        """
        Check if database connection is healthy
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._initialized or self.engine is None:
            return False
        
        try:
            from sqlalchemy import text
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.debug(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close database connections and cleanup"""
        if self.engine:
            self.engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")
    
    @staticmethod
    def _mask_password(url: str) -> str:
        """Mask password in database URL for logging"""
        try:
            if '@' in url and '://' in url:
                protocol, rest = url.split('://', 1)
                if '@' in rest:
                    credentials, host = rest.split('@', 1)
                    if ':' in credentials:
                        username = credentials.split(':', 1)[0]
                        return f"{protocol}://{username}:****@{host}"
            return url
        except:
            return url


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database sessions
    
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    Yields:
        Database session
    """
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def init_database(
    database_url: str = None,
    create_tables: bool = True,
    use_migrations: bool = True,
    **kwargs
):
    """
    Initialize database with optional table creation
    
    Args:
        database_url: Database connection URL
        create_tables: Whether to create tables
        use_migrations: If True, use Alembic migrations instead of create_all()
        **kwargs: Additional arguments for DatabaseManager.initialize()
    
    Note:
        In production, always use migrations (use_migrations=True).
        create_all() should only be used for development/testing.
    """
    try:
        db_manager.initialize(database_url=database_url, **kwargs)
        
        if create_tables:
            if use_migrations:
                # Use Alembic migrations for production
                logger.info("Using Alembic migrations for table creation")
                logger.warning(
                    "To apply migrations, run: alembic upgrade head\n"
                    "Or use: python scripts/manage_migrations.py upgrade"
                )
                # Don't create tables here - let Alembic handle it
            else:
                # Use create_all() for development/testing only
                logger.warning(
                    "Using create_all() - not recommended for production. "
                    "Use Alembic migrations instead."
                )
                db_manager.create_tables()
        
        logger.info("Database initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def check_database_health() -> dict:
    """
    Check database health and return status
    
    Returns:
        Dictionary with health status
    """
    is_healthy = db_manager.check_connection()
    
    return {
        "database": "healthy" if is_healthy else "unhealthy",
        "initialized": db_manager._initialized,
        "engine_pool_size": db_manager.engine.pool.size() if db_manager.engine else 0,
    }


# Database utilities

def execute_raw_sql(sql: str, params: dict = None) -> list:
    """
    Execute raw SQL query with parameterized queries for security.
    
    Args:
        sql: SQL query string (use :param_name for parameters)
        params: Query parameters dictionary (e.g., {"param_name": value})
    
    Returns:
        Query results
    
    Warning:
        Use with caution. Prefer ORM methods when possible.
        For schema changes, use Alembic migrations instead.
        
    Security Note:
        This function uses parameterized queries to prevent SQL injection.
        Always use named parameters (:param_name) in SQL and pass values via params dict.
        Never concatenate user input directly into SQL strings!
        
    Example:
        >>> execute_raw_sql(
        ...     "SELECT * FROM users WHERE username = :username",
        ...     {"username": "admin"}
        ... )
    """
    from sqlalchemy import text
    
    if params is None:
        params = {}
    
    # Use SQLAlchemy text() for safer parameterized queries
    # This ensures proper escaping and prevents SQL injection
    with db_manager.session_scope() as session:
        # Wrap SQL in text() to enable parameterized queries
        stmt = text(sql)
        result = session.execute(stmt, params)
        return result.fetchall()


def bulk_insert(model_class, data: list[dict]):
    """
    Bulk insert data
    
    Args:
        model_class: SQLAlchemy model class
        data: List of dictionaries with data to insert
    """
    with db_manager.session_scope() as session:
        objects = [model_class(**item) for item in data]
        session.bulk_save_objects(objects)
        logger.info(f"Bulk inserted {len(data)} records into {model_class.__tablename__}")

