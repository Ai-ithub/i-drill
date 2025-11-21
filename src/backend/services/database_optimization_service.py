"""
Database Optimization Service
Manages PostgreSQL optimizations: partitioning, indexing, connection pooling, TimescaleDB
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os

try:
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.pool import QueuePool
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from config_loader import config_loader
from database_manager import db_manager

logger = logging.getLogger(__name__)


class DatabaseOptimizationService:
    """
    Service for database optimizations.
    
    Features:
    - Partition management
    - Index optimization
    - Connection pooling configuration
    - TimescaleDB support
    - Query optimization
    """
    
    def __init__(self):
        """Initialize DatabaseOptimizationService."""
        self.available = SQLALCHEMY_AVAILABLE and db_manager.is_available()
        if not self.available:
            logger.warning("Database optimization service not available")
            return
        
        try:
            db_config = config_loader.get_database_config()
            self.database_url = self._build_database_url(db_config)
            self.engine = None
            self.timescaledb_enabled = False
            self._initialize_engine()
            self._check_timescaledb()
            
            logger.info("Database optimization service initialized")
        except Exception as e:
            logger.error(f"Error initializing database optimization service: {e}")
            self.available = False
    
    def _build_database_url(self, db_config: Dict[str, Any]) -> str:
        """Build database URL from config."""
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        database = db_config.get('database', 'drilling_db')
        username = db_config.get('username', 'drill_user')
        password = db_config.get('password', '')
        
        # Support for PgBouncer (connection pooling proxy)
        use_pgbouncer = os.getenv('USE_PGBOUNCER', 'false').lower() == 'true'
        if use_pgbouncer:
            pgbouncer_port = os.getenv('PGBOUNCER_PORT', '6432')
            port = pgbouncer_port
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with optimized connection pooling."""
        try:
            pool_size = int(os.getenv('DB_POOL_SIZE', '20'))
            max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '40'))
            pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
            pool_recycle = int(os.getenv('DB_POOL_RECYCLE', '3600'))
            
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,  # Verify connections before using
                echo=False,
                connect_args={
                    'connect_timeout': 10,
                    'application_name': 'i-drill-optimization'
                }
            )
            
            logger.info(
                f"Database engine initialized with pool_size={pool_size}, "
                f"max_overflow={max_overflow}"
            )
        except Exception as e:
            logger.error(f"Error initializing database engine: {e}")
            self.engine = None
    
    def _check_timescaledb(self) -> bool:
        """Check if TimescaleDB extension is available."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                    );
                """))
                self.timescaledb_enabled = result.scalar()
                
                if self.timescaledb_enabled:
                    logger.info("✅ TimescaleDB extension is enabled")
                else:
                    logger.info("ℹ️ TimescaleDB extension not available")
                
                return self.timescaledb_enabled
        except Exception as e:
            logger.warning(f"Error checking TimescaleDB: {e}")
            return False
    
    def enable_timescaledb(self) -> Dict[str, Any]:
        """Enable TimescaleDB extension."""
        if not self.engine:
            return {"success": False, "message": "Database engine not available"}
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                conn.commit()
                self.timescaledb_enabled = True
                logger.info("✅ TimescaleDB extension enabled")
                return {
                    "success": True,
                    "message": "TimescaleDB extension enabled"
                }
        except Exception as e:
            logger.error(f"Error enabling TimescaleDB: {e}")
            return {
                "success": False,
                "message": f"Failed to enable TimescaleDB: {e}",
                "error": str(e)
            }
    
    def convert_to_hypertable(self, table_name: str = "sensor_data", time_column: str = "timestamp") -> Dict[str, Any]:
        """Convert table to TimescaleDB hypertable."""
        if not self.timescaledb_enabled:
            return {
                "success": False,
                "message": "TimescaleDB extension not enabled"
            }
        
        if not self.engine:
            return {"success": False, "message": "Database engine not available"}
        
        try:
            with self.engine.connect() as conn:
                # Check if already a hypertable
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT 1 FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = '{table_name}'
                    );
                """))
                is_hypertable = result.scalar()
                
                if is_hypertable:
                    return {
                        "success": True,
                        "message": f"Table {table_name} is already a hypertable"
                    }
                
                # Convert to hypertable
                conn.execute(text(f"""
                    SELECT create_hypertable(
                        '{table_name}',
                        '{time_column}',
                        chunk_time_interval => INTERVAL '1 month',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
                
                logger.info(f"✅ Converted {table_name} to hypertable")
                return {
                    "success": True,
                    "message": f"Table {table_name} converted to hypertable"
                }
        except Exception as e:
            logger.error(f"Error converting to hypertable: {e}")
            return {
                "success": False,
                "message": f"Failed to convert to hypertable: {e}",
                "error": str(e)
            }
    
    def enable_compression(self, table_name: str = "sensor_data", compress_after: str = "30 days") -> Dict[str, Any]:
        """Enable compression for old data in TimescaleDB."""
        if not self.timescaledb_enabled:
            return {
                "success": False,
                "message": "TimescaleDB extension not enabled"
            }
        
        if not self.engine:
            return {"success": False, "message": "Database engine not available"}
        
        try:
            with self.engine.connect() as conn:
                # Add compression policy
                conn.execute(text(f"""
                    SELECT add_compression_policy(
                        '{table_name}',
                        INTERVAL '{compress_after}',
                        if_not_exists => TRUE
                    );
                """))
                conn.commit()
                
                logger.info(f"✅ Compression enabled for {table_name}")
                return {
                    "success": True,
                    "message": f"Compression enabled for {table_name}"
                }
        except Exception as e:
            logger.error(f"Error enabling compression: {e}")
            return {
                "success": False,
                "message": f"Failed to enable compression: {e}",
                "error": str(e)
            }
    
    def create_partition(self, partition_date: datetime) -> Dict[str, Any]:
        """Create a monthly partition for sensor_data table."""
        if not self.engine:
            return {"success": False, "message": "Database engine not available"}
        
        if self.timescaledb_enabled:
            # TimescaleDB handles partitions automatically
            return {
                "success": True,
                "message": "TimescaleDB handles partitioning automatically"
            }
        
        try:
            partition_start = partition_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            partition_end = (partition_start + timedelta(days=32)).replace(day=1)
            partition_name = f"sensor_data_{partition_start.strftime('%Y_%m')}"
            
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {partition_name} 
                    PARTITION OF sensor_data
                    FOR VALUES FROM ('{partition_start}') TO ('{partition_end}');
                """))
                conn.commit()
                
                logger.info(f"✅ Created partition: {partition_name}")
                return {
                    "success": True,
                    "message": f"Partition {partition_name} created",
                    "partition": partition_name
                }
        except Exception as e:
            logger.error(f"Error creating partition: {e}")
            return {
                "success": False,
                "message": f"Failed to create partition: {e}",
                "error": str(e)
            }
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get index usage statistics."""
        if not self.engine:
            return {}
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan as index_scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                    AND tablename = 'sensor_data'
                    ORDER BY idx_scan DESC;
                """))
                
                indexes = []
                for row in result:
                    indexes.append({
                        "table": row.tablename,
                        "index": row.indexname,
                        "scans": row.index_scans,
                        "tuples_read": row.tuples_read,
                        "tuples_fetched": row.tuples_fetched
                    })
                
                return {"indexes": indexes}
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            return {}
    
    def analyze_table(self, table_name: str = "sensor_data") -> Dict[str, Any]:
        """Run ANALYZE on table to update statistics."""
        if not self.engine:
            return {"success": False, "message": "Database engine not available"}
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"ANALYZE {table_name};"))
                conn.commit()
                
                logger.info(f"✅ Analyzed table: {table_name}")
                return {
                    "success": True,
                    "message": f"Table {table_name} analyzed"
                }
        except Exception as e:
            logger.error(f"Error analyzing table: {e}")
            return {
                "success": False,
                "message": f"Failed to analyze table: {e}",
                "error": str(e)
            }
    
    def get_table_size(self, table_name: str = "sensor_data") -> Dict[str, Any]:
        """Get table size information."""
        if not self.engine:
            return {}
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT
                        pg_size_pretty(pg_total_relation_size('{table_name}')) as total_size,
                        pg_size_pretty(pg_relation_size('{table_name}')) as table_size,
                        pg_size_pretty(pg_total_relation_size('{table_name}') - pg_relation_size('{table_name}')) as indexes_size,
                        (SELECT COUNT(*) FROM {table_name}) as row_count;
                """))
                
                row = result.fetchone()
                if row:
                    return {
                        "table": table_name,
                        "total_size": row.total_size,
                        "table_size": row.table_size,
                        "indexes_size": row.indexes_size,
                        "row_count": row.row_count
                    }
                return {}
        except Exception as e:
            logger.error(f"Error getting table size: {e}")
            return {}
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self.engine:
            return {}
        
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }


# Global instance
database_optimization_service = DatabaseOptimizationService()

