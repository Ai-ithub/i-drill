"""Optimize sensor_data table with partitioning and indexes

Revision ID: 002_optimize_sensor_data
Revises: 001_initial
Create Date: 2025-01-27

Optimizations:
- Monthly partitioning based on timestamp
- Optimized indexes for fast queries
- TimescaleDB support (if extension available)
- Composite indexes for common query patterns
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime, timedelta

# revision identifiers, used by Alembic.
revision: str = '002_optimize_sensor_data'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade database schema with optimizations:
    1. Create partitioned table structure
    2. Add optimized indexes
    3. Enable TimescaleDB if available
    """
    
    # Check if TimescaleDB extension is available
    conn = op.get_bind()
    
    # Try to enable TimescaleDB extension
    try:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        print("✅ TimescaleDB extension enabled")
        use_timescaledb = True
    except Exception as e:
        print(f"⚠️ TimescaleDB not available: {e}. Using standard PostgreSQL partitioning.")
        use_timescaledb = False
    
    # Check if sensor_data table exists and has data
    result = conn.execute(sa.text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'sensor_data'
        );
    """))
    table_exists = result.scalar()
    
    if not table_exists:
        # Create new partitioned table
        if use_timescaledb:
            # Create as TimescaleDB hypertable
            op.create_table(
                'sensor_data',
                sa.Column('id', sa.BigInteger(), nullable=False),
                sa.Column('rig_id', sa.String(length=50), nullable=False),
                sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
                sa.Column('depth', sa.Float(), nullable=False),
                sa.Column('wob', sa.Float(), nullable=False),
                sa.Column('rpm', sa.Float(), nullable=False),
                sa.Column('torque', sa.Float(), nullable=False),
                sa.Column('rop', sa.Float(), nullable=False),
                sa.Column('mud_flow', sa.Float(), nullable=False),
                sa.Column('mud_pressure', sa.Float(), nullable=False),
                sa.Column('mud_temperature', sa.Float(), nullable=True),
                sa.Column('gamma_ray', sa.Float(), nullable=True),
                sa.Column('resistivity', sa.Float(), nullable=True),
                sa.Column('density', sa.Float(), nullable=True),
                sa.Column('porosity', sa.Float(), nullable=True),
                sa.Column('hook_load', sa.Float(), nullable=True),
                sa.Column('vibration', sa.Float(), nullable=True),
                sa.Column('status', sa.String(length=20), server_default='normal', nullable=True),
                sa.PrimaryKeyConstraint('id', 'timestamp')
            )
            
            # Convert to hypertable
            conn.execute(sa.text("""
                SELECT create_hypertable('sensor_data', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 month',
                    if_not_exists => TRUE);
            """))
            print("✅ Created TimescaleDB hypertable")
        else:
            # Create partitioned table (monthly partitions)
            op.execute("""
                CREATE TABLE sensor_data (
                    id BIGSERIAL,
                    rig_id VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    depth FLOAT NOT NULL,
                    wob FLOAT NOT NULL,
                    rpm FLOAT NOT NULL,
                    torque FLOAT NOT NULL,
                    rop FLOAT NOT NULL,
                    mud_flow FLOAT NOT NULL,
                    mud_pressure FLOAT NOT NULL,
                    mud_temperature FLOAT,
                    gamma_ray FLOAT,
                    resistivity FLOAT,
                    density FLOAT,
                    porosity FLOAT,
                    hook_load FLOAT,
                    vibration FLOAT,
                    status VARCHAR(20) DEFAULT 'normal',
                    PRIMARY KEY (id, timestamp)
                ) PARTITION BY RANGE (timestamp);
            """)
            
            # Create partitions for current and next 3 months
            create_monthly_partitions(conn)
            print("✅ Created partitioned table with monthly partitions")
    else:
        # Table exists - migrate to partitioned structure
        print("⚠️ Table exists. Manual migration required for partitioning.")
        print("   Consider using pg_partman or manual migration script.")
    
    # Create optimized indexes
    create_optimized_indexes(conn, use_timescaledb)
    
    # Create function for automatic partition creation
    create_partition_management_functions(conn, use_timescaledb)


def create_monthly_partitions(conn):
    """Create monthly partitions for the next 12 months."""
    from datetime import datetime, timedelta
    
    base_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    for i in range(-1, 13):  # Create partitions for past month and next 12 months
        partition_date = base_date + timedelta(days=32 * i)
        partition_start = partition_date.replace(day=1)
        partition_end = (partition_start + timedelta(days=32)).replace(day=1)
        
        partition_name = f"sensor_data_{partition_start.strftime('%Y_%m')}"
        
        try:
            conn.execute(sa.text(f"""
                CREATE TABLE IF NOT EXISTS {partition_name} 
                PARTITION OF sensor_data
                FOR VALUES FROM ('{partition_start}') TO ('{partition_end}');
            """))
            print(f"✅ Created partition: {partition_name}")
        except Exception as e:
            print(f"⚠️ Partition {partition_name} may already exist: {e}")


def create_optimized_indexes(conn, use_timescaledb: bool):
    """Create optimized indexes for fast queries."""
    
    indexes = [
        # Composite index for common query: rig_id + timestamp
        ("ix_sensor_data_rig_timestamp", 
         "CREATE INDEX IF NOT EXISTS ix_sensor_data_rig_timestamp ON sensor_data (rig_id, timestamp DESC);"),
        
        # Index for timestamp-only queries
        ("ix_sensor_data_timestamp_desc",
         "CREATE INDEX IF NOT EXISTS ix_sensor_data_timestamp_desc ON sensor_data (timestamp DESC);"),
        
        # Index for rig_id queries
        ("ix_sensor_data_rig_id",
         "CREATE INDEX IF NOT EXISTS ix_sensor_data_rig_id ON sensor_data (rig_id);"),
        
        # Partial index for active status
        ("ix_sensor_data_status_active",
         "CREATE INDEX IF NOT EXISTS ix_sensor_data_status_active ON sensor_data (rig_id, timestamp) WHERE status != 'normal';"),
        
        # Index for depth-based queries
        ("ix_sensor_data_rig_depth",
         "CREATE INDEX IF NOT EXISTS ix_sensor_data_rig_depth ON sensor_data (rig_id, depth);"),
        
        # Index for time-range queries (useful for analytics)
        ("ix_sensor_data_timestamp_brin",
         "CREATE INDEX IF NOT EXISTS ix_sensor_data_timestamp_brin ON sensor_data USING BRIN (timestamp);"),
    ]
    
    for index_name, index_sql in indexes:
        try:
            conn.execute(sa.text(index_sql))
            print(f"✅ Created index: {index_name}")
        except Exception as e:
            print(f"⚠️ Error creating index {index_name}: {e}")


def create_partition_management_functions(conn, use_timescaledb: bool):
    """Create functions for automatic partition management."""
    
    if use_timescaledb:
        # TimescaleDB handles partitioning automatically
        return
    
    # Function to create next month's partition
    conn.execute(sa.text("""
        CREATE OR REPLACE FUNCTION create_next_sensor_data_partition()
        RETURNS void AS $$
        DECLARE
            next_month_start DATE;
            next_month_end DATE;
            partition_name TEXT;
        BEGIN
            -- Calculate next month
            next_month_start := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month');
            next_month_end := DATE_TRUNC('month', next_month_start + INTERVAL '1 month');
            partition_name := 'sensor_data_' || TO_CHAR(next_month_start, 'YYYY_MM');
            
            -- Create partition if it doesn't exist
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I 
                PARTITION OF sensor_data
                FOR VALUES FROM (%L) TO (%L);
            ', partition_name, next_month_start, next_month_end);
            
            RAISE NOTICE 'Created partition: %', partition_name;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    # Function to drop old partitions (older than retention period)
    conn.execute(sa.text("""
        CREATE OR REPLACE FUNCTION drop_old_sensor_data_partitions(retention_months INTEGER DEFAULT 12)
        RETURNS void AS $$
        DECLARE
            cutoff_date DATE;
            partition_record RECORD;
        BEGIN
            cutoff_date := DATE_TRUNC('month', CURRENT_DATE - (retention_months || ' months')::INTERVAL);
            
            FOR partition_record IN
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE 'sensor_data_%'
                AND tablename ~ '^sensor_data_[0-9]{4}_[0-9]{2}$'
            LOOP
                -- Extract date from partition name
                DECLARE
                    partition_date DATE;
                BEGIN
                    partition_date := TO_DATE(
                        SUBSTRING(partition_record.tablename FROM 'sensor_data_([0-9]{4}_[0-9]{2})$'),
                        'YYYY_MM'
                    );
                    
                    IF partition_date < cutoff_date THEN
                        EXECUTE format('DROP TABLE IF EXISTS %I CASCADE;', partition_record.tablename);
                        RAISE NOTICE 'Dropped old partition: %', partition_record.tablename;
                    END IF;
                EXCEPTION WHEN OTHERS THEN
                    -- Skip partitions with invalid names
                    CONTINUE;
                END;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    print("✅ Created partition management functions")


def downgrade() -> None:
    """Downgrade database schema - remove optimizations."""
    conn = op.get_bind()
    
    # Drop indexes
    indexes_to_drop = [
        'ix_sensor_data_rig_timestamp',
        'ix_sensor_data_timestamp_desc',
        'ix_sensor_data_rig_id',
        'ix_sensor_data_status_active',
        'ix_sensor_data_rig_depth',
        'ix_sensor_data_timestamp_brin',
    ]
    
    for index_name in indexes_to_drop:
        try:
            conn.execute(sa.text(f"DROP INDEX IF EXISTS {index_name};"))
        except Exception as e:
            print(f"⚠️ Error dropping index {index_name}: {e}")
    
    # Drop functions
    try:
        conn.execute(sa.text("DROP FUNCTION IF EXISTS create_next_sensor_data_partition();"))
        conn.execute(sa.text("DROP FUNCTION IF EXISTS drop_old_sensor_data_partitions(INTEGER);"))
    except Exception as e:
        print(f"⚠️ Error dropping functions: {e}")
    
    # Note: Partitioned table structure should be manually migrated back
    # This is a complex operation and should be done carefully
    print("⚠️ Manual migration required to convert partitioned table back to regular table")

