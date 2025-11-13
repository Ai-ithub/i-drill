"""Initial database schema

Revision ID: 001_initial
Revises: 
Create Date: 2025-01-XX

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create sensor_data table
    op.create_table(
        'sensor_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
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
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sensor_data_id'), 'sensor_data', ['id'], unique=False)
    op.create_index(op.f('ix_sensor_data_rig_id'), 'sensor_data', ['rig_id'], unique=False)
    op.create_index(op.f('ix_sensor_data_timestamp'), 'sensor_data', ['timestamp'], unique=False)

    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('role', sa.String(length=20), server_default='viewer', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), server_default='0', nullable=True),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create maintenance_alerts table
    op.create_table(
        'maintenance_alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('alert_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('predicted_failure_time', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('acknowledged', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('acknowledged_by', sa.String(length=100), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
        sa.Column('acknowledgement_notes', sa.Text(), nullable=True),
        sa.Column('resolved', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('resolved_by', sa.String(length=100), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('dvr_history_id', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_maintenance_alerts_id'), 'maintenance_alerts', ['id'], unique=False)
    op.create_index(op.f('ix_maintenance_alerts_rig_id'), 'maintenance_alerts', ['rig_id'], unique=False)
    op.create_index(op.f('ix_maintenance_alerts_severity'), 'maintenance_alerts', ['severity'], unique=False)
    op.create_index(op.f('ix_maintenance_alerts_created_at'), 'maintenance_alerts', ['created_at'], unique=False)

    # Create maintenance_schedules table
    op.create_table(
        'maintenance_schedules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('maintenance_type', sa.String(length=50), nullable=False),
        sa.Column('scheduled_date', sa.DateTime(), nullable=False),
        sa.Column('estimated_duration_hours', sa.Float(), nullable=False),
        sa.Column('priority', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='scheduled', nullable=True),
        sa.Column('assigned_to', sa.String(length=100), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_maintenance_schedules_id'), 'maintenance_schedules', ['id'], unique=False)
    op.create_index(op.f('ix_maintenance_schedules_rig_id'), 'maintenance_schedules', ['rig_id'], unique=False)
    op.create_index(op.f('ix_maintenance_schedules_scheduled_date'), 'maintenance_schedules', ['scheduled_date'], unique=False)
    op.create_index(op.f('ix_maintenance_schedules_status'), 'maintenance_schedules', ['status'], unique=False)

    # Create password_reset_tokens table
    op.create_table(
        'password_reset_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('token', sa.String(length=255), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )
    op.create_index(op.f('ix_password_reset_tokens_id'), 'password_reset_tokens', ['id'], unique=False)
    op.create_index(op.f('ix_password_reset_tokens_user_id'), 'password_reset_tokens', ['user_id'], unique=False)
    op.create_index(op.f('ix_password_reset_tokens_token'), 'password_reset_tokens', ['token'], unique=True)
    op.create_index(op.f('ix_password_reset_tokens_expires_at'), 'password_reset_tokens', ['expires_at'], unique=False)

    # Create blacklisted_tokens table
    op.create_table(
        'blacklisted_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('token', sa.String(length=500), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('reason', sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )
    op.create_index(op.f('ix_blacklisted_tokens_id'), 'blacklisted_tokens', ['id'], unique=False)
    op.create_index(op.f('ix_blacklisted_tokens_token'), 'blacklisted_tokens', ['token'], unique=True)
    op.create_index(op.f('ix_blacklisted_tokens_expires_at'), 'blacklisted_tokens', ['expires_at'], unique=False)
    op.create_index(op.f('ix_blacklisted_tokens_user_id'), 'blacklisted_tokens', ['user_id'], unique=False)

    # Create login_attempts table
    op.create_table(
        'login_attempts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('success', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('attempted_at', sa.DateTime(), nullable=True),
        sa.Column('user_agent', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_login_attempts_id'), 'login_attempts', ['id'], unique=False)
    op.create_index(op.f('ix_login_attempts_username'), 'login_attempts', ['username'], unique=False)
    op.create_index(op.f('ix_login_attempts_attempted_at'), 'login_attempts', ['attempted_at'], unique=False)
    op.create_index(op.f('ix_login_attempts_ip_address'), 'login_attempts', ['ip_address'], unique=False)

    # Create change_requests table
    op.create_table(
        'change_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('change_type', sa.String(length=20), nullable=False),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('parameter', sa.String(length=100), nullable=False),
        sa.Column('old_value', sa.Text(), nullable=True),
        sa.Column('new_value', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='pending', nullable=True),
        sa.Column('auto_execute', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('requested_by', sa.Integer(), nullable=True),
        sa.Column('approved_by', sa.Integer(), nullable=True),
        sa.Column('applied_by', sa.Integer(), nullable=True),
        sa.Column('requested_at', sa.DateTime(), nullable=True),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('applied_at', sa.DateTime(), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['requested_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['approved_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['applied_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_change_requests_id'), 'change_requests', ['id'], unique=False)
    op.create_index(op.f('ix_change_requests_rig_id'), 'change_requests', ['rig_id'], unique=False)
    op.create_index(op.f('ix_change_requests_change_type'), 'change_requests', ['change_type'], unique=False)
    op.create_index(op.f('ix_change_requests_status'), 'change_requests', ['status'], unique=False)
    op.create_index(op.f('ix_change_requests_requested_at'), 'change_requests', ['requested_at'], unique=False)
    op.create_index(op.f('ix_change_requests_requested_by'), 'change_requests', ['requested_by'], unique=False)

    # Create dvr_process_history table
    op.create_table(
        'dvr_process_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=True),
        sa.Column('raw_record', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('reconciled_record', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_valid', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('anomaly_flag', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('anomaly_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.String(length=20), server_default='processed', nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_dvr_process_history_id'), 'dvr_process_history', ['id'], unique=False)
    op.create_index(op.f('ix_dvr_process_history_rig_id'), 'dvr_process_history', ['rig_id'], unique=False)
    op.create_index(op.f('ix_dvr_process_history_status'), 'dvr_process_history', ['status'], unique=False)
    op.create_index(op.f('ix_dvr_process_history_created_at'), 'dvr_process_history', ['created_at'], unique=False)

    # Add foreign key constraint for maintenance_alerts.dvr_history_id
    op.create_foreign_key(
        'maintenance_alerts_dvr_history_id_fkey',
        'maintenance_alerts', 'dvr_process_history',
        ['dvr_history_id'], ['id'],
        ondelete='SET NULL'
    )

    # Create rul_predictions table
    op.create_table(
        'rul_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('predicted_rul', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('model_used', sa.String(length=50), nullable=False),
        sa.Column('recommendation', sa.Text(), nullable=True),
        sa.Column('actual_failure_time', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_rul_predictions_id'), 'rul_predictions', ['id'], unique=False)
    op.create_index(op.f('ix_rul_predictions_rig_id'), 'rul_predictions', ['rig_id'], unique=False)
    op.create_index(op.f('ix_rul_predictions_timestamp'), 'rul_predictions', ['timestamp'], unique=False)

    # Create anomaly_detections table
    op.create_table(
        'anomaly_detections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('is_anomaly', sa.Boolean(), nullable=False),
        sa.Column('anomaly_score', sa.Float(), nullable=False),
        sa.Column('affected_parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('investigated', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('investigation_notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_anomaly_detections_id'), 'anomaly_detections', ['id'], unique=False)
    op.create_index(op.f('ix_anomaly_detections_rig_id'), 'anomaly_detections', ['rig_id'], unique=False)
    op.create_index(op.f('ix_anomaly_detections_timestamp'), 'anomaly_detections', ['timestamp'], unique=False)

    # Create model_versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('file_path', sa.String(length=255), nullable=False),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('training_date', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_versions_id'), 'model_versions', ['id'], unique=False)
    op.create_index(op.f('ix_model_versions_model_name'), 'model_versions', ['model_name'], unique=False)

    # Create well_profiles table
    op.create_table(
        'well_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('well_id', sa.String(length=50), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('total_depth', sa.Float(), nullable=False),
        sa.Column('kick_off_point', sa.Float(), nullable=False),
        sa.Column('build_rate', sa.Float(), nullable=False),
        sa.Column('max_inclination', sa.Float(), nullable=False),
        sa.Column('target_zone_start', sa.Float(), nullable=False),
        sa.Column('target_zone_end', sa.Float(), nullable=False),
        sa.Column('geological_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('well_id')
    )
    op.create_index(op.f('ix_well_profiles_id'), 'well_profiles', ['id'], unique=False)
    op.create_index(op.f('ix_well_profiles_well_id'), 'well_profiles', ['well_id'], unique=True)
    op.create_index(op.f('ix_well_profiles_rig_id'), 'well_profiles', ['rig_id'], unique=False)

    # Create drilling_sessions table
    op.create_table(
        'drilling_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('well_id', sa.String(length=50), nullable=False),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('start_depth', sa.Float(), nullable=False),
        sa.Column('end_depth', sa.Float(), nullable=True),
        sa.Column('average_rop', sa.Float(), nullable=True),
        sa.Column('total_drilling_time_hours', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=20), server_default='active', nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_drilling_sessions_id'), 'drilling_sessions', ['id'], unique=False)
    op.create_index(op.f('ix_drilling_sessions_rig_id'), 'drilling_sessions', ['rig_id'], unique=False)
    op.create_index(op.f('ix_drilling_sessions_well_id'), 'drilling_sessions', ['well_id'], unique=False)

    # Create system_logs table
    op.create_table(
        'system_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('level', sa.String(length=20), nullable=False),
        sa.Column('service', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_system_logs_id'), 'system_logs', ['id'], unique=False)
    op.create_index(op.f('ix_system_logs_timestamp'), 'system_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_system_logs_level'), 'system_logs', ['level'], unique=False)
    op.create_index(op.f('ix_system_logs_service'), 'system_logs', ['service'], unique=False)
    op.create_index(op.f('ix_system_logs_user_id'), 'system_logs', ['user_id'], unique=False)

    # Create drilling_parameters_config table
    op.create_table(
        'drilling_parameters_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rig_id', sa.String(length=50), nullable=False),
        sa.Column('target_wob', sa.Float(), nullable=False),
        sa.Column('target_rpm', sa.Float(), nullable=False),
        sa.Column('target_mud_flow', sa.Float(), nullable=False),
        sa.Column('target_rop', sa.Float(), nullable=False),
        sa.Column('safety_limits', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('rig_id')
    )
    op.create_index(op.f('ix_drilling_parameters_config_id'), 'drilling_parameters_config', ['id'], unique=False)
    op.create_index(op.f('ix_drilling_parameters_config_rig_id'), 'drilling_parameters_config', ['rig_id'], unique=True)


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_table('drilling_parameters_config')
    op.drop_table('system_logs')
    op.drop_table('drilling_sessions')
    op.drop_table('well_profiles')
    op.drop_table('model_versions')
    op.drop_table('anomaly_detections')
    op.drop_table('rul_predictions')
    op.drop_table('change_requests')
    op.drop_table('dvr_process_history')
    op.drop_table('login_attempts')
    op.drop_table('blacklisted_tokens')
    op.drop_table('password_reset_tokens')
    op.drop_table('maintenance_schedules')
    op.drop_table('maintenance_alerts')
    op.drop_table('users')
    op.drop_table('sensor_data')

