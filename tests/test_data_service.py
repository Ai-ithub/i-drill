"""
Unit tests for DataService
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "backend"
sys.path.insert(0, str(src_path))

from services.data_service import DataService
from api.models.database_models import (
    SensorData,
    MaintenanceAlertDB,
    MaintenanceScheduleDB,
    RULPredictionDB,
    DVRProcessHistoryDB
)


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    manager = Mock()
    manager._initialized = True
    manager.session_scope = MagicMock()
    return manager


@pytest.fixture
def data_service(mock_db_manager):
    """Create DataService instance with mocked db_manager"""
    with patch('services.data_service.db_manager', mock_db_manager):
        service = DataService()
        service.db_manager = mock_db_manager
        return service


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session"""
    session = Mock(spec=Session)
    return session


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data for testing"""
    return {
        'id': 1,
        'rig_id': 'RIG_01',
        'timestamp': datetime.now(),
        'depth': 5000.0,
        'wob': 1500.0,
        'rpm': 80.0,
        'torque': 400.0,
        'rop': 12.0,
        'mud_flow': 1200.0,
        'mud_pressure': 3000.0,
        'mud_temperature': 60.0,
        'gamma_ray': 85.0,
        'resistivity': 20.0,
        'density': 2.5,
        'porosity': 0.15,
        'hook_load': 200.0,
        'vibration': 1.5,
        'status': 'normal'
    }


@pytest.fixture
def sample_maintenance_alert():
    """Sample maintenance alert for testing"""
    return {
        'id': 1,
        'rig_id': 'RIG_01',
        'component': 'Motor',
        'alert_type': 'vibration',
        'severity': 'high',
        'message': 'High vibration detected',
        'predicted_failure_time': datetime.now() + timedelta(hours=24),
        'created_at': datetime.now(),
        'acknowledged': False,
        'resolved': False
    }


class TestDataService:
    """Test suite for DataService"""

    def test_db_ready_true(self, data_service, mock_db_manager):
        """Test _db_ready returns True when database is initialized"""
        mock_db_manager._initialized = True
        assert data_service._db_ready() is True

    def test_db_ready_false(self, data_service, mock_db_manager):
        """Test _db_ready returns False when database is not initialized"""
        mock_db_manager._initialized = False
        assert data_service._db_ready() is False

    def test_get_latest_sensor_data_success(self, data_service, mock_session, sample_sensor_data):
        """Test getting latest sensor data successfully"""
        # Setup
        mock_record = Mock(spec=SensorData)
        for key, value in sample_sensor_data.items():
            setattr(mock_record, key, value)
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_record]
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.get_latest_sensor_data(rig_id='RIG_01', limit=10)
        
        # Assert
        assert len(result) == 1
        assert result[0]['rig_id'] == 'RIG_01'
        mock_session.query.assert_called_once_with(SensorData)

    def test_get_latest_sensor_data_no_db(self, data_service, mock_db_manager):
        """Test getting latest sensor data when database is not ready"""
        mock_db_manager._initialized = False
        result = data_service.get_latest_sensor_data()
        assert result == []

    def test_get_historical_data_with_filters(self, data_service, mock_session, sample_sensor_data):
        """Test getting historical data with filters"""
        # Setup
        mock_record = Mock(spec=SensorData)
        for key, value in sample_sensor_data.items():
            setattr(mock_record, key, value)
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_record]
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        # Execute
        result = data_service.get_historical_data(
            rig_id='RIG_01',
            start_time=start_time,
            end_time=end_time,
            limit=100
        )
        
        # Assert
        assert len(result) == 1
        mock_query.filter.assert_called()

    def test_get_time_series_aggregated(self, data_service, mock_session):
        """Test getting aggregated time series data"""
        # Setup
        mock_row = Mock()
        mock_row.time_bucket = datetime.now()
        mock_row.avg_wob = 1500.0
        mock_row.avg_rpm = 80.0
        mock_row.avg_torque = 400.0
        mock_row.avg_rop = 12.0
        mock_row.avg_mud_flow = 1200.0
        mock_row.avg_mud_pressure = 3000.0
        mock_row.max_depth = 5000.0
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [mock_row]
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.get_time_series_aggregated(rig_id='RIG_01', time_bucket_seconds=60)
        
        # Assert
        assert len(result) == 1
        assert result[0]['avg_wob'] == 1500.0

    def test_get_analytics_summary(self, data_service, mock_session, sample_sensor_data):
        """Test getting analytics summary"""
        # Setup
        mock_latest = Mock(spec=SensorData)
        mock_latest.depth = 5000.0
        mock_latest.timestamp = datetime.now()
        
        mock_stats = Mock()
        mock_stats.avg_rop = 12.0
        mock_stats.data_points = 3600
        mock_stats.total_distance = 12.0
        
        mock_query_latest = Mock()
        mock_query_latest.filter.return_value = mock_query_latest
        mock_query_latest.order_by.return_value = mock_query_latest
        mock_query_latest.first.return_value = mock_latest
        
        mock_query_stats = Mock()
        mock_query_stats.filter.return_value = mock_query_stats
        mock_query_stats.first.return_value = mock_stats
        
        mock_query_alerts = Mock()
        mock_query_alerts.filter.return_value = mock_query_alerts
        mock_query_alerts.scalar.return_value = 2
        
        def query_side_effect(model):
            if model == SensorData:
                # First call for latest
                if mock_session.query.call_count == 1:
                    return mock_query_latest
                # Second call for stats
                return mock_query_stats
            elif model == MaintenanceAlertDB:
                return mock_query_alerts
        
        mock_session.query.side_effect = query_side_effect
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.get_analytics_summary(rig_id='RIG_01')
        
        # Assert
        assert result is not None
        assert result['rig_id'] == 'RIG_01'
        assert result['current_depth'] == 5000.0

    def test_insert_sensor_data_success(self, data_service, mock_session):
        """Test inserting sensor data successfully"""
        # Setup
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        sensor_data = {
            'rig_id': 'RIG_01',
            'timestamp': datetime.now(),
            'depth': 5000.0,
            'wob': 1500.0
        }
        
        # Execute
        result = data_service.insert_sensor_data(sensor_data)
        
        # Assert
        assert result is True
        mock_session.add.assert_called_once()

    def test_bulk_insert_sensor_data_success(self, data_service, mock_session):
        """Test bulk inserting sensor data successfully"""
        # Setup
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        sensor_data_list = [
            {'rig_id': 'RIG_01', 'timestamp': datetime.now(), 'depth': 5000.0},
            {'rig_id': 'RIG_01', 'timestamp': datetime.now(), 'depth': 5001.0}
        ]
        
        # Execute
        result = data_service.bulk_insert_sensor_data(sensor_data_list)
        
        # Assert
        assert result is True
        mock_session.bulk_save_objects.assert_called_once()

    def test_get_maintenance_alerts(self, data_service, mock_session, sample_maintenance_alert):
        """Test getting maintenance alerts"""
        # Setup
        mock_alert = Mock(spec=MaintenanceAlertDB)
        for key, value in sample_maintenance_alert.items():
            setattr(mock_alert, key, value)
        
        mock_alert.acknowledged_by = None
        mock_alert.acknowledged_at = None
        mock_alert.acknowledgement_notes = None
        mock_alert.resolved_at = None
        mock_alert.resolved_by = None
        mock_alert.resolution_notes = None
        mock_alert.dvr_history_id = None
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_alert]
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.get_maintenance_alerts(rig_id='RIG_01', limit=10)
        
        # Assert
        assert len(result) == 1
        assert result[0]['rig_id'] == 'RIG_01'

    def test_create_maintenance_alert(self, data_service, mock_session):
        """Test creating maintenance alert"""
        # Setup
        mock_alert = Mock(spec=MaintenanceAlertDB)
        mock_alert.id = 1
        
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        alert_data = {
            'rig_id': 'RIG_01',
            'component': 'Motor',
            'severity': 'high',
            'message': 'High vibration detected'
        }
        
        # Execute
        with patch('services.data_service.MaintenanceAlertDB', return_value=mock_alert):
            result = data_service.create_maintenance_alert(alert_data)
        
        # Assert
        assert result == 1
        mock_session.add.assert_called_once()

    def test_acknowledge_maintenance_alert(self, data_service, mock_session, sample_maintenance_alert):
        """Test acknowledging maintenance alert"""
        # Setup
        mock_alert = Mock(spec=MaintenanceAlertDB)
        for key, value in sample_maintenance_alert.items():
            setattr(mock_alert, key, value)
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_alert
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.acknowledge_maintenance_alert(
            alert_id=1,
            acknowledged_by='user1',
            notes='Acknowledged'
        )
        
        # Assert
        assert result is not None
        assert mock_alert.acknowledged is True
        assert mock_alert.acknowledged_by == 'user1'

    def test_resolve_maintenance_alert(self, data_service, mock_session, sample_maintenance_alert):
        """Test resolving maintenance alert"""
        # Setup
        mock_alert = Mock(spec=MaintenanceAlertDB)
        for key, value in sample_maintenance_alert.items():
            setattr(mock_alert, key, value)
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_alert
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.resolve_maintenance_alert(
            alert_id=1,
            resolved_by='user1',
            notes='Resolved'
        )
        
        # Assert
        assert result is not None
        assert mock_alert.resolved is True
        assert mock_alert.resolved_by == 'user1'

    def test_create_maintenance_schedule(self, data_service, mock_session):
        """Test creating maintenance schedule"""
        # Setup
        mock_schedule = Mock(spec=MaintenanceScheduleDB)
        mock_schedule.id = 1
        
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        schedule_data = {
            'rig_id': 'RIG_01',
            'component': 'Motor',
            'maintenance_type': 'preventive',
            'scheduled_date': datetime.now() + timedelta(days=7)
        }
        
        # Execute
        with patch('services.data_service.MaintenanceScheduleDB', return_value=mock_schedule):
            result = data_service.create_maintenance_schedule(schedule_data)
        
        # Assert
        assert result is not None
        mock_session.add.assert_called_once()

    def test_get_maintenance_schedules(self, data_service, mock_session):
        """Test getting maintenance schedules"""
        # Setup
        mock_schedule = Mock(spec=MaintenanceScheduleDB)
        mock_schedule.id = 1
        mock_schedule.rig_id = 'RIG_01'
        mock_schedule.component = 'Motor'
        mock_schedule.maintenance_type = 'preventive'
        mock_schedule.scheduled_date = datetime.now()
        mock_schedule.estimated_duration_hours = 2.0
        mock_schedule.priority = 'high'
        mock_schedule.status = 'scheduled'
        mock_schedule.assigned_to = 'technician1'
        mock_schedule.notes = 'Regular maintenance'
        mock_schedule.created_at = datetime.now()
        mock_schedule.updated_at = datetime.now()
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_schedule]
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.get_maintenance_schedules(rig_id='RIG_01')
        
        # Assert
        assert len(result) == 1
        assert result[0]['rig_id'] == 'RIG_01'

    def test_update_maintenance_schedule(self, data_service, mock_session):
        """Test updating maintenance schedule"""
        # Setup
        mock_schedule = Mock(spec=MaintenanceScheduleDB)
        mock_schedule.id = 1
        mock_schedule.rig_id = 'RIG_01'
        mock_schedule.component = 'Motor'
        mock_schedule.maintenance_type = 'preventive'
        mock_schedule.scheduled_date = datetime.now()
        mock_schedule.estimated_duration_hours = 2.0
        mock_schedule.priority = 'high'
        mock_schedule.status = 'scheduled'
        mock_schedule.assigned_to = 'technician1'
        mock_schedule.notes = 'Regular maintenance'
        mock_schedule.created_at = datetime.now()
        mock_schedule.updated_at = datetime.now()
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_schedule
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        update_data = {'status': 'completed', 'notes': 'Completed successfully'}
        result = data_service.update_maintenance_schedule(schedule_id=1, data=update_data)
        
        # Assert
        assert result is not None
        assert mock_schedule.status == 'completed'

    def test_delete_maintenance_schedule(self, data_service, mock_session):
        """Test deleting maintenance schedule"""
        # Setup
        mock_schedule = Mock(spec=MaintenanceScheduleDB)
        mock_schedule.id = 1
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_schedule
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.delete_maintenance_schedule(schedule_id=1)
        
        # Assert
        assert result is True
        mock_session.delete.assert_called_once_with(mock_schedule)

    def test_save_rul_prediction(self, data_service, mock_session):
        """Test saving RUL prediction"""
        # Setup
        mock_prediction = Mock(spec=RULPredictionDB)
        mock_prediction.id = 1
        
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        prediction_data = {
            'rig_id': 'RIG_01',
            'component': 'Motor',
            'predicted_rul': 1000.0,
            'confidence': 0.85
        }
        
        # Execute
        with patch('services.data_service.RULPredictionDB', return_value=mock_prediction):
            result = data_service.save_rul_prediction(prediction_data)
        
        # Assert
        assert result == 1
        mock_session.add.assert_called_once()

    def test_get_rul_predictions(self, data_service, mock_session):
        """Test getting RUL predictions"""
        # Setup
        mock_prediction = Mock(spec=RULPredictionDB)
        mock_prediction.id = 1
        mock_prediction.rig_id = 'RIG_01'
        mock_prediction.component = 'Motor'
        mock_prediction.predicted_rul = 1000.0
        mock_prediction.confidence = 0.85
        mock_prediction.timestamp = datetime.now()
        mock_prediction.model_used = 'lstm'
        mock_prediction.recommendation = 'Schedule maintenance'
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_prediction]
        
        mock_session.query.return_value = mock_query
        data_service.db_manager.session_scope.return_value.__enter__.return_value = mock_session
        
        # Execute
        result = data_service.get_rul_predictions(rig_id='RIG_01')
        
        # Assert
        assert len(result) == 1
        assert result[0]['rig_id'] == 'RIG_01'

    def test_sensor_data_to_dict(self, sample_sensor_data):
        """Test converting sensor data to dictionary"""
        # Setup
        mock_record = Mock(spec=SensorData)
        for key, value in sample_sensor_data.items():
            setattr(mock_record, key, value)
        
        # Execute
        result = DataService._sensor_data_to_dict(mock_record)
        
        # Assert
        assert result['id'] == 1
        assert result['rig_id'] == 'RIG_01'
        assert result['depth'] == 5000.0

    def test_alert_to_dict(self, sample_maintenance_alert):
        """Test converting alert to dictionary"""
        # Setup
        mock_alert = Mock(spec=MaintenanceAlertDB)
        for key, value in sample_maintenance_alert.items():
            setattr(mock_alert, key, value)
        mock_alert.acknowledged_by = None
        mock_alert.acknowledged_at = None
        mock_alert.acknowledgement_notes = None
        mock_alert.resolved_at = None
        mock_alert.resolved_by = None
        mock_alert.resolution_notes = None
        mock_alert.dvr_history_id = None
        
        # Execute
        result = DataService._alert_to_dict(mock_alert)
        
        # Assert
        assert result['id'] == 1
        assert result['rig_id'] == 'RIG_01'
        assert result['severity'] == 'high'

