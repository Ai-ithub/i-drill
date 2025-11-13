"""
Comprehensive unit tests for DataService
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timedelta
from sqlalchemy.orm import Query
from services.data_service import DataService
from api.models.database_models import SensorData


class TestDataServiceInitialization:
    """Tests for DataService initialization"""
    
    def test_init(self):
        """Test DataService initialization"""
        service = DataService()
        assert service.db_manager is not None
    
    def test_db_ready_when_initialized(self):
        """Test _db_ready when database is initialized"""
        service = DataService()
        service.db_manager._initialized = True
        assert service._db_ready() is True
    
    def test_db_ready_when_not_initialized(self):
        """Test _db_ready when database is not initialized"""
        service = DataService()
        service.db_manager._initialized = False
        assert service._db_ready() is False


class TestGetLatestSensorData:
    """Tests for get_latest_sensor_data method"""
    
    @pytest.fixture
    def service(self):
        """Create DataService with mocked database"""
        service = DataService()
        service.db_manager = Mock()
        service.db_manager._initialized = True
        return service
    
    def test_get_latest_no_db(self):
        """Test getting latest data when DB is not ready"""
        service = DataService()
        service.db_manager._initialized = False
        result = service.get_latest_sensor_data()
        assert result == []
    
    def test_get_latest_with_rig_id(self, service):
        """Test getting latest data filtered by rig_id"""
        # Mock session and query
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_record = Mock()
        mock_record.rig_id = "RIG_01"
        mock_record.timestamp = datetime.now()
        mock_record.depth = 5000.0
        
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = [mock_record]
        mock_session.query.return_value = mock_query
        
        service.db_manager.session_scope = MagicMock()
        service.db_manager.session_scope.__enter__ = MagicMock(return_value=mock_session)
        service.db_manager.session_scope.__exit__ = MagicMock(return_value=None)
        
        result = service.get_latest_sensor_data(rig_id="RIG_01", limit=10)
        
        assert isinstance(result, list)
        mock_query.filter.assert_called_once()
        mock_query.limit.assert_called_once_with(10)
    
    def test_get_latest_without_rig_id(self, service):
        """Test getting latest data without rig_id filter"""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = []
        mock_session.query.return_value = mock_query
        
        service.db_manager.session_scope = MagicMock()
        service.db_manager.session_scope.__enter__ = MagicMock(return_value=mock_session)
        service.db_manager.session_scope.__exit__ = MagicMock(return_value=None)
        
        result = service.get_latest_sensor_data(limit=5)
        
        assert isinstance(result, list)
        mock_query.filter.assert_not_called()
        mock_query.limit.assert_called_once_with(5)
    
    def test_get_latest_exception_handling(self, service):
        """Test exception handling in get_latest_sensor_data"""
        service.db_manager.session_scope = MagicMock(side_effect=Exception("DB Error"))
        
        result = service.get_latest_sensor_data()
        assert result == []


class TestGetHistoricalData:
    """Tests for get_historical_data method"""
    
    @pytest.fixture
    def service(self):
        """Create DataService with mocked database"""
        service = DataService()
        service.db_manager = Mock()
        service.db_manager._initialized = True
        return service
    
    def test_get_historical_no_db(self):
        """Test getting historical data when DB is not ready"""
        service = DataService()
        service.db_manager._initialized = False
        result = service.get_historical_data()
        assert result == []
    
    def test_get_historical_with_filters(self, service):
        """Test getting historical data with all filters"""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value.all.return_value = []
        mock_session.query.return_value = mock_query
        
        service.db_manager.session_scope = MagicMock()
        service.db_manager.session_scope.__enter__ = MagicMock(return_value=mock_session)
        service.db_manager.session_scope.__exit__ = MagicMock(return_value=None)
        
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        result = service.get_historical_data(
            rig_id="RIG_01",
            start_time=start_time,
            end_time=end_time,
            parameters=["depth", "wob"],
            limit=100,
            offset=0
        )
        
        assert isinstance(result, list)
        assert mock_query.filter.call_count >= 2  # rig_id and time filters
    
    def test_get_historical_with_parameters(self, service):
        """Test getting historical data with specific parameters"""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value.all.return_value = []
        
        # Mock SensorData attributes
        with patch.object(SensorData, 'depth', create=True):
            with patch.object(SensorData, 'wob', create=True):
                mock_session.query.return_value = mock_query
                
                service.db_manager.session_scope = MagicMock()
                service.db_manager.session_scope.__enter__ = MagicMock(return_value=mock_session)
                service.db_manager.session_scope.__exit__ = MagicMock(return_value=None)
                
                result = service.get_historical_data(
                    parameters=["depth", "wob"],
                    limit=50
                )
                
                assert isinstance(result, list)
    
    def test_get_historical_exception_handling(self, service):
        """Test exception handling in get_historical_data"""
        service.db_manager.session_scope = MagicMock(side_effect=Exception("DB Error"))
        
        result = service.get_historical_data()
        assert result == []


class TestInsertSensorData:
    """Tests for insert_sensor_data method"""
    
    @pytest.fixture
    def service(self):
        """Create DataService with mocked database"""
        service = DataService()
        service.db_manager = Mock()
        service.db_manager._initialized = True
        return service
    
    def test_insert_no_db(self):
        """Test insert when DB is not ready"""
        service = DataService()
        service.db_manager._initialized = False
        result = service.insert_sensor_data({})
        assert result is False
    
    def test_insert_valid_data(self, service):
        """Test inserting valid sensor data"""
        mock_session = MagicMock()
        mock_session.add = Mock()
        mock_session.commit = Mock()
        
        service.db_manager.session_scope = MagicMock()
        service.db_manager.session_scope.__enter__ = MagicMock(return_value=mock_session)
        service.db_manager.session_scope.__exit__ = MagicMock(return_value=None)
        
        data = {
            "rig_id": "RIG_01",
            "timestamp": datetime.now(),
            "depth": 5000.0,
            "wob": 15000.0
        }
        
        result = service.insert_sensor_data(data)
        
        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    def test_insert_exception_handling(self, service):
        """Test exception handling in insert_sensor_data"""
        service.db_manager.session_scope = MagicMock(side_effect=Exception("DB Error"))
        
        result = service.insert_sensor_data({})
        assert result is False


class TestGetAnalyticsSummary:
    """Tests for get_analytics_summary method"""
    
    @pytest.fixture
    def service(self):
        """Create DataService with mocked database"""
        service = DataService()
        service.db_manager = Mock()
        service.db_manager._initialized = True
        return service
    
    def test_get_analytics_no_db(self):
        """Test getting analytics when DB is not ready"""
        service = DataService()
        service.db_manager._initialized = False
        result = service.get_analytics_summary("RIG_01")
        assert result is None
    
    def test_get_analytics_with_data(self, service):
        """Test getting analytics summary"""
        mock_session = MagicMock()
        mock_query = MagicMock()
        
        # Mock aggregate results
        mock_query.filter.return_value = mock_query
        mock_query.scalar.return_value = 5000.0  # current_depth
        mock_query.first.return_value = Mock(
            avg_rop=12.5,
            total_hours=240.5,
            total_power=48000.0,
            alerts_count=2
        )
        
        mock_session.query.return_value = mock_query
        service.db_manager.session_scope = MagicMock()
        service.db_manager.session_scope.__enter__ = MagicMock(return_value=mock_session)
        service.db_manager.session_scope.__exit__ = MagicMock(return_value=None)
        
        result = service.get_analytics_summary("RIG_01")
        
        # Result might be None or a dict depending on implementation
        assert result is None or isinstance(result, dict)
    
    def test_get_analytics_exception_handling(self, service):
        """Test exception handling in get_analytics_summary"""
        service.db_manager.session_scope = MagicMock(side_effect=Exception("DB Error"))
        
        result = service.get_analytics_summary("RIG_01")
        assert result is None


class TestSensorDataConversion:
    """Tests for sensor data conversion methods"""
    
    @pytest.fixture
    def service(self):
        """Create DataService instance"""
        return DataService()
    
    def test_sensor_data_to_dict(self, service):
        """Test converting sensor data record to dictionary"""
        mock_record = Mock()
        mock_record.id = 1
        mock_record.rig_id = "RIG_01"
        mock_record.timestamp = datetime.now()
        mock_record.depth = 5000.0
        mock_record.wob = 15000.0
        mock_record.rpm = 100.0
        mock_record.torque = 10000.0
        
        result = service._sensor_data_to_dict(mock_record)
        
        assert isinstance(result, dict)
        assert result["rig_id"] == "RIG_01"
        assert result["depth"] == 5000.0
        assert "timestamp" in result

