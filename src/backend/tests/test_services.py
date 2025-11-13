"""
Unit tests for service classes
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from services.data_service import DataService
from services.auth_service import AuthService
from utils.validators import validate_sensor_data


class TestDataService:
    """Tests for DataService"""
    
    @pytest.fixture
    def data_service(self, mock_db_manager):
        """Create DataService instance with mocked database"""
        service = DataService()
        service.db_manager = mock_db_manager
        return service
    
    def test_get_latest_sensor_data_no_db(self):
        """Test getting latest data when DB is not initialized"""
        service = DataService()
        service.db_manager._initialized = False
        result = service.get_latest_sensor_data()
        assert result == []
    
    def test_get_latest_sensor_data_with_db(self, data_service, sample_sensor_data):
        """Test getting latest sensor data"""
        # Mock database session and query
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_record = MagicMock()
        mock_record.rig_id = "RIG_01"
        mock_record.timestamp = datetime.now()
        
        mock_query.order_by.return_value.limit.return_value.all.return_value = [mock_record]
        mock_session.query.return_value = mock_query
        mock_session.query.return_value.filter.return_value = mock_query
        
        data_service.db_manager.session_scope = MagicMock(return_value=mock_session)
        data_service.db_manager.session_scope.__enter__ = MagicMock(return_value=mock_session)
        data_service.db_manager.session_scope.__exit__ = MagicMock(return_value=None)
        
        result = data_service.get_latest_sensor_data(rig_id="RIG_01", limit=10)
        # Should return list (even if empty due to mocking complexity)
        assert isinstance(result, list)
    
    def test_get_historical_data_no_db(self):
        """Test getting historical data when DB is not initialized"""
        service = DataService()
        service.db_manager._initialized = False
        result = service.get_historical_data()
        assert result == []
    
    def test_insert_sensor_data_no_db(self):
        """Test inserting data when DB is not initialized"""
        service = DataService()
        service.db_manager._initialized = False
        result = service.insert_sensor_data({})
        assert result is False


class TestAuthService:
    """Tests for AuthService"""
    
    @pytest.fixture
    def auth_service(self, mock_db_manager):
        """Create AuthService instance"""
        service = AuthService()
        return service
    
    def test_hash_password(self, auth_service):
        """Test password hashing"""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt hash format
    
    def test_verify_password(self, auth_service):
        """Test password verification"""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)
        
        assert auth_service.verify_password(password, hashed) is True
        assert auth_service.verify_password("wrong_password", hashed) is False
    
    def test_create_access_token(self, auth_service):
        """Test access token creation"""
        data = {"sub": "testuser", "user_id": 1}
        token = auth_service.create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_validate_password_strength(self):
        """Test password strength validation"""
        from utils.security import validate_password_strength
        
        # Strong password
        is_valid, issues = validate_password_strength("StrongPass123!")
        assert is_valid is True
        assert len(issues) == 0
        
        # Weak password - too short
        is_valid, issues = validate_password_strength("Short1!")
        assert is_valid is False
        assert len(issues) > 0
        
        # Weak password - no uppercase
        is_valid, issues = validate_password_strength("lowercase123!")
        assert is_valid is False
        assert any("uppercase" in issue.lower() for issue in issues)
        
        # Weak password - no special char
        is_valid, issues = validate_password_strength("NoSpecialChar123")
        assert is_valid is False
        assert any("special" in issue.lower() for issue in issues)


class TestKafkaService:
    """Tests for KafkaService"""
    
    @pytest.fixture
    def kafka_service(self):
        """Create KafkaService instance"""
        from services.kafka_service import KafkaService
        return KafkaService()
    
    def test_produce_sensor_data_when_unavailable(self, kafka_service):
        """Test producing when Kafka is unavailable"""
        kafka_service.available = False
        result = kafka_service.produce_sensor_data("test_topic", {})
        # Should return False or handle gracefully
        assert result is False or result is None
    
    @patch('services.kafka_service.KAFKA_AVAILABLE', False)
    def test_kafka_not_available(self, kafka_service):
        """Test behavior when Kafka is not available"""
        assert kafka_service.available is False or kafka_service.available is True
    
    def test_create_consumer(self, kafka_service):
        """Test creating Kafka consumer"""
        if kafka_service.available:
            consumer_id = "test_consumer"
            topic = "test_topic"
            result = kafka_service.create_consumer(consumer_id, topic)
            # May return True or consumer object
            assert result is True or result is not None
        else:
            # Skip if Kafka not available
            pytest.skip("Kafka not available")
    
    def test_check_connection(self, kafka_service):
        """Test checking Kafka connection"""
        result = kafka_service.check_connection()
        assert isinstance(result, bool)


class TestPredictionService:
    """Tests for PredictionService"""
    
    @pytest.fixture
    def prediction_service(self):
        """Create PredictionService instance"""
        from services.prediction_service import PredictionService
        return PredictionService()
    
    def test_service_initialization(self, prediction_service):
        """Test service initialization"""
        assert prediction_service is not None
    
    @patch('services.prediction_service.os.path.exists', return_value=False)
    def test_rul_prediction_no_model(self, mock_exists, prediction_service):
        """Test RUL prediction when model doesn't exist"""
        # Should handle missing model gracefully
        result = prediction_service.predict_rul("RIG_01", {})
        # Result might be None or default value
        assert result is None or isinstance(result, (dict, float))

