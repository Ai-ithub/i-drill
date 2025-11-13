"""
Tests for database operations
"""
import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from api.models.database_models import SensorData, UserDB, ChangeRequestDB


class TestDatabaseModels:
    """Tests for database models"""
    
    def test_sensor_data_model(self, test_db_session):
        """Test SensorData model creation"""
        sensor_data = SensorData(
            rig_id="RIG_01",
            timestamp=datetime.now(),
            depth=5000.0,
            wob=15000.0,
            rpm=100.0,
            torque=10000.0
        )
        test_db_session.add(sensor_data)
        test_db_session.commit()
        
        assert sensor_data.id is not None
        assert sensor_data.rig_id == "RIG_01"
        assert sensor_data.depth == 5000.0
    
    def test_user_model(self, test_db_session):
        """Test UserDB model creation"""
        from services.auth_service import AuthService
        auth_service = AuthService()
        
        user = UserDB(
            username="testuser",
            email="test@example.com",
            hashed_password=auth_service.hash_password("TestPassword123!"),
            role="engineer",
            is_active=True
        )
        test_db_session.add(user)
        test_db_session.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.is_active is True
    
    def test_change_request_model(self, test_db_session):
        """Test ChangeRequestDB model creation"""
        change_request = ChangeRequestDB(
            rig_id="RIG_01",
            change_type="parameter",
            component="drilling",
            parameter="rpm",
            old_value="100.0",
            new_value="120.0",
            status="pending"
        )
        test_db_session.add(change_request)
        test_db_session.commit()
        
        assert change_request.id is not None
        assert change_request.rig_id == "RIG_01"
        assert change_request.status == "pending"


class TestDatabaseQueries:
    """Tests for database queries"""
    
    def test_query_sensor_data(self, test_db_session):
        """Test querying sensor data"""
        # Create test data
        sensor_data = SensorData(
            rig_id="RIG_01",
            timestamp=datetime.now(),
            depth=5000.0,
            wob=15000.0
        )
        test_db_session.add(sensor_data)
        test_db_session.commit()
        
        # Query data
        result = test_db_session.query(SensorData).filter(
            SensorData.rig_id == "RIG_01"
        ).first()
        
        assert result is not None
        assert result.rig_id == "RIG_01"
    
    def test_query_user_by_username(self, test_db_session):
        """Test querying user by username"""
        from services.auth_service import AuthService
        auth_service = AuthService()
        
        user = UserDB(
            username="testuser",
            email="test@example.com",
            hashed_password=auth_service.hash_password("TestPassword123!"),
            role="engineer"
        )
        test_db_session.add(user)
        test_db_session.commit()
        
        # Query user
        result = test_db_session.query(UserDB).filter(
            UserDB.username == "testuser"
        ).first()
        
        assert result is not None
        assert result.username == "testuser"
    
    def test_query_change_requests_by_status(self, test_db_session):
        """Test querying change requests by status"""
        # Create test data
        change1 = ChangeRequestDB(
            rig_id="RIG_01",
            change_type="parameter",
            component="drilling",
            parameter="rpm",
            new_value="120.0",
            status="pending"
        )
        change2 = ChangeRequestDB(
            rig_id="RIG_01",
            change_type="parameter",
            component="drilling",
            parameter="wob",
            new_value="16000.0",
            status="applied"
        )
        test_db_session.add_all([change1, change2])
        test_db_session.commit()
        
        # Query pending changes
        pending = test_db_session.query(ChangeRequestDB).filter(
            ChangeRequestDB.status == "pending"
        ).all()
        
        assert len(pending) == 1
        assert pending[0].status == "pending"


class TestDatabaseConstraints:
    """Tests for database constraints"""
    
    def test_unique_username(self, test_db_session):
        """Test unique username constraint"""
        from services.auth_service import AuthService
        auth_service = AuthService()
        
        user1 = UserDB(
            username="testuser",
            email="test1@example.com",
            hashed_password=auth_service.hash_password("Password123!"),
            role="engineer"
        )
        test_db_session.add(user1)
        test_db_session.commit()
        
        # Try to create duplicate username
        user2 = UserDB(
            username="testuser",
            email="test2@example.com",
            hashed_password=auth_service.hash_password("Password123!"),
            role="operator"
        )
        test_db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_unique_email(self, test_db_session):
        """Test unique email constraint"""
        from services.auth_service import AuthService
        auth_service = AuthService()
        
        user1 = UserDB(
            username="user1",
            email="test@example.com",
            hashed_password=auth_service.hash_password("Password123!"),
            role="engineer"
        )
        test_db_session.add(user1)
        test_db_session.commit()
        
        # Try to create duplicate email
        user2 = UserDB(
            username="user2",
            email="test@example.com",
            hashed_password=auth_service.hash_password("Password123!"),
            role="operator"
        )
        test_db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()

