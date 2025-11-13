"""
Integration tests for Control API endpoints
Tests the control API routes with authentication and authorization
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the app
import sys
from pathlib import Path
BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from app import app


@pytest.mark.integration
@pytest.mark.api
class TestControlAPI:
    """Tests for Control API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock user for authentication"""
        from api.models.database_models import UserDB
        user = MagicMock(spec=UserDB)
        user.id = 1
        user.username = "test_engineer"
        user.role = "engineer"
        user.is_active = True
        return user
    
    @pytest.fixture
    def mock_viewer_user(self):
        """Mock viewer user"""
        from api.models.database_models import UserDB
        user = MagicMock(spec=UserDB)
        user.id = 2
        user.username = "test_viewer"
        user.role = "viewer"
        user.is_active = True
        return user
    
    @pytest.fixture
    def sample_change_request(self):
        """Sample change request data"""
        return {
            "rig_id": "RIG_01",
            "change_type": "optimization",
            "component": "drilling",
            "parameter": "rpm",
            "value": 120.0,
            "auto_execute": False
        }
    
    @patch('api.routes.control.get_current_active_user')
    @patch('api.routes.control.control_service')
    @patch('api.routes.control.data_service')
    @patch('api.routes.control.db_manager')
    def test_apply_change_success(
        self,
        mock_db_manager,
        mock_data_service,
        mock_control_service,
        mock_get_user,
        client,
        mock_user,
        sample_change_request
    ):
        """Test successful change application"""
        from api.models.database_models import ChangeRequestDB
        
        mock_get_user.return_value = mock_user
        mock_control_service.is_available.return_value = True
        mock_control_service.get_parameter_value.return_value = None
        mock_data_service.get_latest_sensor_data.return_value = [
            {"rpm": 100.0, "rig_id": "RIG_01"}
        ]
        
        # Mock database session
        mock_session = MagicMock()
        mock_change = MagicMock(spec=ChangeRequestDB)
        mock_change.id = 1
        mock_change.rig_id = "RIG_01"
        mock_change.status = "pending"
        mock_change.requested_at = datetime.now()
        mock_change.change_type = "optimization"
        mock_change.component = "drilling"
        mock_change.parameter = "rpm"
        mock_change.old_value = "100.0"
        mock_change.new_value = "120.0"
        
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        
        response = client.post(
            "/control/apply-change",
            json=sample_change_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["success"] is True
        assert "change_id" in data
    
    @patch('api.routes.control.get_current_active_user')
    @patch('api.routes.control.control_service')
    def test_apply_change_control_system_unavailable(
        self,
        mock_control_service,
        mock_get_user,
        client,
        mock_user,
        sample_change_request
    ):
        """Test change application when control system is unavailable"""
        mock_get_user.return_value = mock_user
        mock_control_service.is_available.return_value = False
        
        # Set auto_execute to True to trigger availability check
        sample_change_request["auto_execute"] = True
        sample_change_request["value"] = 120.0
        
        # Mock user role to engineer for auto_execute
        mock_user.role = "engineer"
        
        response = client.post(
            "/control/apply-change",
            json=sample_change_request,
            headers={"Authorization": "Bearer test-token"}
        )
        
        # Should return 503 if control system unavailable
        assert response.status_code == 503
    
    @patch('api.routes.control.get_current_active_user')
    @patch('api.routes.control.db_manager')
    def test_get_change_history(
        self,
        mock_db_manager,
        mock_get_user,
        client,
        mock_user
    ):
        """Test getting change history"""
        from api.models.database_models import ChangeRequestDB
        
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = MagicMock()
        mock_change = MagicMock(spec=ChangeRequestDB)
        mock_change.id = 1
        mock_change.rig_id = "RIG_01"
        mock_change.status = "pending"
        mock_change.requested_at = datetime.now()
        mock_change.change_type = "optimization"
        mock_change.component = "drilling"
        mock_change.parameter = "rpm"
        mock_change.old_value = "100.0"
        mock_change.new_value = "120.0"
        mock_change.auto_execute = False
        mock_change.approved_at = None
        mock_change.applied_at = None
        mock_change.rejection_reason = None
        mock_change.error_message = None
        mock_change.metadata = None
        
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = [mock_change]
        mock_session.query.return_value = mock_query
        
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        
        response = client.get(
            "/control/change-history",
            params={"rig_id": "RIG_01", "limit": 50},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "changes" in data
        assert isinstance(data["changes"], list)
    
    @patch('api.routes.control.get_current_engineer_user')
    @patch('api.routes.control.control_service')
    @patch('api.routes.control.db_manager')
    def test_approve_change(
        self,
        mock_db_manager,
        mock_control_service,
        mock_get_user,
        client,
        mock_user
    ):
        """Test approving a change request"""
        from api.models.database_models import ChangeRequestDB
        
        mock_get_user.return_value = mock_user
        mock_control_service.is_available.return_value = True
        mock_control_service.apply_parameter_change.return_value = {
            "success": True,
            "message": "Change applied",
            "applied_at": datetime.now().isoformat()
        }
        mock_control_service.get_parameter_value.return_value = 120.0
        
        # Mock database session
        mock_session = MagicMock()
        mock_change = MagicMock(spec=ChangeRequestDB)
        mock_change.id = 1
        mock_change.rig_id = "RIG_01"
        mock_change.status = "pending"
        mock_change.auto_execute = True
        mock_change.component = "drilling"
        mock_change.parameter = "rpm"
        mock_change.new_value = "120.0"
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_change
        mock_session.commit = MagicMock()
        
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        
        response = client.post(
            "/control/change/1/approve",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @patch('api.routes.control.get_current_engineer_user')
    @patch('api.routes.control.db_manager')
    def test_reject_change(
        self,
        mock_db_manager,
        mock_get_user,
        client,
        mock_user
    ):
        """Test rejecting a change request"""
        from api.models.database_models import ChangeRequestDB
        
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = MagicMock()
        mock_change = MagicMock(spec=ChangeRequestDB)
        mock_change.id = 1
        mock_change.status = "pending"
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_change
        mock_session.commit = MagicMock()
        
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        
        response = client.post(
            "/control/change/1/reject",
            params={"reason": "Not safe"},
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "rejected"
    
    @patch('api.routes.control.get_current_active_user')
    @patch('api.routes.control.db_manager')
    def test_get_change_details(
        self,
        mock_db_manager,
        mock_get_user,
        client,
        mock_user
    ):
        """Test getting change details"""
        from api.models.database_models import ChangeRequestDB
        
        mock_get_user.return_value = mock_user
        
        # Mock database session
        mock_session = MagicMock()
        mock_change = MagicMock(spec=ChangeRequestDB)
        mock_change.id = 1
        mock_change.rig_id = "RIG_01"
        mock_change.status = "pending"
        mock_change.requested_at = datetime.now()
        mock_change.approved_at = None
        mock_change.applied_at = None
        mock_change.rejection_reason = None
        mock_change.error_message = None
        mock_change.metadata = None
        mock_change.change_type = "optimization"
        mock_change.component = "drilling"
        mock_change.parameter = "rpm"
        mock_change.old_value = "100.0"
        mock_change.new_value = "120.0"
        mock_change.auto_execute = False
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_change
        
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        
        response = client.get(
            "/control/change/1",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "change" in data

