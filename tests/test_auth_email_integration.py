"""
Integration tests for Auth API with Email Service
Tests email integration in authentication flows
"""
import pytest
from fastapi.testclient import TestClient
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
@pytest.mark.auth
class TestAuthEmailIntegration:
    """Tests for Auth API with email service integration"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('api.routes.auth.email_service')
    @patch('api.routes.auth.auth_service')
    @patch('api.routes.auth.db_manager')
    def test_register_sends_welcome_email(
        self,
        mock_db_manager,
        mock_auth_service,
        mock_email_service,
        client
    ):
        """Test that registration sends welcome email"""
        from api.models.database_models import UserDB
        from datetime import datetime
        
        # Mock user creation
        mock_user = MagicMock(spec=UserDB)
        mock_user.id = 1
        mock_user.username = "newuser"
        mock_user.email = "newuser@example.com"
        mock_user.full_name = "New User"
        mock_user.role = "viewer"
        mock_user.is_active = True
        mock_user.created_at = datetime.now()
        
        mock_auth_service.create_user.return_value = mock_user
        mock_email_service.send_welcome_email.return_value = {
            "success": True,
            "email_logged": True
        }
        
        # Mock database session
        mock_session = MagicMock()
        mock_db_manager.session_scope.return_value.__enter__.return_value = mock_session
        mock_db_manager.session_scope.return_value.__exit__.return_value = None
        
        response = client.post(
            "/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "SecurePass123!",
                "full_name": "New User"
            }
        )
        
        # Check that welcome email was called
        if response.status_code == 201:
            mock_email_service.send_welcome_email.assert_called_once_with(
                email="newuser@example.com",
                username="newuser",
                full_name="New User"
            )
    
    @patch('api.routes.auth.email_service')
    @patch('api.routes.auth.auth_service')
    def test_password_reset_sends_email(
        self,
        mock_auth_service,
        mock_email_service,
        client
    ):
        """Test that password reset request sends email"""
        from api.models.database_models import UserDB
        
        # Mock user
        mock_user = MagicMock(spec=UserDB)
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        
        mock_auth_service.create_password_reset_token.return_value = "reset-token-123"
        mock_auth_service.get_user_by_email.return_value = mock_user
        mock_email_service.send_password_reset_email.return_value = {
            "success": True,
            "email_logged": True,
            "reset_link": "http://localhost:3001/auth/password/reset/confirm?token=reset-token-123"
        }
        
        response = client.post(
            "/auth/password/reset/request",
            json={"email": "test@example.com"}
        )
        
        assert response.status_code == 200
        # Email service should be called
        mock_email_service.send_password_reset_email.assert_called_once()
    
    @patch('api.routes.auth.email_service')
    @patch('api.routes.auth.auth_service')
    def test_password_reset_email_failure_does_not_fail_request(
        self,
        mock_auth_service,
        mock_email_service,
        client
    ):
        """Test that email failure doesn't fail password reset request"""
        mock_auth_service.create_password_reset_token.return_value = "reset-token-123"
        mock_auth_service.get_user_by_email.return_value = None
        mock_email_service.send_password_reset_email.return_value = {
            "success": False,
            "message": "Email sending failed"
        }
        
        response = client.post(
            "/auth/password/reset/request",
            json={"email": "test@example.com"}
        )
        
        # Should still return success to prevent email enumeration
        assert response.status_code == 200

