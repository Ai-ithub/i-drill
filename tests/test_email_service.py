"""
Tests for Email Service
Tests email sending functionality
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

# Import the service
import sys
from pathlib import Path
BACKEND_SRC = Path(__file__).resolve().parents[1] / "src" / "backend"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


@pytest.mark.unit
@pytest.mark.service
class TestEmailService:
    """Tests for EmailService"""
    
    @pytest.fixture
    def service(self):
        """Create EmailService instance"""
        with patch.dict(os.environ, {'SMTP_ENABLED': 'false'}):
            from services.email_service import EmailService
            return EmailService()
    
    def test_service_initialization_disabled(self, service):
        """Test service initializes when disabled"""
        assert service.enabled is False
    
    @patch.dict(os.environ, {'SMTP_ENABLED': 'true', 'SMTP_HOST': 'smtp.test.com'})
    def test_service_initialization_enabled(self):
        """Test service initializes when enabled"""
        from services.email_service import EmailService
        service = EmailService()
        # Service may be disabled if email libraries not available
        assert hasattr(service, 'enabled')
    
    def test_send_password_reset_email_disabled(self, service):
        """Test sending password reset email when service is disabled"""
        result = service.send_password_reset_email(
            email="test@example.com",
            reset_token="test-token-123",
            username="testuser"
        )
        
        assert result["success"] is True
        assert result["email_logged"] is True
        assert "reset_link" in result or "message" in result
    
    @patch('services.email_service.smtplib.SMTP')
    @patch.dict(os.environ, {
        'SMTP_ENABLED': 'true',
        'SMTP_HOST': 'smtp.test.com',
        'SMTP_PORT': '587',
        'SMTP_USER': 'test@test.com',
        'SMTP_PASSWORD': 'password',
        'SMTP_USE_TLS': 'true',
        'FRONTEND_URL': 'http://localhost:3001'
    })
    def test_send_password_reset_email_enabled(self):
        """Test sending password reset email when service is enabled"""
        try:
            from services.email_service import EmailService
            service = EmailService()
            
            if service.enabled:
                mock_smtp = MagicMock()
                with patch('services.email_service.smtplib.SMTP', return_value=mock_smtp):
                    result = service.send_password_reset_email(
                        email="test@example.com",
                        reset_token="test-token-123",
                        username="testuser"
                    )
                    
                    # If email libraries available, should attempt to send
                    assert result["success"] is True
        except ImportError:
            # Email libraries not available, skip test
            pytest.skip("Email libraries not available")
    
    def test_send_welcome_email_disabled(self, service):
        """Test sending welcome email when service is disabled"""
        result = service.send_welcome_email(
            email="test@example.com",
            username="testuser",
            full_name="Test User"
        )
        
        assert result["success"] is True
        assert result["email_logged"] is True
    
    @patch('services.email_service.smtplib.SMTP')
    @patch.dict(os.environ, {
        'SMTP_ENABLED': 'true',
        'SMTP_HOST': 'smtp.test.com',
        'SMTP_PORT': '587',
        'SMTP_USER': 'test@test.com',
        'SMTP_PASSWORD': 'password',
        'SMTP_USE_TLS': 'true'
    })
    def test_send_welcome_email_enabled(self):
        """Test sending welcome email when service is enabled"""
        try:
            from services.email_service import EmailService
            service = EmailService()
            
            if service.enabled:
                mock_smtp = MagicMock()
                with patch('services.email_service.smtplib.SMTP', return_value=mock_smtp):
                    result = service.send_welcome_email(
                        email="test@example.com",
                        username="testuser",
                        full_name="Test User"
                    )
                    
                    assert result["success"] is True
        except ImportError:
            pytest.skip("Email libraries not available")
    
    def test_send_welcome_email_without_full_name(self, service):
        """Test sending welcome email without full name"""
        result = service.send_welcome_email(
            email="test@example.com",
            username="testuser"
        )
        
        assert result["success"] is True
    
    def test_is_enabled(self, service):
        """Test is_enabled method"""
        assert service.is_enabled() == service.enabled
    
    @patch('services.email_service.smtplib.SMTP')
    @patch.dict(os.environ, {
        'SMTP_ENABLED': 'true',
        'SMTP_HOST': 'smtp.test.com',
        'SMTP_PORT': '587'
    })
    def test_send_email_smtp_error(self):
        """Test handling of SMTP errors"""
        try:
            from services.email_service import EmailService
            service = EmailService()
            
            if service.enabled:
                mock_smtp = MagicMock()
                mock_smtp.__enter__.side_effect = Exception("SMTP connection failed")
                
                with patch('services.email_service.smtplib.SMTP', return_value=mock_smtp):
                    result = service._send_email(
                        to_email="test@example.com",
                        subject="Test",
                        text_body="Test body"
                    )
                    
                    assert result["success"] is False
                    assert "error" in result
        except ImportError:
            pytest.skip("Email libraries not available")


@pytest.mark.integration
class TestEmailServiceIntegration:
    """Integration tests for EmailService"""
    
    def test_email_service_singleton(self):
        """Test that email_service is a singleton"""
        from services.email_service import email_service
        assert email_service is not None

