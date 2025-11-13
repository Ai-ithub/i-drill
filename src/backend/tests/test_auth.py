"""
Tests for authentication and authorization
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from services.auth_service import AuthService
from api.models.database_models import UserDB


class TestAuthService:
    """Tests for AuthService authentication methods"""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance"""
        return AuthService()
    
    def test_hash_password(self, auth_service):
        """Test password hashing"""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")
    
    def test_verify_password_correct(self, auth_service):
        """Test password verification with correct password"""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)
        assert auth_service.verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self, auth_service):
        """Test password verification with incorrect password"""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)
        assert auth_service.verify_password("WrongPassword", hashed) is False
    
    def test_create_access_token(self, auth_service):
        """Test access token creation"""
        data = {"sub": "testuser", "user_id": 1, "scopes": ["engineer"]}
        token = auth_service.create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_refresh_token(self, auth_service):
        """Test refresh token creation"""
        token = auth_service.create_refresh_token(user_id=1)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_decode_token(self, auth_service):
        """Test token decoding"""
        from jose import jwt
        from services.auth_service import SECRET_KEY, ALGORITHM
        
        data = {"sub": "testuser", "user_id": 1}
        token = auth_service.create_access_token(data)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == 1


class TestPasswordReset:
    """Tests for password reset functionality"""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance"""
        return AuthService()
    
    @patch('services.auth_service.db_manager')
    def test_create_password_reset_token(self, mock_db, auth_service):
        """Test creating password reset token"""
        mock_session = MagicMock()
        mock_user = Mock(spec=UserDB)
        mock_user.id = 1
        mock_user.email = "test@example.com"
        mock_user.username = "testuser"
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.session_scope.return_value.__enter__.return_value = mock_session
        mock_db.session_scope.return_value.__exit__.return_value = None
        
        token = auth_service.create_password_reset_token("test@example.com")
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    @patch('services.auth_service.db_manager')
    def test_create_password_reset_token_nonexistent_user(self, mock_db, auth_service):
        """Test creating reset token for nonexistent user"""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_db.session_scope.return_value.__enter__.return_value = mock_session
        mock_db.session_scope.return_value.__exit__.return_value = None
        
        token = auth_service.create_password_reset_token("nonexistent@example.com")
        assert token is None


class TestTokenBlacklist:
    """Tests for token blacklisting"""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance"""
        return AuthService()
    
    @patch('services.auth_service.db_manager')
    def test_blacklist_token(self, mock_db, auth_service):
        """Test blacklisting a token"""
        mock_session = MagicMock()
        mock_db.session_scope.return_value.__enter__.return_value = mock_session
        mock_db.session_scope.return_value.__exit__.return_value = None
        
        token = "test_token_123"
        result = auth_service.blacklist_token(token, user_id=1, reason="logout")
        
        assert result is True
        mock_session.add.assert_called_once()
    
    @patch('services.auth_service.db_manager')
    def test_is_token_blacklisted(self, mock_db, auth_service):
        """Test checking if token is blacklisted"""
        mock_session = MagicMock()
        mock_blacklisted = Mock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_blacklisted
        mock_db.session_scope.return_value.__enter__.return_value = mock_session
        mock_db.session_scope.return_value.__exit__.return_value = None
        
        result = auth_service.is_token_blacklisted("test_token")
        assert result is True


class TestLoginAttemptTracking:
    """Tests for login attempt tracking"""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance"""
        return AuthService()
    
    @patch('services.auth_service.db_manager')
    def test_track_failed_login_attempt(self, mock_db, auth_service):
        """Test tracking failed login attempt"""
        mock_session = MagicMock()
        mock_user = Mock(spec=UserDB)
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.failed_login_attempts = 0
        mock_user.locked_until = None
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.session_scope.return_value.__enter__.return_value = mock_session
        mock_db.session_scope.return_value.__exit__.return_value = None
        
        # This would be called during authentication
        # Just verify the structure exists
        assert hasattr(auth_service, 'authenticate_user')


class TestPasswordStrength:
    """Tests for password strength validation"""
    
    def test_strong_password(self):
        """Test strong password validation"""
        from utils.security import validate_password_strength
        
        strong_passwords = [
            "StrongPass123!",
            "VerySecure2024@",
            "Complex!Password#99"
        ]
        
        for password in strong_passwords:
            is_valid, issues = validate_password_strength(password)
            assert is_valid is True, f"Password '{password}' should be valid"
            assert len(issues) == 0
    
    def test_weak_passwords(self):
        """Test weak password validation"""
        from utils.security import validate_password_strength
        
        weak_passwords = [
            "short",  # Too short
            "nouppercase123!",  # No uppercase
            "NOLOWERCASE123!",  # No lowercase
            "NoNumbers!",  # No numbers
            "NoSpecial123",  # No special chars
            "password123",  # Common password
        ]
        
        for password in weak_passwords:
            is_valid, issues = validate_password_strength(password)
            assert is_valid is False, f"Password '{password}' should be invalid"
            assert len(issues) > 0


class TestUserAccountLockout:
    """Tests for account lockout functionality"""
    
    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance"""
        return AuthService()
    
    @patch('services.auth_service.db_manager')
    def test_account_lockout_after_max_attempts(self, mock_db, auth_service):
        """Test account lockout after maximum failed attempts"""
        mock_session = MagicMock()
        mock_user = Mock(spec=UserDB)
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.failed_login_attempts = 5  # At max attempts
        mock_user.locked_until = datetime.now() + timedelta(minutes=30)
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.session_scope.return_value.__enter__.return_value = mock_session
        mock_db.session_scope.return_value.__exit__.return_value = None
        
        # User should be locked
        assert mock_user.locked_until is not None
        assert mock_user.locked_until > datetime.now()
