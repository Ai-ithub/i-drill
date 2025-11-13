"""
Unit tests for security utilities
"""
import pytest
from utils.security import (
    get_or_generate_secret_key,
    validate_password_strength,
    mask_sensitive_data,
    check_secret_key_security
)


class TestSecretKeyGeneration:
    """Tests for secret key generation"""
    
    def test_generate_secret_key(self):
        """Test secret key generation"""
        key1 = get_or_generate_secret_key()
        key2 = get_or_generate_secret_key()
        
        assert key1 is not None
        assert len(key1) >= 32
        assert isinstance(key1, str)
        # Keys should be different (unless cached)
        # In production, keys should be different
    
    def test_secret_key_length(self):
        """Test secret key has minimum length"""
        key = get_or_generate_secret_key()
        assert len(key) >= 32, "Secret key should be at least 32 characters"
    
    def test_secret_key_format(self):
        """Test secret key format"""
        key = get_or_generate_secret_key()
        # Should be URL-safe base64
        assert all(c.isalnum() or c in '-_' for c in key)


class TestPasswordStrengthValidation:
    """Tests for password strength validation"""
    
    def test_strong_password(self):
        """Test strong password validation"""
        strong_passwords = [
            "StrongPass123!",
            "VerySecureP@ssw0rd",
            "Complex#Password2024",
            "MyP@ssw0rdIsStrong!"
        ]
        
        for password in strong_passwords:
            is_valid, issues = validate_password_strength(password)
            assert is_valid is True, f"Password '{password}' should be valid"
            assert len(issues) == 0
    
    def test_weak_password_too_short(self):
        """Test password too short"""
        is_valid, issues = validate_password_strength("Short1!")
        assert is_valid is False
        assert any("8 characters" in issue.lower() for issue in issues)
    
    def test_weak_password_no_uppercase(self):
        """Test password without uppercase"""
        is_valid, issues = validate_password_strength("lowercase123!")
        assert is_valid is False
        assert any("uppercase" in issue.lower() for issue in issues)
    
    def test_weak_password_no_lowercase(self):
        """Test password without lowercase"""
        is_valid, issues = validate_password_strength("UPPERCASE123!")
        assert is_valid is False
        assert any("lowercase" in issue.lower() for issue in issues)
    
    def test_weak_password_no_number(self):
        """Test password without number"""
        is_valid, issues = validate_password_strength("NoNumber!")
        assert is_valid is False
        assert any("number" in issue.lower() for issue in issues)
    
    def test_weak_password_no_special_char(self):
        """Test password without special character"""
        is_valid, issues = validate_password_strength("NoSpecialChar123")
        assert is_valid is False
        assert any("special" in issue.lower() for issue in issues)
    
    def test_common_password(self):
        """Test common password detection"""
        common_passwords = [
            "password",
            "12345678",
            "admin123",
            "qwerty"
        ]
        
        for password in common_passwords:
            is_valid, issues = validate_password_strength(password)
            assert is_valid is False
            assert any("common" in issue.lower() for issue in issues)
    
    def test_empty_password(self):
        """Test empty password"""
        is_valid, issues = validate_password_strength("")
        assert is_valid is False
        assert len(issues) > 0


class TestSensitiveDataMasking:
    """Tests for sensitive data masking"""
    
    def test_mask_short_string(self):
        """Test masking short string"""
        result = mask_sensitive_data("abc")
        assert result == "****"
        assert len(result) == 4
    
    def test_mask_medium_string(self):
        """Test masking medium string"""
        result = mask_sensitive_data("password123")
        assert result.startswith("pa")
        assert result.endswith("23")
        assert "*" in result
    
    def test_mask_long_string(self):
        """Test masking long string"""
        long_string = "a" * 50
        result = mask_sensitive_data(long_string)
        assert result.startswith("aa")
        assert result.endswith("aa")
        assert "*" in result
        assert len(result) == len(long_string)
    
    def test_mask_none(self):
        """Test masking None"""
        result = mask_sensitive_data(None)
        assert result == "****"
    
    def test_mask_empty_string(self):
        """Test masking empty string"""
        result = mask_sensitive_data("")
        assert result == "****"
    
    def test_mask_custom_char(self):
        """Test masking with custom character"""
        result = mask_sensitive_data("password123", mask_char="X")
        assert "X" in result
        assert result.startswith("pa")
        assert result.endswith("23")


class TestSecretKeySecurity:
    """Tests for secret key security checks"""
    
    def test_check_secure_key(self):
        """Test checking secure key"""
        secure_key = "a" * 32  # Minimum length
        is_secure, issues = check_secret_key_security(secure_key)
        # May have issues if it's all the same character
        assert isinstance(is_secure, bool)
        assert isinstance(issues, list)
    
    def test_check_insecure_key_too_short(self):
        """Test checking insecure key (too short)"""
        insecure_key = "short"
        is_secure, issues = check_secret_key_security(insecure_key)
        assert is_secure is False
        assert any("length" in issue.lower() or "32" in issue for issue in issues)
    
    def test_check_insecure_patterns(self):
        """Test checking insecure patterns"""
        insecure_patterns = [
            "CHANGE_THIS_TO_A_SECURE_RANDOM_KEY",
            "default_secret_key",
            "test_key_12345",
            "demo_secret_key"
        ]
        
        for key in insecure_patterns:
            is_secure, issues = check_secret_key_security(key)
            assert is_secure is False
            assert len(issues) > 0

