"""
Unit tests for validation utilities
"""
import pytest
from datetime import datetime, timedelta
from utils.validators import (
    validate_rig_id,
    validate_timestamp,
    validate_numeric_range,
    sanitize_string,
    validate_sensor_data,
    validate_pagination_params,
    validate_time_range,
    validate_email,
    validate_username,
    sanitize_sql_input
)


class TestRigIdValidation:
    """Tests for rig ID validation"""
    
    def test_valid_rig_id(self):
        """Test valid rig IDs"""
        assert validate_rig_id("RIG_01") is True
        assert validate_rig_id("RIG-01") is True
        assert validate_rig_id("RIG01") is True
        assert validate_rig_id("rig_01") is True
    
    def test_invalid_rig_id(self):
        """Test invalid rig IDs"""
        assert validate_rig_id("") is False
        assert validate_rig_id(None) is False
        assert validate_rig_id("RIG 01") is False  # Space not allowed
        assert validate_rig_id("RIG@01") is False  # Special char
        assert validate_rig_id("A" * 51) is False  # Too long


class TestTimestampValidation:
    """Tests for timestamp validation"""
    
    def test_valid_datetime(self):
        """Test datetime object"""
        assert validate_timestamp(datetime.now()) is True
    
    def test_valid_iso_string(self):
        """Test ISO format string"""
        assert validate_timestamp(datetime.now().isoformat()) is True
        assert validate_timestamp("2024-01-01T00:00:00Z") is True
    
    def test_invalid_timestamp(self):
        """Test invalid timestamps"""
        assert validate_timestamp("invalid") is False
        assert validate_timestamp(12345) is False
    
    def test_none_timestamp(self):
        """Test None timestamp"""
        assert validate_timestamp(None) is True  # None is allowed


class TestNumericRangeValidation:
    """Tests for numeric range validation"""
    
    def test_valid_range(self):
        """Test values within range"""
        assert validate_numeric_range(50, min_val=0, max_val=100) is True
        assert validate_numeric_range(0, min_val=0, max_val=100) is True
        assert validate_numeric_range(100, min_val=0, max_val=100) is True
    
    def test_below_minimum(self):
        """Test values below minimum"""
        assert validate_numeric_range(-1, min_val=0) is False
    
    def test_above_maximum(self):
        """Test values above maximum"""
        assert validate_numeric_range(101, max_val=100) is False
    
    def test_no_bounds(self):
        """Test validation without bounds"""
        assert validate_numeric_range(50) is True
        assert validate_numeric_range(-50) is True
    
    def test_invalid_type(self):
        """Test invalid types"""
        assert validate_numeric_range("not a number") is False
        assert validate_numeric_range(None) is False


class TestStringSanitization:
    """Tests for string sanitization"""
    
    def test_normal_string(self):
        """Test normal string"""
        result = sanitize_string("Hello World")
        assert result == "Hello World"
    
    def test_control_characters(self):
        """Test removal of control characters"""
        result = sanitize_string("Hello\x00World")
        assert "\x00" not in result
    
    def test_max_length(self):
        """Test max length truncation"""
        long_string = "A" * 2000
        result = sanitize_string(long_string, max_length=100)
        assert len(result) == 100
    
    def test_none_input(self):
        """Test None input"""
        assert sanitize_string(None) is None
    
    def test_non_string_input(self):
        """Test non-string input"""
        result = sanitize_string(12345)
        assert result == "12345"


class TestSensorDataValidation:
    """Tests for sensor data validation"""
    
    def test_valid_sensor_data(self):
        """Test valid sensor data"""
        data = {
            "rig_id": "RIG_01",
            "timestamp": datetime.now().isoformat(),
            "depth": 5000.0,
            "wob": 15000.0,
            "rpm": 100.0
        }
        is_valid, error = validate_sensor_data(data)
        assert is_valid is True
        assert error is None
    
    def test_missing_required_fields(self):
        """Test missing required fields"""
        data = {"rig_id": "RIG_01"}  # Missing timestamp
        is_valid, error = validate_sensor_data(data)
        assert is_valid is False
        assert "timestamp" in error.lower()
    
    def test_invalid_rig_id(self):
        """Test invalid rig ID"""
        data = {
            "rig_id": "INVALID RIG ID",
            "timestamp": datetime.now().isoformat()
        }
        is_valid, error = validate_sensor_data(data)
        assert is_valid is False
        assert "rig_id" in error.lower()
    
    def test_invalid_numeric_values(self):
        """Test invalid numeric values"""
        data = {
            "rig_id": "RIG_01",
            "timestamp": datetime.now().isoformat(),
            "depth": -100.0,  # Negative depth
            "rpm": 500.0  # Too high RPM
        }
        is_valid, error = validate_sensor_data(data)
        assert is_valid is False
        assert error is not None


class TestPaginationValidation:
    """Tests for pagination parameter validation"""
    
    def test_valid_pagination(self):
        """Test valid pagination"""
        is_valid, error = validate_pagination_params(limit=10, offset=0)
        assert is_valid is True
        assert error is None
    
    def test_invalid_limit(self):
        """Test invalid limit"""
        is_valid, error = validate_pagination_params(limit=0, offset=0)
        assert is_valid is False
        assert "limit" in error.lower()
    
    def test_limit_exceeds_max(self):
        """Test limit exceeding maximum"""
        is_valid, error = validate_pagination_params(limit=20000, offset=0, max_limit=10000)
        assert is_valid is False
        assert "exceed" in error.lower()
    
    def test_invalid_offset(self):
        """Test invalid offset"""
        is_valid, error = validate_pagination_params(limit=10, offset=-1)
        assert is_valid is False
        assert "offset" in error.lower()


class TestTimeRangeValidation:
    """Tests for time range validation"""
    
    def test_valid_time_range(self):
        """Test valid time range"""
        start = datetime.now()
        end = start + timedelta(days=1)
        is_valid, error = validate_time_range(start, end)
        assert is_valid is True
        assert error is None
    
    def test_invalid_range_start_after_end(self):
        """Test invalid range where start is after end"""
        start = datetime.now()
        end = start - timedelta(days=1)
        is_valid, error = validate_time_range(start, end)
        assert is_valid is False
        assert "before" in error.lower()
    
    def test_range_too_large(self):
        """Test range exceeding maximum"""
        start = datetime.now()
        end = start + timedelta(days=400)  # More than 365 days
        is_valid, error = validate_time_range(start, end)
        assert is_valid is False
        assert "exceed" in error.lower()


class TestEmailValidation:
    """Tests for email validation"""
    
    def test_valid_emails(self):
        """Test valid email addresses"""
        assert validate_email("test@example.com") is True
        assert validate_email("user.name@domain.co.uk") is True
        assert validate_email("user+tag@example.com") is True
    
    def test_invalid_emails(self):
        """Test invalid email addresses"""
        assert validate_email("invalid") is False
        assert validate_email("@example.com") is False
        assert validate_email("test@") is False
        assert validate_email("") is False
        assert validate_email(None) is False


class TestUsernameValidation:
    """Tests for username validation"""
    
    def test_valid_usernames(self):
        """Test valid usernames"""
        assert validate_username("user123") is True
        assert validate_username("user_name") is True
        assert validate_username("user-name") is True
        assert validate_username("a" * 50) is True
    
    def test_invalid_usernames(self):
        """Test invalid usernames"""
        assert validate_username("ab") is False  # Too short
        assert validate_username("a" * 51) is False  # Too long
        assert validate_username("user name") is False  # Space
        assert validate_username("user@name") is False  # Special char
        assert validate_username("") is False
        assert validate_username(None) is False


class TestSQLSanitization:
    """Tests for SQL input sanitization"""
    
    def test_normal_input(self):
        """Test normal input"""
        result = sanitize_sql_input("normal string")
        assert result == "normal string"
    
    def test_sql_injection_patterns(self):
        """Test SQL injection pattern detection"""
        dangerous_inputs = [
            "'; DROP TABLE users; --",
            "/* comment */",
            "'; DELETE FROM users;",
            "'; UPDATE users SET password='hacked';"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(ValueError):
                sanitize_sql_input(dangerous_input)
    
    def test_non_string_input(self):
        """Test non-string input"""
        assert sanitize_sql_input(123) == 123
        assert sanitize_sql_input(None) is None

