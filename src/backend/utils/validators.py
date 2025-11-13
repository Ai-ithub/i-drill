"""
Input validation utilities for i-Drill API

This module provides comprehensive input validation and sanitization functions
for the i-Drill backend API. It includes validation for:
- Rig IDs and identifiers
- Timestamps and time ranges
- Numeric ranges and values
- Sensor data structures
- Pagination parameters
- Email addresses and usernames
- SQL injection prevention

All validation functions return boolean values or tuples of (is_valid, error_message)
to provide clear feedback about validation failures.

Example:
    >>> from utils.validators import validate_rig_id, validate_sensor_data
    >>> 
    >>> # Validate rig ID
    >>> is_valid = validate_rig_id("RIG_01")
    >>> assert is_valid is True
    >>> 
    >>> # Validate sensor data
    >>> data = {"rig_id": "RIG_01", "timestamp": "2024-01-01T00:00:00"}
    >>> is_valid, error = validate_sensor_data(data)
    >>> assert is_valid is True
"""
from typing import Any, Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)


def validate_rig_id(rig_id: str) -> bool:
    """
    Validate rig ID format
    
    Args:
        rig_id: Rig identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not rig_id or not isinstance(rig_id, str):
        return False
    
    # Rig ID should be alphanumeric with underscores/hyphens, 1-50 chars
    pattern = r'^[a-zA-Z0-9_-]{1,50}$'
    return bool(re.match(pattern, rig_id))


def validate_timestamp(timestamp: Any) -> bool:
    """
    Validate timestamp format
    
    Args:
        timestamp: Timestamp to validate (datetime, string, or None)
        
    Returns:
        True if valid, False otherwise
    """
    if timestamp is None:
        return True
    
    if isinstance(timestamp, datetime):
        return True
    
    if isinstance(timestamp, str):
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except (ValueError, AttributeError):
            return False
    
    return False


def validate_numeric_range(value: Any, min_val: float = None, max_val: float = None) -> bool:
    """
    Validate numeric value is within range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        num_value = float(value)
        
        if min_val is not None and num_value < min_val:
            return False
        
        if max_val is not None and num_value > max_val:
            return False
        
        return True
    except (ValueError, TypeError):
        return False


def sanitize_string(value: Any, max_length: int = 1000) -> Optional[str]:
    """
    Sanitize string input
    
    Args:
        value: Value to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string or None if invalid
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        try:
            value = str(value)
        except:
            return None
    
    # Remove control characters except newline and tab
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', value)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"String truncated to {max_length} characters")
    
    return sanitized.strip()


def validate_sensor_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate sensor data structure and values
    
    Args:
        data: Sensor data dictionary
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    # Required fields
    required_fields = ['rig_id', 'timestamp']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate rig_id
    if not validate_rig_id(data['rig_id']):
        return False, f"Invalid rig_id format: {data['rig_id']}"
    
    # Validate timestamp
    if not validate_timestamp(data['timestamp']):
        return False, f"Invalid timestamp format: {data['timestamp']}"
    
    # Validate numeric fields if present
    numeric_fields = {
        'depth': (0, None),
        'wob': (0, None),
        'rpm': (0, 300),
        'torque': (0, None),
        'rop': (0, None),
        'mud_flow': (0, None),
        'mud_pressure': (0, None),
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in data and data[field] is not None:
            if not validate_numeric_range(data[field], min_val, max_val):
                return False, f"Invalid value for {field}: {data[field]}"
    
    return True, None


def validate_pagination_params(limit: int, offset: int, max_limit: int = 10000) -> tuple[bool, Optional[str]]:
    """
    Validate pagination parameters
    
    Args:
        limit: Number of records to return
        offset: Number of records to skip
        max_limit: Maximum allowed limit
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(limit, int) or limit < 1:
        return False, "limit must be a positive integer"
    
    if limit > max_limit:
        return False, f"limit cannot exceed {max_limit}"
    
    if not isinstance(offset, int) or offset < 0:
        return False, "offset must be a non-negative integer"
    
    return True, None


def validate_time_range(start_time: datetime, end_time: datetime) -> tuple[bool, Optional[str]]:
    """
    Validate time range
    
    Args:
        start_time: Start time
        end_time: End time
        
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(start_time, datetime):
        return False, "start_time must be a datetime object"
    
    if not isinstance(end_time, datetime):
        return False, "end_time must be a datetime object"
    
    if start_time >= end_time:
        return False, "start_time must be before end_time"
    
    # Check for reasonable range (not more than 1 year)
    max_range = timedelta(days=365)
    if end_time - start_time > max_range:
        return False, "Time range cannot exceed 365 days"
    
    return True, None


def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_username(username: str) -> bool:
    """
    Validate username format
    
    Args:
        username: Username to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not username or not isinstance(username, str):
        return False
    
    # Username: 3-50 chars, alphanumeric, underscore, hyphen
    pattern = r'^[a-zA-Z0-9_-]{3,50}$'
    return bool(re.match(pattern, username))


def sanitize_sql_input(value: Any) -> Any:
    """
    Sanitize input for SQL queries (basic protection)
    
    Note: This is a basic sanitization. Always use parameterized queries!
    
    Args:
        value: Value to sanitize
        
    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        # Remove SQL injection patterns
        dangerous_patterns = [
            r'--',
            r'/\*',
            r'\*/',
            r';',
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'UPDATE\s+.*\s+SET',
            r'INSERT\s+INTO',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potentially dangerous SQL pattern detected: {pattern}")
                raise ValueError(f"Invalid input detected: {pattern}")
    
    return value

