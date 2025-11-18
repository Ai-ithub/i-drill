"""
Security Logging Utility
Provides centralized logging for security-related events
"""
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Security event severity levels
class SecuritySeverity(str, Enum):
    """Security event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Security event types
class SecurityEventType(str, Enum):
    """Security event types"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    
    # Authorization events
    PERMISSION_DENIED = "permission_denied"
    ROLE_CHANGED = "role_changed"
    
    # Token events
    TOKEN_BLACKLISTED = "token_blacklisted"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_EXPIRED = "token_expired"
    INVALID_TOKEN = "invalid_token"
    
    # WebSocket events
    WEBSOCKET_CONNECTED = "websocket_connected"
    WEBSOCKET_DISCONNECTED = "websocket_disconnected"
    WEBSOCKET_RATE_LIMIT = "websocket_rate_limit"
    WEBSOCKET_AUTH_FAILED = "websocket_auth_failed"
    
    # API events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    
    # System events
    CONFIGURATION_CHANGED = "configuration_changed"
    SECURITY_SETTING_CHANGED = "security_setting_changed"


def log_security_event(
    event_type: str,
    severity: str = SecuritySeverity.INFO.value,
    message: str = "",
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_path: Optional[str] = None,
    request_method: Optional[str] = None
) -> None:
    """
    Log a security-related event.
    
    This function provides centralized logging for security events with structured
    information that can be used for security monitoring and analysis.
    
    Args:
        event_type: Type of security event (from SecurityEventType enum or string)
        severity: Severity level (info, warning, error, critical)
        message: Human-readable message describing the event
        user_id: Optional user ID associated with the event
        username: Optional username associated with the event
        ip_address: Optional IP address of the client
        user_agent: Optional user agent string
        details: Optional dictionary with additional event details
        request_path: Optional API path that triggered the event
        request_method: Optional HTTP method
    """
    # Build structured log data
    log_data = {
        "event_type": event_type,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        "message": message,
    }
    
    # Add optional fields
    if user_id:
        log_data["user_id"] = user_id
    if username:
        log_data["username"] = username
    if ip_address:
        log_data["ip_address"] = ip_address
    if user_agent:
        log_data["user_agent"] = user_agent
    if request_path:
        log_data["request_path"] = request_path
    if request_method:
        log_data["request_method"] = request_method
    if details:
        log_data["details"] = details
    
    # Log based on severity
    log_message = f"[SECURITY] {event_type}: {message}"
    if user_id or username:
        log_message += f" (user: {username or user_id})"
    if ip_address:
        log_message += f" (IP: {ip_address})"
    
    if severity == SecuritySeverity.CRITICAL.value:
        logger.critical(log_message, extra={"security_event": log_data})
    elif severity == SecuritySeverity.ERROR.value:
        logger.error(log_message, extra={"security_event": log_data})
    elif severity == SecuritySeverity.WARNING.value:
        logger.warning(log_message, extra={"security_event": log_data})
    else:
        logger.info(log_message, extra={"security_event": log_data})
    
    # Optionally write to database (if enabled)
    if os.getenv("ENABLE_SECURITY_EVENT_DB_LOGGING", "false").lower() == "true":
        try:
            _log_to_database(log_data)
        except Exception as e:
            logger.error(f"Failed to log security event to database: {e}")


def _log_to_database(log_data: Dict[str, Any]) -> None:
    """
    Log security event to database (if enabled).
    
    Args:
        log_data: Dictionary with security event data
    """
    try:
        from api.models.database_models import SystemLogDB
        from database import db_manager
        
        with db_manager.session_scope() as session:
            security_log = SystemLogDB(
                level=log_data.get("severity", "info").upper(),
                service="security",
                message=log_data.get("message", ""),
                details=log_data,
                user_id=log_data.get("user_id")
            )
            session.add(security_log)
            session.commit()
    except Exception as e:
        # Don't fail if database logging fails
        logger.debug(f"Database logging not available: {e}")


def log_authentication_event(
    event_type: SecurityEventType,
    username: str,
    success: bool,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    user_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an authentication-related event.
    
    Args:
        event_type: Type of authentication event
        username: Username involved
        success: Whether the authentication was successful
        ip_address: Optional IP address
        user_agent: Optional user agent
        user_id: Optional user ID
        details: Optional additional details
    """
    severity = SecuritySeverity.INFO.value if success else SecuritySeverity.WARNING.value
    
    if event_type == SecurityEventType.LOGIN_SUCCESS:
        message = f"Successful login for user: {username}"
    elif event_type == SecurityEventType.LOGIN_FAILURE:
        message = f"Failed login attempt for user: {username}"
    elif event_type == SecurityEventType.ACCOUNT_LOCKED:
        message = f"Account locked for user: {username}"
        severity = SecuritySeverity.WARNING.value
    else:
        message = f"Authentication event: {event_type.value} for user: {username}"
    
    log_security_event(
        event_type=event_type.value,
        severity=severity,
        message=message,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details
    )


def log_authorization_event(
    event_type: SecurityEventType,
    user_id: int,
    username: str,
    resource: str,
    action: str,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an authorization-related event.
    
    Args:
        event_type: Type of authorization event
        user_id: User ID
        username: Username
        resource: Resource being accessed
        action: Action being performed
        ip_address: Optional IP address
        details: Optional additional details
    """
    if event_type == SecurityEventType.PERMISSION_DENIED:
        message = f"Permission denied for user {username}: {action} on {resource}"
        severity = SecuritySeverity.WARNING.value
    else:
        message = f"Authorization event: {event_type.value} for user {username}"
        severity = SecuritySeverity.INFO.value
    
    if details is None:
        details = {}
    details.update({
        "resource": resource,
        "action": action
    })
    
    log_security_event(
        event_type=event_type.value,
        severity=severity,
        message=message,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        details=details
    )


def log_suspicious_activity(
    activity_type: str,
    description: str,
    ip_address: Optional[str] = None,
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log suspicious activity.
    
    Args:
        activity_type: Type of suspicious activity
        description: Description of the activity
        ip_address: Optional IP address
        user_id: Optional user ID
        username: Optional username
        details: Optional additional details
    """
    log_security_event(
        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY.value,
        severity=SecuritySeverity.WARNING.value,
        message=f"Suspicious activity detected: {activity_type} - {description}",
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        details=details or {}
    )

