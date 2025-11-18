"""
FastAPI Dependencies
Authentication, authorization, and other reusable dependencies
"""
from fastapi import Depends, HTTPException, status, Request, WebSocket
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from typing import Optional
import logging
from datetime import datetime

from api.models.schemas import User, TokenData, UserRole
from api.models.database_models import UserDB
from services.auth_service import auth_service

logger = logging.getLogger(__name__)

# Cookie names for tokens
ACCESS_TOKEN_COOKIE_NAME = "i_drill_access_token"
REFRESH_TOKEN_COOKIE_NAME = "i_drill_refresh_token"

# OAuth2 scheme for token authentication (kept for backward compatibility)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    scopes={
        "admin": "Full system access",
        "data_scientist": "Model and data access",
        "engineer": "Engineering and configuration access",
        "operator": "Operational control access",
        "maintenance": "Maintenance management access",
        "viewer": "Read-only access"
    }
)


async def get_token_from_cookie_or_header(request: Request) -> Optional[str]:
    """
    Get access token from cookie (preferred) or Authorization header (fallback)
    
    Args:
        request: FastAPI request object
        
    Returns:
        Token string or None
    """
    # First, try to get token from cookie (httpOnly cookie)
    token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    
    # Fallback to Authorization header for backward compatibility
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    return token


async def get_token_from_websocket(websocket: WebSocket) -> Optional[str]:
    """
    Get access token from WebSocket connection.
    
    WebSocket connections can pass tokens via:
    1. Query parameter (token) - fallback method
    2. Cookie (httpOnly cookie) - preferred method (automatically sent by browser)
    
    Args:
        websocket: FastAPI WebSocket instance
        
    Returns:
        Token string or None
    """
    # Try to get token from query parameters (fallback method)
    query_params = websocket.query_params
    token = query_params.get("token")
    
    # Try to get token from cookies (preferred method - httpOnly cookies are automatically sent)
    if not token:
        # WebSocket in FastAPI can access cookies through headers
        cookie_header = websocket.headers.get("cookie", "")
        if cookie_header:
            # Parse cookies manually
            cookies = {}
            for cookie in cookie_header.split(";"):
                cookie = cookie.strip()
                if "=" in cookie:
                    key, value = cookie.split("=", 1)
                    cookies[key.strip()] = value.strip()
            token = cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    
    return token


async def authenticate_websocket(websocket: WebSocket) -> Optional[UserDB]:
    """
    Authenticate WebSocket connection using token.
    
    Validates the token and returns the authenticated user.
    Closes the WebSocket connection if authentication fails.
    
    Args:
        websocket: FastAPI WebSocket instance
        
    Returns:
        Authenticated user object or None if authentication fails
    """
    try:
        # Get token from WebSocket
        token = await get_token_from_websocket(websocket)
        
        if not token:
            from utils.security_logging import log_security_event, SecurityEventType
            await websocket.close(code=1008, reason="Authentication required")
            log_security_event(
                event_type=SecurityEventType.WEBSOCKET_AUTH_FAILED.value,
                severity="warning",
                message="WebSocket connection rejected: No token provided",
                ip_address=websocket.client.host if websocket.client else None
            )
            logger.warning("WebSocket connection rejected: No token provided")
            return None
        
        # Check if token is blacklisted
        if auth_service.is_token_blacklisted(token):
            await websocket.close(code=1008, reason="Token revoked")
            logger.warning("WebSocket connection rejected: Token is blacklisted")
            return None
        
        # Decode token
        token_data = auth_service.decode_access_token(token)
        
        if token_data is None or token_data.username is None:
            await websocket.close(code=1008, reason="Invalid token")
            logger.warning("WebSocket connection rejected: Invalid token")
            return None
        
        # Get user from database
        user = auth_service.get_user_by_username(token_data.username)
        
        if user is None:
            await websocket.close(code=1008, reason="User not found")
            logger.warning(f"WebSocket connection rejected: User not found: {token_data.username}")
            return None
        
        if not user.is_active:
            await websocket.close(code=1008, reason="User inactive")
            logger.warning(f"WebSocket connection rejected: User inactive: {user.username}")
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now():
            await websocket.close(code=1008, reason="Account locked")
            logger.warning(f"WebSocket connection rejected: Account locked: {user.username}")
            return None
        
        logger.info(f"WebSocket authenticated: {user.username}")
        return user
        
    except Exception as e:
        logger.error(f"Error authenticating WebSocket: {e}")
        try:
            await websocket.close(code=1011, reason="Authentication error")
        except:
            pass
        return None


async def get_current_user(
    security_scopes: SecurityScopes,
    request: Request
) -> UserDB:
    """
    Get current authenticated user from JWT token (from cookie or header)
    
    Args:
        security_scopes: Required security scopes
        request: FastAPI request object
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If authentication fails
    """
    # Prepare authentication error
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    # Get token from cookie (preferred) or Authorization header (fallback)
    token = await get_token_from_cookie_or_header(request)
    
    if not token:
        raise credentials_exception
    
    # Check if token is blacklisted
    if auth_service.is_token_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": authenticate_value},
        )
    
    # Decode token
    token_data = auth_service.decode_access_token(token)
    
    if token_data is None or token_data.username is None:
        raise credentials_exception
    
    # Get user from database
    user = auth_service.get_user_by_username(token_data.username)
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is inactive"
        )
    
    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.now():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is temporarily locked"
        )
    
    # Check scopes/roles
    for scope in security_scopes.scopes:
        try:
            required_role = UserRole(scope)
            if not auth_service.check_user_permission(user, required_role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions. Required role: {scope}",
                    headers={"WWW-Authenticate": authenticate_value},
                )
        except ValueError:
            # Invalid role
            pass
    
    return user


async def get_current_active_user(
    current_user: UserDB = Depends(get_current_user)
) -> UserDB:
    """
    Get current active user
    
    Args:
        current_user: Current user from token
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: UserDB = Depends(get_current_user)
) -> UserDB:
    """
    Get current user with admin role
    
    Args:
        current_user: Current user from token
        
    Returns:
        Admin user object
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_current_engineer_user(
    current_user: UserDB = Depends(get_current_user)
) -> UserDB:
    """
    Get current user with engineer role or higher
    
    Args:
        current_user: Current user from token
        
    Returns:
        Engineer user object
        
    Raises:
        HTTPException: If user is not engineer or higher
    """
    if not auth_service.check_user_permission(current_user, UserRole.ENGINEER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Engineer access required"
        )
    return current_user


async def get_optional_current_user(
    request: Request
) -> Optional[UserDB]:
    """
    Get current user if token is provided, otherwise None
    Useful for endpoints that work with or without authentication
    
    Args:
        request: FastAPI request object
        
    Returns:
        User object or None
    """
    token = await get_token_from_cookie_or_header(request)
    
    if token is None:
        return None
    
    try:
        token_data = auth_service.decode_access_token(token)
        if token_data is None:
            return None
        
        user = auth_service.get_user_by_username(token_data.username)
        return user if user and user.is_active else None
        
    except Exception as e:
        logger.warning(f"Error getting optional user: {e}")
        return None


def require_role(required_role: UserRole):
    """
    Decorator factory for requiring specific role
    
    Args:
        required_role: Required user role
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: UserDB = Depends(get_current_user)
    ) -> UserDB:
        if not auth_service.check_user_permission(current_user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role.value}' or higher required"
            )
        return current_user
    
    return role_checker

