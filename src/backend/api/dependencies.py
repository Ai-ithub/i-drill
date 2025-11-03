"""
FastAPI Dependencies
Authentication, authorization, and other reusable dependencies
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from typing import Optional
import logging

from api.models.schemas import User, TokenData, UserRole
from api.models.database_models import UserDB
from services.auth_service import auth_service

logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
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


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> UserDB:
    """
    Get current authenticated user from JWT token
    
    Args:
        security_scopes: Required security scopes
        token: JWT token from request
        
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
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[UserDB]:
    """
    Get current user if token is provided, otherwise None
    Useful for endpoints that work with or without authentication
    
    Args:
        token: Optional JWT token
        
    Returns:
        User object or None
    """
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

