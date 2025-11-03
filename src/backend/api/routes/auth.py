"""
Authentication API Routes
Handles user login, registration, and profile management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
import logging

from api.models.schemas import (
    Token,
    User,
    UserCreate,
    UserLogin,
    UserResponse,
    UserRole
)
from api.models.database_models import UserDB
from services.auth_service import auth_service, ACCESS_TOKEN_EXPIRE_MINUTES
from api.dependencies import (
    get_current_active_user,
    get_current_admin_user
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_create: UserCreate
):
    """
    Register a new user
    
    Creates a new user account with the provided information.
    Default role is 'viewer' unless specified.
    """
    try:
        # Create user
        user = auth_service.create_user(
            username=user_create.username,
            email=user_create.email,
            password=user_create.password,
            full_name=user_create.full_name,
            role=user_create.role
        )
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already exists"
            )
        
        # Convert to response model
        user_response = User(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=UserRole(user.role),
            is_active=user.is_active,
            created_at=user.created_at
        )
        
        return UserResponse(
            success=True,
            user=user_response,
            message="User registered successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    Login with username and password
    
    Returns a JWT access token for authentication.
    Token expires in 24 hours by default.
    """
    try:
        # Authenticate user
        user = auth_service.authenticate_user(
            form_data.username,
            form_data.password
        )
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "scopes": [user.role]
            },
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/login/json", response_model=Token)
async def login_json(
    user_login: UserLogin
):
    """
    Login with JSON payload (alternative to form data)
    
    Returns a JWT access token for authentication.
    Useful for programmatic access.
    """
    try:
        # Authenticate user
        user = auth_service.authenticate_user(
            user_login.username,
            user_login.password
        )
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "scopes": [user.role]
            },
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in (JSON): {user.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during JSON login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=User)
async def get_current_user_profile(
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Get current user profile
    
    Returns the profile information of the currently authenticated user.
    Requires valid JWT token.
    """
    return User(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=UserRole(current_user.role),
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )


@router.put("/me/password")
async def update_password(
    current_password: str,
    new_password: str,
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Update current user's password
    
    Requires the current password for verification.
    """
    try:
        # Verify current password
        if not auth_service.verify_password(
            current_password,
            current_user.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters"
            )
        
        # Update password
        success = auth_service.update_user_password(
            current_user.id,
            new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        logger.info(f"Password updated for user: {current_user.username}")
        
        return {
            "success": True,
            "message": "Password updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )


@router.get("/users", response_model=list[User])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: UserDB = Depends(get_current_admin_user)
):
    """
    List all users (Admin only)
    
    Returns a list of all users in the system.
    Requires admin role.
    """
    try:
        from database import db_manager
        
        with db_manager.session_scope() as session:
            users = session.query(UserDB).offset(skip).limit(limit).all()
            
            return [
                User(
                    id=user.id,
                    username=user.username,
                    email=user.email,
                    full_name=user.full_name,
                    role=UserRole(user.role),
                    is_active=user.is_active,
                    created_at=user.created_at
                )
                for user in users
            ]
            
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: int,
    current_user: UserDB = Depends(get_current_admin_user)
):
    """
    Get user by ID (Admin only)
    
    Returns detailed information about a specific user.
    Requires admin role.
    """
    try:
        user = auth_service.get_user_by_id(user_id)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=UserRole(user.role),
            is_active=user.is_active,
            created_at=user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    new_role: UserRole,
    current_user: UserDB = Depends(get_current_admin_user)
):
    """
    Update user role (Admin only)
    
    Changes the role of a specific user.
    Requires admin role.
    """
    try:
        from database import db_manager
        
        with db_manager.session_scope() as session:
            user = session.query(UserDB).filter(UserDB.id == user_id).first()
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Prevent changing own role
            if user.id == current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot change your own role"
                )
            
            user.role = new_role.value
            session.commit()
            
            logger.info(f"Role updated for user {user.username} to {new_role.value}")
            
            return {
                "success": True,
                "message": f"User role updated to {new_role.value}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user role: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user role"
        )


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    is_active: bool,
    current_user: UserDB = Depends(get_current_admin_user)
):
    """
    Activate or deactivate user (Admin only)
    
    Changes the active status of a specific user.
    Requires admin role.
    """
    try:
        from database import db_manager
        
        with db_manager.session_scope() as session:
            user = session.query(UserDB).filter(UserDB.id == user_id).first()
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Prevent deactivating own account
            if user.id == current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot change your own status"
                )
            
            user.is_active = is_active
            session.commit()
            
            status_text = "activated" if is_active else "deactivated"
            logger.info(f"User {user.username} {status_text}")
            
            return {
                "success": True,
                "message": f"User {status_text} successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: UserDB = Depends(get_current_admin_user)
):
    """
    Delete user (Admin only)
    
    Permanently deletes a user account.
    Requires admin role.
    """
    try:
        from database import db_manager
        
        with db_manager.session_scope() as session:
            user = session.query(UserDB).filter(UserDB.id == user_id).first()
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Prevent deleting own account
            if user.id == current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete your own account"
                )
            
            username = user.username
            session.delete(user)
            session.commit()
            
            logger.info(f"User deleted: {username}")
            
            return {
                "success": True,
                "message": "User deleted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )

