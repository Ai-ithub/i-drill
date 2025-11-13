"""
Authentication API Routes
Handles user login, registration, and profile management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
import logging

from api.models.schemas import (
    Token,
    User,
    UserCreate,
    UserLogin,
    UserResponse,
    UserRole,
    PasswordResetRequest,
    PasswordResetConfirm,
    PasswordChangeRequest,
    TokenRefreshRequest
)
from api.models.database_models import UserDB
from services.auth_service import auth_service, ACCESS_TOKEN_EXPIRE_MINUTES
from services.email_service import email_service
from api.dependencies import (
    get_current_active_user,
    get_current_admin_user
)
from api.dependencies import oauth2_scheme
from utils.security import validate_password_strength, mask_sensitive_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Rate limiting helper for auth endpoints
# Note: Rate limiting is configured in app.py with specific limits per endpoint type
# Auth endpoints use RATE_LIMIT_AUTH (default: 5/minute)
# The middleware applies these limits automatically


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_create: UserCreate,
    request: Request
):
    """
    Register a new user
    
    Creates a new user account with the provided information.
    Default role is 'viewer' unless specified.
    
    Rate limited: 5 requests per minute
    """
    try:
        # Rate limiting is handled by middleware with RATE_LIMIT_AUTH (default: 5/minute)
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
        
        # Email service integration: Send welcome email to newly registered user
        try:
            email_result = email_service.send_welcome_email(
                email=user.email,
                username=user.username,
                full_name=user.full_name
            )
            
            if email_result.get("email_logged"):
                logger.info(
                    f"Welcome email logged for {mask_sensitive_data(user.email)}. "
                    f"User: {user.username}"
                )
            elif email_result.get("success"):
                logger.info(f"Welcome email sent to {mask_sensitive_data(user.email)}")
            else:
                logger.warning(
                    f"User registered but welcome email failed to send: "
                    f"{email_result.get('message', 'Unknown error')}"
                )
        except Exception as email_error:
            # Don't fail registration if email fails, just log it
            logger.warning(f"Error sending welcome email during registration: {email_error}")
        
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
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None
):
    """
    Login with username and password
    
    Returns a JWT access token and refresh token for authentication.
    Token expires in 24 hours by default.
    """
    try:
        # Authenticate user (with request for IP tracking)
        user = auth_service.authenticate_user(
            form_data.username,
            form_data.password,
            request=request
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
        
        # Create refresh token
        refresh_token = auth_service.create_refresh_token(user.id)
        
        logger.info(f"User logged in: {user.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # in seconds
            refresh_token=refresh_token
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
    user_login: UserLogin,
    request: Request
):
    """
    Login with JSON payload (alternative to form data)
    
    Returns a JWT access token and refresh token for authentication.
    Useful for programmatic access.
    """
    try:
        # Authenticate user
        user = auth_service.authenticate_user(
            user_login.username,
            user_login.password,
            request=request
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
        
        # Create refresh token
        refresh_token = auth_service.create_refresh_token(user.id)
        
        logger.info(f"User logged in (JSON): {user.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token
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


@router.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme),
    current_user: UserDB = Depends(get_current_active_user)
):
    """
    Logout current user
    
    Blacklists the current access token.
    """
    try:
        auth_service.blacklist_token(token, current_user.id, reason="logout")
        
        logger.info(f"User logged out: {current_user.username}")
        
        return {
            "success": True,
            "message": "Logged out successfully"
        }
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_request: TokenRefreshRequest
):
    """
    Refresh access token using refresh token
    
    Returns a new access token and refresh token.
    """
    try:
        from jose import jwt
        from services.auth_service import SECRET_KEY, ALGORITHM
        
        # Decode refresh token
        try:
            payload = jwt.decode(refresh_request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            token_type = payload.get("type")
            user_id = payload.get("user_id")
            
            if token_type != "refresh" or not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
        except:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Get user
        user = auth_service.get_user_by_id(user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "scopes": [user.role]
            },
            expires_delta=access_token_expires
        )
        
        # Create new refresh token
        refresh_token = auth_service.create_refresh_token(user.id)
        
        # Blacklist old refresh token
        auth_service.blacklist_token(refresh_request.refresh_token, user.id, reason="refresh")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/password/reset/request")
async def request_password_reset(
    reset_request: PasswordResetRequest
):
    """
    Request password reset
    
    Sends a password reset token to the user's email (if configured).
    Always returns success to prevent email enumeration.
    """
    try:
        token = auth_service.create_password_reset_token(reset_request.email)
        
        # Send email with reset link if token was generated
        if token:
            # Get user for username (optional, for email personalization)
            user = auth_service.get_user_by_email(reset_request.email)
            username = user.username if user else None
            
            # Send password reset email
            email_result = email_service.send_password_reset_email(
                email=reset_request.email,
                reset_token=token,
                username=username
            )
            
            if email_result.get("email_logged"):
                # In development, email is logged instead of sent
                logger.info(
                    f"Password reset email logged for {mask_sensitive_data(reset_request.email)}. "
                    f"Reset link: {email_result.get('reset_link', 'N/A')}"
                )
            elif email_result.get("success"):
                logger.info(f"Password reset email sent to {mask_sensitive_data(reset_request.email)}")
            else:
                logger.warning(
                    f"Password reset token generated but email failed to send: "
                    f"{email_result.get('message', 'Unknown error')}"
                )
        else:
            logger.debug(f"Password reset token not generated (email may not exist): {mask_sensitive_data(reset_request.email)}")
        
        # Always return success to prevent email enumeration
        return {
            "success": True,
            "message": "If the email exists, a password reset link has been sent."
        }
        
    except Exception as e:
        logger.error(f"Error requesting password reset: {e}")
        # Still return success to prevent enumeration
        return {
            "success": True,
            "message": "If the email exists, a password reset link has been sent."
        }


@router.post("/password/reset/confirm")
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm
):
    """
    Confirm password reset with token
    
    Resets the user's password using the reset token.
    """
    try:
        # Verify token and get user
        user = auth_service.verify_password_reset_token(reset_confirm.token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Validate password strength
        is_valid, issues = validate_password_strength(reset_confirm.new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password does not meet requirements: {', '.join(issues)}"
            )
        
        # Update password
        success = auth_service.update_user_password(user.id, reset_confirm.new_password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset password"
            )
        
        # Mark token as used
        auth_service.use_password_reset_token(reset_confirm.token)
        
        logger.info(f"Password reset completed for user: {user.username}")
        
        return {
            "success": True,
            "message": "Password reset successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error confirming password reset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )


@router.put("/me/password", response_model=dict)
async def update_password(
    password_change: PasswordChangeRequest,
    current_user: UserDB = Depends(get_current_active_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Update current user's password
    
    Requires the current password for verification.
    Blacklists the current token after password change.
    """
    try:
        # Verify current password
        if not auth_service.verify_password(
            password_change.current_password,
            current_user.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Validate new password strength
        is_valid, issues = validate_password_strength(password_change.new_password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password does not meet requirements: {', '.join(issues)}"
            )
        
        # Update password
        success = auth_service.update_user_password(
            current_user.id,
            password_change.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        # Blacklist current token (force re-login)
        auth_service.blacklist_token(token, current_user.id, reason="password_change")
        
        logger.info(f"Password updated for user: {current_user.username}")
        
        return {
            "success": True,
            "message": "Password updated successfully. Please login again."
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

