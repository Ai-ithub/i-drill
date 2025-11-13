"""
Authentication Service
Handles user authentication, JWT tokens, and password management
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
import os
import secrets
from fastapi import Request

from api.models.schemas import User, Token, TokenData, UserRole
from api.models.database_models import (
    UserDB, PasswordResetTokenDB, BlacklistedTokenDB, LoginAttemptDB
)
from database import db_manager
from utils.security import validate_password_strength, mask_sensitive_data

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Import security utilities
from utils.security import get_or_generate_secret_key

# JWT Configuration
SECRET_KEY = get_or_generate_secret_key()
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))  # 30 days

# Security Configuration
MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
ACCOUNT_LOCKOUT_MINUTES = int(os.getenv("ACCOUNT_LOCKOUT_MINUTES", "30"))
PASSWORD_RESET_TOKEN_EXPIRE_HOURS = int(os.getenv("PASSWORD_RESET_TOKEN_EXPIRE_HOURS", "24"))


class AuthService:
    """Service for authentication operations"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password from database
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """
        Hash a password
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow()  # Issued at
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise
    
    @staticmethod
    def decode_access_token(token: str) -> Optional[TokenData]:
        """
        Decode and verify a JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            scopes: list = payload.get("scopes", [])
            
            if username is None:
                return None
            
            return TokenData(
                username=username,
                user_id=user_id,
                scopes=scopes
            )
        except JWTError as e:
            logger.error(f"JWT decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error decoding token: {e}")
            return None
    
    def authenticate_user(
        self, 
        username: str, 
        password: str,
        request: Optional[Request] = None
    ) -> Optional[UserDB]:
        """
        Authenticate a user with username and password
        
        Includes brute force protection and login attempt tracking.
        
        Args:
            username: Username
            password: Plain text password
            request: FastAPI request object for IP tracking
            
        Returns:
            User object if authenticated, None otherwise
        """
        ip_address = None
        user_agent = None
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
        
        try:
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(
                    UserDB.username == username
                ).first()
                
                # Log login attempt
                attempt = LoginAttemptDB(
                    username=username,
                    ip_address=ip_address,
                    success=False,
                    user_agent=user_agent
                )
                
                if not user:
                    logger.warning(f"User not found: {username}")
                    session.add(attempt)
                    session.commit()
                    return None
                
                # Check if account is locked
                if user.locked_until and user.locked_until > datetime.now():
                    logger.warning(f"Account locked for user: {username}")
                    session.add(attempt)
                    session.commit()
                    return None
                
                if not user.is_active:
                    logger.warning(f"User inactive: {username}")
                    session.add(attempt)
                    session.commit()
                    return None
                
                # Verify password
                if not self.verify_password(password, user.hashed_password):
                    # Increment failed attempts
                    user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
                    
                    # Lock account if max attempts reached
                    if user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
                        user.locked_until = datetime.now() + timedelta(minutes=ACCOUNT_LOCKOUT_MINUTES)
                        logger.warning(
                            f"Account locked for user: {username} "
                            f"after {user.failed_login_attempts} failed attempts"
                        )
                    
                    session.add(attempt)
                    session.commit()
                    logger.warning(f"Invalid password for user: {username}")
                    return None
                
                # Successful login - reset failed attempts
                user.failed_login_attempts = 0
                user.locked_until = None
                user.last_login = datetime.now()
                
                attempt.success = True
                session.add(attempt)
                session.commit()
                
                logger.info(f"User authenticated successfully: {username}")
                return user
                
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[UserDB]:
        """
        Get user by username
        
        Args:
            username: Username
            
        Returns:
            User object if found, None otherwise
        """
        try:
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(
                    UserDB.username == username
                ).first()
                return user
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[UserDB]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found, None otherwise
        """
        try:
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(
                    UserDB.id == user_id
                ).first()
                return user
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[UserDB]:
        """
        Get user by email address
        
        Args:
            email: User email address
            
        Returns:
            User object if found, None otherwise
        """
        try:
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(
                    UserDB.email == email
                ).first()
                return user
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        role: UserRole = UserRole.VIEWER
    ) -> Optional[UserDB]:
        """
        Create a new user
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            full_name: Full name
            role: User role
            
        Returns:
            Created user object if successful, None otherwise
        """
        try:
            # Check if user already exists
            with db_manager.session_scope() as session:
                existing_user = session.query(UserDB).filter(
                    (UserDB.username == username) | (UserDB.email == email)
                ).first()
                
                if existing_user:
                    logger.warning(f"User already exists: {username} / {email}")
                    return None
                
                # Hash password
                hashed_password = self.get_password_hash(password)
                
                # Create user
                user = UserDB(
                    username=username,
                    email=email,
                    hashed_password=hashed_password,
                    full_name=full_name,
                    role=role.value,
                    is_active=True
                )
                
                session.add(user)
                session.commit()
                session.refresh(user)
                
                logger.info(f"User created successfully: {username}")
                return user
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def update_user_password(self, user_id: int, new_password: str) -> bool:
        """
        Update user password
        
        Args:
            user_id: User ID
            new_password: New plain text password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate password strength
            is_valid, issues = validate_password_strength(new_password)
            if not is_valid:
                logger.warning(f"Password validation failed: {', '.join(issues)}")
                return False
            
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(UserDB.id == user_id).first()
                
                if not user:
                    return False
                
                user.hashed_password = self.get_password_hash(new_password)
                user.password_changed_at = datetime.now()
                user.failed_login_attempts = 0  # Reset failed attempts
                user.locked_until = None
                session.commit()
                
                logger.info(f"Password updated for user: {user.username}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating password: {e}")
            return False
    
    def create_refresh_token(self, user_id: int) -> str:
        """
        Create a refresh token for user
        
        Args:
            user_id: User ID
            
        Returns:
            Refresh token string
        """
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        return self.create_access_token(
            data={"sub": f"refresh_{user_id}", "type": "refresh", "user_id": user_id},
            expires_delta=expires_delta
        )
    
    def create_password_reset_token(self, email: str) -> Optional[str]:
        """
        Create a password reset token for user
        
        Args:
            email: User email address
            
        Returns:
            Reset token if user exists, None otherwise
        """
        try:
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(UserDB.email == email).first()
                
                if not user:
                    # Don't reveal if user exists
                    logger.info(f"Password reset requested for non-existent email: {mask_sensitive_data(email)}")
                    return None
                
                # Generate secure token
                token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(hours=PASSWORD_RESET_TOKEN_EXPIRE_HOURS)
                
                # Invalidate previous tokens
                session.query(PasswordResetTokenDB).filter(
                    PasswordResetTokenDB.user_id == user.id,
                    PasswordResetTokenDB.used == False
                ).update({"used": True})
                
                # Create new token
                reset_token = PasswordResetTokenDB(
                    user_id=user.id,
                    token=token,
                    expires_at=expires_at
                )
                
                session.add(reset_token)
                session.commit()
                
                logger.info(f"Password reset token created for user: {user.username}")
                return token
                
        except Exception as e:
            logger.error(f"Error creating password reset token: {e}")
            return None
    
    def verify_password_reset_token(self, token: str) -> Optional[UserDB]:
        """
        Verify password reset token and return user
        
        Args:
            token: Password reset token
            
        Returns:
            User object if token is valid, None otherwise
        """
        try:
            with db_manager.session_scope() as session:
                reset_token = session.query(PasswordResetTokenDB).filter(
                    PasswordResetTokenDB.token == token,
                    PasswordResetTokenDB.used == False,
                    PasswordResetTokenDB.expires_at > datetime.now()
                ).first()
                
                if not reset_token:
                    return None
                
                user = session.query(UserDB).filter(UserDB.id == reset_token.user_id).first()
                return user
                
        except Exception as e:
            logger.error(f"Error verifying password reset token: {e}")
            return None
    
    def use_password_reset_token(self, token: str) -> bool:
        """
        Mark password reset token as used
        
        Args:
            token: Password reset token
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with db_manager.session_scope() as session:
                reset_token = session.query(PasswordResetTokenDB).filter(
                    PasswordResetTokenDB.token == token
                ).first()
                
                if reset_token:
                    reset_token.used = True
                    session.commit()
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error using password reset token: {e}")
            return False
    
    def blacklist_token(self, token: str, user_id: Optional[int] = None, reason: str = "logout") -> bool:
        """
        Add token to blacklist
        
        Args:
            token: JWT token to blacklist
            user_id: User ID (optional)
            reason: Reason for blacklisting
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Decode token to get expiration
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                exp = payload.get("exp")
                if exp:
                    expires_at = datetime.fromtimestamp(exp)
                else:
                    expires_at = datetime.now() + timedelta(days=1)
            except:
                expires_at = datetime.now() + timedelta(days=1)
            
            with db_manager.session_scope() as session:
                blacklisted = BlacklistedTokenDB(
                    token=token,
                    user_id=user_id,
                    expires_at=expires_at,
                    reason=reason
                )
                
                session.add(blacklisted)
                session.commit()
                
                logger.info(f"Token blacklisted (reason: {reason})")
                return True
                
        except Exception as e:
            logger.error(f"Error blacklisting token: {e}")
            return False
    
    def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if token is blacklisted
        
        Args:
            token: JWT token to check
            
        Returns:
            True if blacklisted, False otherwise
        """
        try:
            with db_manager.session_scope() as session:
                blacklisted = session.query(BlacklistedTokenDB).filter(
                    BlacklistedTokenDB.token == token,
                    BlacklistedTokenDB.expires_at > datetime.now()
                ).first()
                
                return blacklisted is not None
                
        except Exception as e:
            logger.error(f"Error checking token blacklist: {e}")
            return False
    
    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired blacklisted tokens and reset tokens
        
        Returns:
            Number of tokens cleaned up
        """
        try:
            count = 0
            with db_manager.session_scope() as session:
                # Clean blacklisted tokens
                expired_blacklisted = session.query(BlacklistedTokenDB).filter(
                    BlacklistedTokenDB.expires_at <= datetime.now()
                ).delete()
                count += expired_blacklisted
                
                # Clean expired reset tokens
                expired_reset = session.query(PasswordResetTokenDB).filter(
                    PasswordResetTokenDB.expires_at <= datetime.now()
                ).delete()
                count += expired_reset
                
                session.commit()
                logger.info(f"Cleaned up {count} expired tokens")
                return count
                
        except Exception as e:
            logger.error(f"Error cleaning up tokens: {e}")
            return 0
    
    def check_user_permission(self, user: UserDB, required_role: UserRole) -> bool:
        """
        Check if user has required role/permission
        
        Role hierarchy: admin > data_scientist > engineer > operator > maintenance > viewer
        
        Args:
            user: User object
            required_role: Required role
            
        Returns:
            True if user has permission, False otherwise
        """
        role_hierarchy = {
            UserRole.ADMIN: 6,
            UserRole.DATA_SCIENTIST: 5,
            UserRole.ENGINEER: 4,
            UserRole.OPERATOR: 3,
            UserRole.MAINTENANCE: 2,
            UserRole.VIEWER: 1
        }
        
        try:
            user_role = UserRole(user.role)
            user_level = role_hierarchy.get(user_role, 0)
            required_level = role_hierarchy.get(required_role, 0)
            
            return user_level >= required_level
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False


# Global auth service instance
auth_service = AuthService()


def ensure_default_admin_account() -> bool:
    """
    Ensure a default admin user exists for initial access
    
    WARNING: This function creates a default admin account if it doesn't exist.
    In production, you should:
    1. Set DEFAULT_ADMIN_PASSWORD to a strong password
    2. Change the default admin password immediately after first login
    3. Consider disabling this function in production
    """
    default_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    default_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
    default_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    app_env = os.getenv("APP_ENV", "development").lower()

    # Security warnings for production
    if app_env == "production":
        if default_password == "admin123" or len(default_password) < 12:
            logger.critical(
                "⚠️ SECURITY WARNING: Default admin password is insecure or too short! "
                "Set DEFAULT_ADMIN_PASSWORD environment variable to a strong password (minimum 12 characters)."
            )
        if default_username == "admin":
            logger.warning(
                "⚠️ SECURITY WARNING: Using default admin username. "
                "Consider changing DEFAULT_ADMIN_USERNAME in production."
            )
    else:
        # Development mode warnings
        if default_password == "admin123":
            logger.warning(
                "⚠️ Using default admin password 'admin123'. "
                "This is acceptable for development but MUST be changed in production!"
            )

    try:
        with db_manager.session_scope() as session:
            existing = session.query(UserDB).filter(UserDB.username == default_username).first()
            if existing:
                logger.debug(f"Default admin user already exists: {default_username}")
                return

            hashed_password = auth_service.get_password_hash(default_password)
            user = UserDB(
                username=default_username,
                email=default_email,
                hashed_password=hashed_password,
                full_name="System Administrator",
                role=UserRole.ADMIN.value,
                is_active=True,
            )
            session.add(user)
            logger.info(f"✅ Default admin user created: {default_username}")
            if app_env == "production":
                logger.warning(
                    "⚠️ IMPORTANT: Change the default admin password immediately after first login!"
                )
    except Exception as seed_error:
        logger.error(f"Failed to ensure default admin account: {seed_error}")
