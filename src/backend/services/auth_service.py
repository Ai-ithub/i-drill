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

from api.models.schemas import User, Token, TokenData, UserRole
from api.models.database_models import UserDB
from database import db_manager

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-09876543210")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours


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
        
        to_encode.update({"exp": expire})
        
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
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserDB]:
        """
        Authenticate a user with username and password
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authenticated, None otherwise
        """
        try:
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(
                    UserDB.username == username
                ).first()
                
                if not user:
                    logger.warning(f"User not found: {username}")
                    return None
                
                if not user.is_active:
                    logger.warning(f"User inactive: {username}")
                    return None
                
                if not self.verify_password(password, user.hashed_password):
                    logger.warning(f"Invalid password for user: {username}")
                    return None
                
                # Update last login
                user.last_login = datetime.now()
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
            with db_manager.session_scope() as session:
                user = session.query(UserDB).filter(UserDB.id == user_id).first()
                
                if not user:
                    return False
                
                user.hashed_password = self.get_password_hash(new_password)
                session.commit()
                
                logger.info(f"Password updated for user: {user.username}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating password: {e}")
            return False
    
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

