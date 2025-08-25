"""
Authentication endpoints for login, registration, and user management.

BULLETPROOF SECURITY VERSION
"""

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime, timedelta, timezone
import logging

security = HTTPBearer()

# Rate limiting imports
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth.models import UserLogin, UserRegister, Token, UserResponse
from api.auth.utils import (
    verify_password_simple,
    hash_password_simple,
    create_access_token,
    add_token_to_blacklist
)
from api.auth.dependencies import get_current_user, get_current_active_user
from api.core.database import get_database, Collections
from api.core.config import settings

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

@router.post("/login", response_model=Token)
@limiter.limit("5/minute") # SECURITY: Max 5 login attempts per minute
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Secure user login with rate limiting and enhanced security.
    """
    try:
        # Find user by username or email
        user = await db[Collections.USERS].find_one({
            "$or": [
                {"username": form_data.username},
                {"email": form_data.username}
            ]
        })
        if not user:
            # Log failed login attempt
            logger.warning(f"Failed login attempt for non-existent user: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user is blocked
        if user.get("is_blocked", False):
            logger.warning(f"Blocked user {user['username']} attempted login")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account has been blocked. Contact support."
            )

        # SECURE: Use only secure password verification
        password_valid = verify_password_simple(form_data.password, user["password"])
        if not password_valid:
            # Log failed login attempt
            logger.warning(f"Failed login attempt for user: {user['username']}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Update last login with timezone-aware datetime
        await db[Collections.USERS].update_one(
            {"username": user["username"]},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )

        # Create JWT token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]},
            expires_delta=access_token_expires
        )

        # Prepare user info for response
        user_info = {
            "username": user["username"],
            "email": user["email"],
            "name": user.get("name", "User"),
            "credits": user.get("credits", 0),
            "plan": user.get("plan", "Free Plan")
        }

        logger.info(f"User {user['username']} logged in successfully")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user_info": user_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/register", response_model=Token)
@limiter.limit("3/minute") # SECURITY: Max 3 registrations per minute
async def register(
    request: Request,
    user_data: UserRegister,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Secure user registration with rate limiting.
    """
    try:
        # Check if username or email already exists
        existing_user = await db[Collections.USERS].find_one({
            "$or": [
                {"username": user_data.username},
                {"email": user_data.email}
            ]
        })
        if existing_user:
            if existing_user["username"] == user_data.username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )

        # Create user record with timezone-aware datetime
        user_record = {
            "username": user_data.username,
            "email": user_data.email,
            "name": user_data.name,
            "password": hash_password_simple(user_data.password), # Uses secure validation
            "created_at": datetime.now(timezone.utc),
            "total_predictions": 0,
            "last_login": datetime.now(timezone.utc),
            "credits": 5, # Start with 5 free credits
            "plan": "Free Plan",
            "credits_used": 0,
            "last_prediction": None,
            "is_blocked": False,
            "is_admin": False # Default to non-admin
        }

        # Insert user into database
        result = await db[Collections.USERS].insert_one(user_record)
        if not result.inserted_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create account"
            )

        # Create JWT token for immediate login
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_data.username},
            expires_delta=access_token_expires
        )

        # Prepare user info for response
        user_info = {
            "username": user_data.username,
            "email": user_data.email,
            "name": user_data.name,
            "credits": 5,
            "plan": "Free Plan"
        }

        logger.info(f"New user {user_data.username} registered successfully")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user_info": user_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get current authenticated user's information.
    """
    return current_user

@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    current_user: UserResponse = Depends(get_current_active_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get detailed user profile with latest information.
    """
    try:
        # Get fresh user data from database
        user = await db[Collections.USERS].find_one({"username": current_user.username})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        return UserResponse(
            username=user["username"],
            email=user["email"],
            name=user["name"],
            credits=user.get("credits", 0),
            plan=user.get("plan", "Free Plan"),
            created_at=user["created_at"],
            total_predictions=user.get("total_predictions", 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch profile"
        )

@router.post("/logout")
@limiter.limit("10/minute")
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    SECURE logout with token blacklisting.
    """
    try:
        # Add token to blacklist for security
        add_token_to_blacklist(credentials.credentials)
        logger.info(f"User {current_user.username} logged out - token revoked")
        return {
            "message": "Successfully logged out",
            "detail": "Token has been revoked"
        }

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )
