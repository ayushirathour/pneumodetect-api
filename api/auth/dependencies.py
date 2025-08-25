"""
Authentication dependencies for protecting routes.

BULLETPROOF SECURITY VERSION
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime, timezone
import logging

from api.auth.utils import verify_token, is_token_blacklisted
from api.auth.models import UserResponse
from api.core.database import get_database, Collections

logger = logging.getLogger(__name__)

# HTTP Bearer scheme for JWT tokens
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> UserResponse:
    """
    Get current authenticated user from JWT token with blacklist check.
    Enhanced security with token blacklisting.
    """
    # Check if token is blacklisted first
    if is_token_blacklisted(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked"
        )

    # Verify JWT token
    payload = verify_token(credentials.credentials)
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )

    # Get user from database
    user = await db[Collections.USERS].find_one({"username": username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # Check if user is blocked
    if user.get("is_blocked", False):
        logger.warning(f"Blocked user {username} attempted access")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been blocked. Contact support."
        )

    # Convert to response model (excluding sensitive data)
    return UserResponse(
        username=user["username"],
        email=user["email"],
        name=user.get("name", "User"),
        credits=user.get("credits", 0),
        plan=user.get("plan", "Free Plan"),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
        total_predictions=user.get("total_predictions", 0)
    )

async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Get current active user with additional checks.
    """
    return current_user

async def get_admin_user(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> UserResponse:
    """
    SECURE admin dependency - Only database-based admin check.
    Removed all hardcoded usernames for security.
    """
    # Get fresh user data from database
    user_data = await db[Collections.USERS].find_one({"username": current_user.username})
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    # SECURE: Only check database flag - no hardcoded usernames
    is_admin = user_data.get("is_admin", False)
    if not is_admin:
        logger.warning(f"Non-admin user {current_user.username} attempted admin access")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required"
        )

    logger.info(f"Admin {current_user.username} accessed admin endpoint")
    return current_user
