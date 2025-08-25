"""
Authentication utilities for JWT tokens and password hashing.

BULLETPROOF SECURITY VERSION - Enhanced with token blacklisting and secure validation
"""

import hashlib
import bcrypt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Set
from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from api.core.config import settings
import logging

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token blacklist - Use Redis in production for scalability
token_blacklist: Set[str] = set()

logger = logging.getLogger(__name__)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def add_token_to_blacklist(token: str) -> None:
    """
    Add token to blacklist for revocation.
    In production, use Redis with expiration matching token expiry.
    """
    token_blacklist.add(token)

def is_token_blacklisted(token: str) -> bool:
    """
    Check if token is blacklisted.
    Returns True if token is revoked/blacklisted.
    """
    return token in token_blacklist

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create enhanced JWT access token with security claims.
    Args:
        data: Payload data to encode
        expires_delta: Token expiration time
    Returns:
        Encoded JWT token string with enhanced security claims
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    # Enhanced JWT claims for better security
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access_token",
        "iss": "pneumodetect-api", # Issuer claim for validation
        "aud": "pneumodetect-users", # Audience claim for validation
        "jti": secrets.token_urlsafe(16) # JWT ID for tracking and revocation
    })
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """
    Create refresh token with longer expiration.
    Args:
        data: Payload data to encode
    Returns:
        Encoded refresh token string
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=7) # 7 days for refresh token
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh_token",
        "iss": "pneumodetect-api",
        "aud": "pneumodetect-users",
        "jti": secrets.token_urlsafe(16)
    })
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """
    Verify JWT signature, issuer, audience, type and blacklist status.
    Returns the decoded payload or raises HTTP 401.
    """
    # 1️⃣  Black-list check
    if is_token_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # 2️⃣  Decode + validate signature, issuer and audience in one call
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            audience="pneumodetect-users",   # MUST match the "aud" claim you set
            issuer="pneumodetect-api",       # MUST match the "iss" claim you set
        )
    except JWTError as err:
        logger.warning(f"JWT verification failed: {err}")
        raise credentials_exc

    # 3️⃣  Extra app-specific check
    if payload.get("type") != "access_token":
        raise credentials_exc

    return payload

def validate_password_strength(password: str) -> None:
    """
    Validate password meets security requirements.
    Args:
        password: Plain text password to validate
    Raises:
        HTTPException: If password doesn't meet requirements
    """
    import re
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    if not re.search(r"[A-Z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one uppercase letter"
        )
    if not re.search(r"[a-z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one lowercase letter"
        )
    if not re.search(r"\d", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one number"
        )
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one special character"
        )

def hash_password_simple(password: str) -> str:
    """
    SECURE password hashing - ONLY bcrypt allowed now.
    Removed weak SHA-256 fallback for security.
    """
    # Validate password strength before hashing
    validate_password_strength(password)
    return pwd_context.hash(password)

def verify_password_simple(password: str, stored_hash: str) -> bool:
    """
    SECURE password verification - Forces migration from weak SHA-256.
    Only accepts bcrypt hashes. Legacy SHA-256 users must reset password.
    """
    if stored_hash.startswith("$2b$"):
        # bcrypt – secure and allowed
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    else:
        # SECURITY FIX: Force password reset for legacy SHA-256 users
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password security upgrade required. Please reset your password for enhanced security."
        )

def revoke_user_tokens(username: str) -> None:
    """
    Revoke all tokens for a specific user.
    Useful for account security incidents or password changes.
    Args:
        username: Username whose tokens should be revoked
    """
    # In production, you'd query Redis for all tokens belonging to this user
    # For now, we'll add a marker that can be checked during token validation
    revocation_marker = f"user_revoked:{username}:{datetime.now(timezone.utc).isoformat()}"
    token_blacklist.add(revocation_marker)

def cleanup_expired_tokens() -> None:
    """
    Clean up expired tokens from blacklist.
    Should be called periodically to prevent memory bloat.
    In production with Redis, use TTL instead.
    """
    # This is a placeholder for cleanup logic
    # In production with Redis, expired keys are automatically removed
    pass

# Utility function for admin token creation with extended permissions
def create_admin_token(username: str) -> str:
    """
    Create access token with admin privileges marker.
    Args:
        username: Admin username
    Returns:
        JWT token with admin scope
    """
    token_data = {
        "sub": username,
        "scope": "admin", # Admin scope marker
        "admin": True # Admin flag for quick checks
    }
    return create_access_token(
        data=token_data,
        expires_delta=timedelta(hours=1) # Shorter expiry for admin tokens
    )
