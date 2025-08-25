"""
Authentication models for requests and responses.
PYDANTIC V1 COMPATIBLE VERSION
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime
import re

class UserLogin(BaseModel):
    """User login request model with enhanced validation."""
    username: str = Field(..., min_length=3, max_length=50, description="Username or email")
    password: str = Field(..., min_length=8, description="User password")
    
    @validator('username')
    @classmethod
    def validate_username(cls, v):
        if '@' in v:
            return v
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must contain only letters, numbers, underscores, and hyphens')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe", 
                "password": "SecurePass123!"
            }
        }

class UserRegister(BaseModel):
    """User registration request model with strong validation."""
    username: str = Field(..., min_length=3, max_length=30, description="Unique username")
    email: EmailStr = Field(..., description="Valid email address")
    name: str = Field(..., min_length=2, max_length=100, description="Full name")
    password: str = Field(..., min_length=8, description="Strong password")
    
    @validator('username')
    @classmethod
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must contain only letters, numbers, underscores, and hyphens')
        
        forbidden_usernames = ['admin', 'administrator', 'root', 'system', 'api', 'test', 'demo']
        if v.lower() in forbidden_usernames:
            raise ValueError('This username is not allowed')
        
        return v.lower()
    
    @validator('name')
    @classmethod
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z\s]+$', v.strip()):
            raise ValueError('Name must contain only letters and spaces')
        return v.strip().title()
    
    @validator('password')
    @classmethod
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain special character')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com", 
                "name": "John Doe",
                "password": "SecurePass123!"
            }
        }

class Token(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: dict

class TokenData(BaseModel):
    """JWT token data model."""
    username: Optional[str] = None
    exp: Optional[datetime] = None

class UserResponse(BaseModel):
    """User response model (without sensitive data)."""
    username: str
    email: str
    name: str
    credits: int
    plan: str
    created_at: datetime
    total_predictions: int
    
    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "name": "John Doe", 
                "credits": 5,
                "plan": "Free Plan",
                "created_at": "2025-08-24T15:30:00Z",
                "total_predictions": 12
            }
        }

class UserInDB(UserResponse):
    """User model as stored in database."""
    password: str
    last_login: Optional[datetime] = None
    is_blocked: bool = False
    is_admin: bool = False
