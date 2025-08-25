"""
Admin dashboard models and schemas
"""

from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    USER = "user"
    DEMO = "demo"
    ADMIN = "admin"

class UserOut(BaseModel):
    """User data for admin dashboard"""
    username: str
    email: EmailStr
    name: str
    credits: int
    plan: str
    role: str = "user"
    total_predictions: int
    credits_used: int
    created_at: datetime
    last_login: Optional[datetime]
    last_prediction: Optional[datetime]
    is_blocked: bool = False
    is_demo_user: bool = False

class UserListResponse(BaseModel):
    users: List[UserOut]
    total_count: int
    demo_users_count: int
    blocked_users_count: int

class CreditOperation(BaseModel):
    """Credit management operations"""
    username: str
    credits_to_add: int  # Can be negative for subtraction
    reason: Optional[str] = None
    operation_type: str = "manual_adjustment"

class CreditLog(BaseModel):
    """Credit transaction log"""
    username: str
    credits_changed: int
    previous_balance: int
    new_balance: int
    reason: Optional[str]
    operation_type: str
    admin_username: str
    timestamp: datetime

class DemoUserRequest(BaseModel):
    """Create demo user for recruiters/business"""
    username: str
    email: EmailStr
    name: str
    company: Optional[str] = None
    purpose: str = "demo"
    admin_secret: str

class SystemStats(BaseModel):
    """System analytics"""
    total_users: int
    active_users_30d: int
    demo_users: int
    total_predictions: int
    predictions_today: int
    total_credits_sold: int
    revenue_estimate: float
    avg_predictions_per_user: float

class UserActivity(BaseModel):
    """User activity tracking"""
    username: str
    last_login: Optional[datetime]
    login_count: int
    prediction_count: int
    credits_purchased: int
    last_prediction: Optional[datetime]
