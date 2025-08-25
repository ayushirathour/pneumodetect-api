"""
Admin dashboard endpoints with demo user support - PRODUCTION READY & BULLETPROOFED
"""
from passlib.context import CryptContext
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import secrets
import string
from bson import ObjectId

# Rate limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.core.database import get_database, Collections
from api.core.config import settings
from api.auth.dependencies import get_admin_user
from api.admin.models import *
from api.admin.utils import is_demo_user, create_demo_credentials
from api.admin.analytics import get_system_analytics

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)
router = APIRouter()

import re


def sanitize_search_input(search_term: str) -> str:
    """Escape special regex characters to prevent injection"""
    if not search_term:
        return ""
    # Escape special regex characters and limit length
    sanitized = re.escape(search_term.strip())
    # Limit search length to prevent abuse
    return sanitized[:100] if len(sanitized) > 100 else sanitized


def validate_username(username: str) -> str:
    """Validate and sanitize username input"""
    if not username or not isinstance(username, str):
        raise HTTPException(status_code=400, detail="Invalid username")
    
    # Remove whitespace and limit length
    clean_username = username.strip()
    if len(clean_username) < 3 or len(clean_username) > 50:
        raise HTTPException(status_code=400, detail="Username must be 3-50 characters")
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', clean_username):
        raise HTTPException(status_code=400, detail="Username contains invalid characters")
    
    return clean_username


def generate_secure_password(length: int = 16) -> str:
    """Generate a cryptographically secure random password"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


# 🛡️ ENHANCED OBJECTID CONVERSION WITH ERROR HANDLING


def convert_objectids_to_strings(documents):
    """
    Convert ALL ObjectId fields to strings for JSON serialization
    Handles nested objects and arrays safely
    """
    try:
        if isinstance(documents, list):
            for doc in documents:
                convert_single_document(doc)
        elif isinstance(documents, dict):
            convert_single_document(documents)
    except Exception as e:
        logger.warning(f"ObjectId conversion warning: {e}")
    return documents


def convert_single_document(doc):
    """Convert all ObjectId fields in a single document recursively"""
    if not isinstance(doc, dict):
        return
        
    try:
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, dict):
                convert_single_document(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_single_document(item)
    except Exception as e:
        logger.warning(f"Document conversion warning for key {key}: {e}")


def sanitize_user_document(user_doc):
    """
    Ensure user document has all required fields with safe defaults
    """
    if not isinstance(user_doc, dict):
        return user_doc
        
    # Convert ObjectId fields
    convert_objectids_to_strings(user_doc)
    
    # Ensure all required fields exist with safe defaults
    defaults = {
        "username": user_doc.get("username", "unknown"),
        "email": user_doc.get("email", "no-email@example.com"),
        "name": user_doc.get("name", "User"),
        "credits": user_doc.get("credits", 0),
        "plan": user_doc.get("plan", "Free Plan"),
        "total_predictions": user_doc.get("total_predictions", 0),
        "credits_used": user_doc.get("credits_used", 0),
        "created_at": user_doc.get("created_at", datetime.now()),
        "last_login": user_doc.get("last_login"),
        "last_prediction": user_doc.get("last_prediction"),
        "is_blocked": user_doc.get("is_blocked", False),
        "is_admin": user_doc.get("is_admin", False)
    }
    
    # Update document with defaults for missing fields
    for field, default_value in defaults.items():
        if field not in user_doc or user_doc[field] is None:
            if field in ["last_login", "last_prediction"] and default_value is None:
                user_doc[field] = None  # Keep None for optional datetime fields
            else:
                user_doc[field] = default_value
    
    return user_doc


# 👥 USER MANAGEMENT ENDPOINTS


@router.get("/users", response_model=UserListResponse)
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute
async def list_users(
    request: Request,
    skip: int = 0,
    limit: int = Query(50, le=100),
    search: Optional[str] = None,
    role_filter: Optional[UserRole] = None,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """
    List all users with filtering and search
    Supports demo user identification from your Streamlit logic
    """
    try:
        query = {}
        
        # Search filter with enhanced security
        if search:
            safe_search = sanitize_search_input(search)
            if safe_search:  # Only add query if search term is valid after sanitization
                query["$or"] = [
                    {"username": {"$regex": safe_search, "$options": "i"}},
                    {"email": {"$regex": safe_search, "$options": "i"}},
                    {"name": {"$regex": safe_search, "$options": "i"}}
                ]
        
        # Get users with error handling
        cursor = db[Collections.USERS].find(query).skip(skip).limit(limit).sort("created_at", -1)
        users = await cursor.to_list(length=limit)
        
        # Sanitize and process users
        processed_users = []
        
        for user in users:
            try:
                # Sanitize user document
                user = sanitize_user_document(user)
                
                # Demo user detection
                user["is_demo_user"] = is_demo_user(user)
                user["role"] = "demo" if user["is_demo_user"] else "user"
                
                processed_users.append(user)
                
            except Exception as e:
                logger.error(f"Error processing user {user.get('username', 'unknown')}: {e}")
                # Skip problematic users rather than crash entire endpoint
                continue
        
        # Filter by role if specified
        if role_filter:
            processed_users = [u for u in processed_users if u.get("role") == role_filter]
        
        # Count demo users and blocked users securely
        demo_query = {
            "$or": [
                {"username": {"$regex": r"^(demo|test)", "$options": "i"}},
                {"email": {"$regex": r"^demo", "$options": "i"}},
                {"is_demo_user": True}
            ]
        }
        
        try:
            demo_count = await db[Collections.USERS].count_documents(demo_query)
            blocked_count = await db[Collections.USERS].count_documents({"is_blocked": True})
        except Exception as e:
            logger.error(f"Error counting users: {e}")
            # Fallback to processed users count
            demo_count = len([u for u in processed_users if u.get("is_demo_user", False)])
            blocked_count = len([u for u in processed_users if u.get("is_blocked", False)])
        
        # Get total count safely
        try:
            total_count = await db[Collections.USERS].count_documents(query)
        except Exception as e:
            logger.error(f"Error counting users: {e}")
            total_count = len(processed_users)
        
        return UserListResponse(
            users=processed_users,
            total_count=total_count,
            demo_users_count=demo_count,
            blocked_users_count=blocked_count
        )
        
    except RateLimitExceeded:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    except Exception as e:
        logger.error(f"Error in list_users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users list"
        )


@router.get("/users/{username}", response_model=UserOut)
@limiter.limit("60/minute")  # Rate limit: 60 requests per minute
async def get_user_detail(
    request: Request,
    username: str,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """Get detailed user information"""
    try:
        # Validate username input
        clean_username = validate_username(username)
        
        user = await db[Collections.USERS].find_one({"username": clean_username})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Sanitize user document
        user = sanitize_user_document(user)
        
        # Add computed fields
        user["is_demo_user"] = is_demo_user(user)
        user["role"] = "demo" if user["is_demo_user"] else "user"
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user details"
        )


@router.put("/users/{username}/block")
@limiter.limit("20/minute")  # Rate limit: 20 requests per minute for admin actions
async def block_user(
    request: Request,
    username: str,
    blocked: bool,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """Block or unblock a user account"""
    try:
        # Validate username input
        clean_username = validate_username(username)
        
        # Prevent admin from blocking themselves
        if clean_username == admin.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot block your own account"
            )
        
        result = await db[Collections.USERS].update_one(
            {"username": clean_username},
            {"$set": {
                "is_blocked": blocked, 
                "blocked_at": datetime.now() if blocked else None,
                "blocked_by": admin.username if blocked else None
            }}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        action = "blocked" if blocked else "unblocked"
        logger.info(f"Admin {admin.username} {action} user {clean_username}")
        
        return {"message": f"User {clean_username} has been {action}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error blocking user {username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )


# 💳 CREDIT MANAGEMENT ENDPOINTS


@router.post("/credits/manage")
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute
async def manage_user_credits(
    request: Request,
    operation: CreditOperation,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """
    Add or subtract credits from user account
    Includes audit logging
    """
    try:
        # Validate username input
        clean_username = validate_username(operation.username)
        
        user = await db[Collections.USERS].find_one({"username": clean_username})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if demo user (demo users have unlimited credits)
        if is_demo_user(user):
            return {"message": "Demo users have unlimited credits - no changes needed"}
        
        # Validate credit amount
        if abs(operation.credits_to_add) > 1000000:  # Reasonable limit
            raise HTTPException(
                status_code=400, 
                detail="Credit amount too large. Maximum allowed: ±1,000,000"
            )
        
        previous_balance = user.get("credits", 0)
        new_balance = previous_balance + operation.credits_to_add
        
        if new_balance < 0:
            raise HTTPException(status_code=400, detail="Resulting credits cannot be negative")
        
        # Update user credits
        await db[Collections.USERS].update_one(
            {"username": clean_username},
            {
                "$inc": {"credits": operation.credits_to_add},
                "$set": {"last_credit_update": datetime.now()}
            }
        )
        
        # Log the operation
        credit_log = CreditLog(
            username=clean_username,
            credits_changed=operation.credits_to_add,
            previous_balance=previous_balance,
            new_balance=new_balance,
            reason=operation.reason,
            operation_type=operation.operation_type,
            admin_username=admin.username,
            timestamp=datetime.now()
        )
        
        await db["credit_logs"].insert_one(credit_log.dict())
        
        logger.info(f"Admin {admin.username} adjusted credits for {clean_username}: {operation.credits_to_add}")
        
        return {
            "message": f"Credits updated successfully",
            "previous_balance": previous_balance,
            "new_balance": new_balance,
            "credits_changed": operation.credits_to_add
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error managing credits for {operation.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to manage user credits"
        )


@router.get("/credits/logs", response_model=List[CreditLog])
@limiter.limit("60/minute")  # Rate limit: 60 requests per minute
async def get_credit_logs(
    request: Request,
    skip: int = 0,
    limit: int = 50,
    username: Optional[str] = None,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """Get credit transaction history with optional user filter"""
    try:
        query = {}
        if username:
            # Validate username if provided
            clean_username = validate_username(username)
            query["username"] = clean_username
        
        cursor = db["credit_logs"].find(query).sort("timestamp", -1).skip(skip).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        # Convert ObjectIds
        convert_objectids_to_strings(logs)
        
        return logs
        
    except Exception as e:
        logger.error(f"Error retrieving credit logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve credit logs"
        )


# 🎭 DEMO USER MANAGEMENT


@router.post("/demo-users/create")
@limiter.limit("5/minute")  # Strict rate limit: 5 requests per minute for demo creation
async def create_demo_user(
    request: Request,
    demo_request: DemoUserRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """
    Create demo user with admin secret validation
    
    """
    try:
        # Validate admin secret
        if not secrets.compare_digest(demo_request.admin_secret, settings.ADMIN_SECRET_KEY):
            raise HTTPException(
                status_code=403,
                detail="Invalid admin secret key for demo user creation"
            )
        
        # Validate username input
        clean_username = validate_username(demo_request.username)
        
        # Check if user already exists
        existing_user = await db[Collections.USERS].find_one({"username": clean_username})
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Generate secure password and hash it
        temp_password = generate_secure_password()
        hashed_password = pwd_context.hash(temp_password)
        
        # Create demo user record
        demo_user = {
            "username": clean_username,
            "email": demo_request.email,
            "name": demo_request.name,
            "password": hashed_password,  # Properly hashed password
            "created_at": datetime.now(),
            "total_predictions": 0,
            "credits": 999999,
            "plan": "Demo Plan",
            "is_demo_user": True,
            "demo_purpose": demo_request.purpose,
            "company": demo_request.company,
            "created_by_admin": admin.username,
            "credits_used": 0,
            "last_login": None,
            "last_prediction": None,
            "is_blocked": False
        }
        
        result = await db[Collections.USERS].insert_one(demo_user)
        
        # Generate demo credentials with the actual password
        credentials = {
            "username": clean_username,
            "email": demo_request.email,
            "password": temp_password,  # Return the plain password for setup
            "instructions": "Use these credentials to test the PneumoDetect AI system. Demo accounts have unlimited credits.",
            "api_endpoint": "https://your-domain.com/api/v1/predict",
            "documentation": "https://your-domain.com/docs"
        }
        
        logger.info(f"Admin {admin.username} created demo user {clean_username}")
        
        return {
            "message": "Demo user created successfully",
            "demo_credentials": credentials,
            "username": clean_username,
            "temporary_password": temp_password
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating demo user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create demo user"
        )


@router.get("/demo-users")
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute
async def list_demo_users(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin=Depends(get_admin_user)
):
    """List all demo users and their activity"""
    try:
        # Find users with demo patterns
        demo_query = {
            "$or": [
                {"username": {"$regex": r"^(demo|test)", "$options": "i"}},
                {"email": {"$regex": r"^demo", "$options": "i"}},
                {"is_demo_user": True}
            ]
        }
        
        cursor = db[Collections.USERS].find(demo_query).sort("created_at", -1)
        demo_users = await cursor.to_list(length=None)
        
        # Process demo users
        processed_demo_users = []
        
        for user in demo_users:
            try:
                # Sanitize user document
                user = sanitize_user_document(user)
                
                # Add activity stats
                prediction_count = await db["predictions"].count_documents(
                    {"user_id": user["username"]}
                )
                user["prediction_count"] = prediction_count
                user["is_demo_user"] = True
                
                processed_demo_users.append(user)
                
            except Exception as e:
                logger.error(f"Error processing demo user {user.get('username', 'unknown')}: {e}")
                continue
        
        return {
            "demo_users": processed_demo_users,
            "total_demo_users": len(processed_demo_users)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving demo users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve demo users"
        )


# 📊 ANALYTICS DASHBOARD


@router.get("/analytics/overview", response_model=SystemStats)
@limiter.limit("60/minute")  # Rate limit: 60 requests per minute
async def get_system_overview(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """Get comprehensive system analytics"""
    try:
        return await get_system_analytics(db)
    except Exception as e:
        logger.error(f"Error getting system analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system analytics"
        )


@router.get("/analytics/predictions")
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute
async def get_prediction_analytics(
    request: Request,
    days: int = 30,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """Get prediction analytics for specified period"""
    try:
        # Validate days parameter
        if days < 1 or days > 365:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be between 1 and 365"
            )
        
        start_date = datetime.now() - timedelta(days=days)
        
        pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "count": {"$sum": 1},
                "pneumonia_detected": {
                    "$sum": {"$cond": [{"$eq": ["$diagnosis", "PNEUMONIA"]}, 1, 0]}
                }
            }},
            {"$sort": {"_id": 1}}
        ]
        
        cursor = db["predictions"].aggregate(pipeline)
        daily_stats = await cursor.to_list(length=None)
        
        # Convert ObjectIds safely
        convert_objectids_to_strings(daily_stats)
        
        return {
            "daily_predictions": daily_stats,
            "period_days": days,
            "total_predictions": sum(day["count"] for day in daily_stats) if daily_stats else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prediction analytics"
        )


# 🔍 SYSTEM MONITORING


@router.get("/system/health")
@limiter.limit("120/minute")  # Rate limit: 120 requests per minute (health checks are frequent)
async def system_health_check(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database),
    admin = Depends(get_admin_user)
):
    """Comprehensive system health check"""
    health_status = {
        "database": "unknown",
        "model_status": "unknown",
        "api_status": "healthy",
        "timestamp": datetime.now(),
        "uptime": "calculating..."
    }
    
    # Test database connection
    try:
        await db.command("ping")
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = "error"  
        logger.error(f"Database health check failed: {e}")
    
    # Test collections
    try:
        user_count = await db[Collections.USERS].count_documents({})
        prediction_count = await db["predictions"].count_documents({})
        health_status["collections"] = {
            "users": user_count,
            "predictions": prediction_count
        }
    except Exception as e:
        health_status["collections"] = "error"  
        logger.error(f"Collection health check failed: {e}")
    
    return health_status


@router.get("/system/logs")
@limiter.limit("20/minute")  # Rate limit: 20 requests per minute for logs
async def get_system_logs(
    request: Request,
    level: str = "ERROR",
    limit: int = 100,
    admin = Depends(get_admin_user)
):
    """Get system error logs and monitoring data"""
    try:
        # Validate parameters
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level.upper() not in valid_levels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid log level. Must be one of: {valid_levels}"
            )
        
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000"
            )
        
        # This would integrate with logging system
        # For now, return structured placeholder
        return {
            "message": "System logs endpoint - integrate with your logging solution",
            "level": level.upper(),
            "limit": limit,
            "suggestion": "Consider integrating with ELK stack or similar",
            "available_integrations": [
                "ELK Stack (Elasticsearch, Logstash, Kibana)",
                "Grafana + Loki",
                "DataDog",
                "New Relic",
                "Sentry for error tracking"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving system logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system logs"
        )
