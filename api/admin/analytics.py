"""
Admin analytics and reporting
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime, timedelta
from api.core.database import Collections
from api.core.config import settings

async def get_system_analytics(db: AsyncIOMotorDatabase):
    """
    Generate comprehensive system analytics
    """
    # Basic counts
    total_users = await db[Collections.USERS].count_documents({})
    
    # Active users (logged in last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    active_users = await db[Collections.USERS].count_documents({
        "last_login": {"$gte": thirty_days_ago}
    })
    
    # Demo users
    demo_users = await db[Collections.USERS].count_documents({
        "$or": [
            {"username": {"$regex": "^(demo|test)", "$options": "i"}},
            {"email": {"$regex": "^demo", "$options": "i"}},
            {"is_demo_user": True}
        ]
    })
    
    # Predictions
    total_predictions = await db["predictions"].count_documents({})
    
    # Today's predictions
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    predictions_today = await db["predictions"].count_documents({
        "timestamp": {"$gte": today_start}
    })
    
    # Calculate averages
    avg_predictions_per_user = total_predictions / total_users if total_users > 0 else 0
    
    # Revenue estimate (credits sold)
    # This would be more sophisticated with actual payment tracking
    total_credits_sold = await db[Collections.USERS].aggregate([
        {"$group": {"_id": None, "total": {"$sum": "$credits_used"}}}
    ]).to_list(1)
    
    credits_sold = total_credits_sold[0]["total"] if total_credits_sold else 0
    revenue_estimate = credits_sold * settings.CREDIT_PRICE_INR  
    
    return {
        "total_users": total_users,
        "active_users_30d": active_users,
        "demo_users": demo_users,
        "total_predictions": total_predictions,
        "predictions_today": predictions_today,
        "total_credits_sold": credits_sold,
        "revenue_estimate": revenue_estimate,
        "avg_predictions_per_user": round(avg_predictions_per_user, 2)
    }
