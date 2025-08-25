"""
Database connection management.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from fastapi import HTTPException
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
from api.core.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """MongoDB connection manager using your existing database."""
    
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database: AsyncIOMotorDatabase = None
    
    async def connect_to_mongo(self):
        """Connect to your existing MongoDB Atlas database."""
        try:
            logger.info("Connecting to your existing MongoDB Atlas (pneumodetect_db)...")
            
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                maxPoolSize=10,
                serverSelectionTimeoutMS=5000,
                retryWrites=True
            )
            
            # Test connection
            await self.client.admin.command('ping')
            self.database = self.client[settings.DB_NAME]
            
            logger.info(f"✅ Connected to MongoDB: {settings.DB_NAME}")
            
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed"
            )
    
    async def close_mongo_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")
    
    async def ping_database(self) -> bool:
        """Health check for database."""
        try:
            if not self.client:
                return False
            await self.client.admin.command('ping')
            return True
        except Exception:
            return False

# Global database manager
db_manager = DatabaseManager()

# Dependency for FastAPI routes
async def get_database() -> AsyncIOMotorDatabase:
    """Database dependency for FastAPI routes."""
    if not await db_manager.ping_database():
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable"
        )
    return db_manager.database

# Collection names (matching your Streamlit app exactly)
class Collections:
    """Database collection names from your Streamlit app."""
    USERS = "users"           # Your existing users collection
    PREDICTIONS = "predictions"  # Your existing predictions collection
    # Future collections for FastAPI
    API_KEYS = "api_keys"
    ADMIN_LOGS = "admin_logs"
    PAYMENTS = "payments"
