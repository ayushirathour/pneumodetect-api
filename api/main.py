"""
FastAPI application entry point with JWT authentication and Admin Dashboard.
"""


from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime

# Rate limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.core.config import settings



from api.core.database import db_manager, get_database
from api.predictions.router import router as predictions_router, load_model
from api.auth.router import router as auth_router
from api.admin.router import router as admin_router # NEW: Admin dashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create rate limiter instance
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("🚀 Starting PneumoDetectAI with authentication and admin dashboard...")
    try:
        await db_manager.connect_to_mongo()
        await load_model()
        logger.info("✅ Application startup completed successfully!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down PneumoDetectAI...")
    await db_manager.close_mongo_connection()
    logger.info("✅ Application shutdown completed")

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="Clinical-grade AI pneumonia screening with JWT authentication and admin dashboard",
    version=settings.VERSION,
    lifespan=lifespan
)

# IMPORTANT: Add rate limiter to app state
app.state.limiter = limiter

# Add rate limit exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 🆕 ADD THIS LINE RIGHT HERE - Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "https://*.vercel.app",
        "https://*.netlify.app",
        "https://*.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    auth_router,
    prefix="/auth", 
    tags=["authentication"]
)

app.include_router(
    predictions_router,
    prefix="/api/v1",
    tags=["predictions"]
)

app.include_router(
    admin_router,
    prefix="/admin",
    tags=["admin"]
)

# Root endpoint
@app.get("/")
def read_root():
    """API root endpoint with authentication and admin info."""
    return {
        "message": f"🏥 {settings.APP_NAME} API",
        "version": settings.VERSION,
        "status": "running",
        "description": "AI-powered pneumonia detection with JWT authentication and admin dashboard",
        "database": settings.DB_NAME,
        "authentication": "JWT Bearer tokens",
        "endpoints": {
            "login": "/auth/login - User authentication",
            "register": "/auth/register - User registration", 
            "profile": "/auth/me - Get user profile (protected)",
            "predict": "/api/v1/predict - Upload chest X-ray for analysis (protected)",
            "health": "/api/v1/health - Check prediction service health",
            "admin_users": "/admin/users - Admin: List all users",
            "admin_analytics": "/admin/analytics/overview - Admin: System analytics",
            "docs": "/docs - Interactive API documentation"
        }
    }

# Database health check
@app.get("/health/database")
async def check_database_health(db = Depends(get_database)):
    """Check connection to MongoDB Atlas database."""
    try:
        collections = await db.list_collection_names()
        users_count = await db.users.estimated_document_count() if "users" in collections else 0
        predictions_count = await db.predictions.estimated_document_count() if "predictions" in collections else 0
        
        return {
            "status": "healthy",
            "database": "connected", 
            "database_name": db.name,
            "timestamp": datetime.now().isoformat(),
            "collections": {
                "available": collections,
                "users_count": users_count,
                "predictions_count": predictions_count,
                "total_collections": len(collections)
            },
            "message": "Successfully connected to MongoDB Atlas database"
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database health check failed")

# Global health check
@app.get("/health")
def global_health_check():
    """Overall application health check."""
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "version": settings.VERSION,
        "database": "connected",
        "authentication": "JWT enabled", 
        "admin_dashboard": "enabled",
        "rate_limiting": "enabled",
        "services": {
            "predictions": "/api/v1/health",
            "database": "/health/database",
            "auth": "/auth/me",
            "admin": "/admin/analytics/overview"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=7860,
        reload=settings.DEBUG
    )
