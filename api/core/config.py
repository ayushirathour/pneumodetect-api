"""
Application configuration for PneumoDetectAI.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings with JWT authentication."""
    
    # Database settings (existing)
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DB_NAME: str = os.getenv("DB_NAME", "pneumodetect_db")
    
    # JWT Authentication settings (NEW)
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Admin settings
    ADMIN_SECRET_KEY: str = os.getenv("ADMIN_SECRET_KEY", "PNEUMO_ADMIN_2025")
    CREDIT_PRICE_INR: float = float(os.getenv("CREDIT_PRICE_INR", "10"))
    
    # Application settings (existing)
    APP_NAME: str = "PneumoDetectAI"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model settings (existing)
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_chest_xray_model.h5")
    MODEL_VERSION: str = "v2.0"
    MODEL_ACCURACY: float = 86.0
    MODEL_SENSITIVITY: float = 96.4
    MODEL_SPECIFICITY: float = 74.8
    
    def validate(self):
        """Validate required settings."""
        required = ["MONGODB_URI", "SECRET_KEY"]
        missing = [field for field in required if not getattr(self, field)]
        if missing:
            raise ValueError(f"Missing required settings: {missing}")

settings = Settings()

