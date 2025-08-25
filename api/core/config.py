"""
Application configuration for PneumoDetectAI.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings with JWT authentication and Hugging Face integration."""
    
    # Database settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DB_NAME: str = os.getenv("DB_NAME", "pneumodetect_db")
    
    # JWT Authentication settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Hugging Face API settings
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_MODEL_URL: str = os.getenv("HF_MODEL_URL", "")
    HF_PREDICT_ENDPOINT: str = os.getenv("HF_PREDICT_ENDPOINT", "/predict")
    HF_HEALTH_ENDPOINT: str = os.getenv("HF_HEALTH_ENDPOINT", "/health")
    HF_TIMEOUT_SECONDS: int = int(os.getenv("HF_TIMEOUT_SECONDS", "30"))
    
    # Admin settings
    ADMIN_SECRET_KEY: str = os.getenv("ADMIN_SECRET_KEY", "")
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")
    
    # Application settings
    APP_NAME: str = os.getenv("APP_NAME", "PneumoDetectAI")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    CREDIT_PRICE_INR: float = float(os.getenv("CREDIT_PRICE_INR", "10"))
    
    # Model performance metrics
    MODEL_VERSION: str = "v2.0"
    MODEL_ACCURACY: float = 86.0
    MODEL_SENSITIVITY: float = 96.4
    MODEL_SPECIFICITY: float = 74.8

    def validate(self):
        """Validate required settings."""
        required = [
            "MONGODB_URI", 
            "SECRET_KEY", 
            "HF_API_TOKEN", 
            "HF_MODEL_URL",
            "ADMIN_SECRET_KEY"
        ]
        missing = [field for field in required if not getattr(self, field)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

    @property
    def hf_full_predict_url(self) -> str:
        """Get full Hugging Face prediction URL."""
        return f"{self.HF_MODEL_URL.rstrip('/')}{self.HF_PREDICT_ENDPOINT}"
    
    @property
    def hf_full_health_url(self) -> str:
        """Get full Hugging Face health URL."""
        return f"{self.HF_MODEL_URL.rstrip('/')}{self.HF_HEALTH_ENDPOINT}"

settings = Settings()
