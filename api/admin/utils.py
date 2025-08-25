"""
Admin utility functions
Implements demo user logic from your Streamlit app
"""

import secrets
import string
from typing import Dict, Any

def is_demo_user(user: Dict[str, Any]) -> bool:
    """
    Detect if user is a demo user using your Streamlit logic
    """
    username = user.get("username", "").lower()
    email = user.get("email", "").lower()
    
    # Demo user patterns from your Streamlit app
    demo_usernames = ['demo', 'demouser', 'test']
    
    is_demo_username = username in demo_usernames
    is_demo_email = email.startswith('demo')
    is_marked_demo = user.get("is_demo_user", False)
    
    return is_demo_username or is_demo_email or is_marked_demo

def create_demo_credentials(username: str, email: str) -> Dict[str, str]:
    """
    Generate secure demo credentials for recruiters/business
    """
    # Generate random password
    password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
    
    return {
        "username": username,
        "email": email,
        "password": password,
        "instructions": "Use these credentials to test the PneumoDetect AI system. Demo accounts have unlimited credits.",
        "api_endpoint": "https://your-domain.com/api/v1/predict",
        "documentation": "https://your-domain.com/docs"
    }

def generate_admin_report(data: Dict[str, Any]) -> str:
    """
    Generate admin summary report
    """
    report = f"""
    PneumoDetect AI - Admin Dashboard Report
    Generated: {data.get('timestamp', 'Unknown')}
    
    USER STATISTICS:
    - Total Users: {data.get('total_users', 0)}
    - Active Users (30d): {data.get('active_users', 0)}
    - Demo Users: {data.get('demo_users', 0)}
    
    USAGE STATISTICS:
    - Total Predictions: {data.get('total_predictions', 0)}
    - Predictions Today: {data.get('predictions_today', 0)}
    - Average per User: {data.get('avg_predictions', 0)}
    
    SYSTEM HEALTH:
    - Database: {data.get('database_status', 'Unknown')}
    - API Status: {data.get('api_status', 'Unknown')}
    """
    
    return report
