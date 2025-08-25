"""
Batch processing models for pneumonia detection
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class BatchFileResult(BaseModel):
    """Result for a single file in batch"""
    filename: str
    file_index: int
    success: bool
    diagnosis: Optional[str] = None
    confidence: Optional[float] = None
    confidence_level: Optional[str] = None
    recommendation: Optional[str] = None
    raw_score: Optional[float] = None
    file_type: Optional[str] = None  # "IMAGE" or "DICOM"
    dicom_metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None

class BatchSummary(BaseModel):
    """Summary statistics for batch processing"""
    total_files: int
    successful_predictions: int
    failed_predictions: int
    pneumonia_detected: int
    normal_cases: int
    average_confidence: float
    processing_time_seconds: float
    credits_consumed: int

class BatchResponse(BaseModel):
    """Complete batch processing response"""
    batch_id: str
    status: BatchStatus
    user: str
    timestamp: datetime
    summary: BatchSummary
    results: List[BatchFileResult] 
    credits_remaining: int
    cross_operator_validation_performance: Dict[str, str]
    disclaimer: str
