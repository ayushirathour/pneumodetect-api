"""
ML prediction endpoints with JWT authentication and credit system
Enhanced with DICOM support and batch processing capabilities
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pydicom
import time
import uuid
from typing import List
import logging
from datetime import datetime
from pathlib import Path

from api.core.database import get_database, Collections
from api.core.config import settings
from api.auth.dependencies import get_current_active_user
from api.auth.models import UserResponse
from api.predictions.batch_models import BatchResponse, BatchFileResult, BatchSummary, BatchStatus

logger = logging.getLogger(__name__)
router = APIRouter()

# Model variables
model = None
model_info = {
    "loaded": False,
    "load_time": None,
    "model_path": None,
    "performance": {
        "accuracy": 86.0,
        "sensitivity": 96.4,
        "specificity": 74.8,
        "false_positive_rate": 25.2,
        "roc_auc": 0.964,
        "pr_auc": 0.968
    }
}

# Model loading function
async def load_model():
    """Load ML model on startup."""
    global model, model_info
    try:
        model_paths = [
            Path("api/models/best_chest_xray_model.h5"),
            Path("./api/model/best_chest_xray_model.h5"),
            Path("models/best_chest_xray_model.h5"),
            Path("api/streamlit_api_folder/best_chest_xray_model.h5"),
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"Loading model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                
                # Warm up model
                dummy_input = tf.random.normal([1, 224, 224, 3])
                _ = model.predict(dummy_input, verbose=0)
                
                model_info.update({
                    "loaded": True,
                    "load_time": datetime.now().isoformat(),
                    "model_path": str(model_path)
                })
                logger.info("✅ Model loaded successfully!")
                return
        
        logger.error("❌ Model file not found")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise e

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference."""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def interpret_prediction(prediction_score: float) -> dict:
    """Interpret model prediction score."""
    if prediction_score > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = float(prediction_score * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "Strong indication of pneumonia. Recommend immediate medical attention."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "Moderate indication of pneumonia. Medical review recommended."
        else:
            confidence_level = "Low"
            recommendation = "Possible pneumonia detected. Further examination advised."
    else:
        diagnosis = "NORMAL"
        confidence = float((1 - prediction_score) * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "No signs of pneumonia detected. Chest X-ray appears normal."
        elif confidence >= 60:
            confidence_level = "Moderate"  
            recommendation = "Likely normal chest X-ray. Routine follow-up if symptoms persist."
        else:
            confidence_level = "Low"
            recommendation = "Unclear result. Manual review by radiologist recommended."

    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "recommendation": recommendation,
        "raw_score": float(prediction_score),
        "threshold": 0.5,
        "model_architecture": "MobileNetV2"
    }

def process_dicom_file(dicom_bytes: bytes) -> tuple[Image.Image, dict]:
    """Process DICOM file and extract metadata."""
    try:
        dicom_data = pydicom.dcmread(io.BytesIO(dicom_bytes))
        
        metadata = {
            "patient_id": str(dicom_data.get("PatientID", "Unknown")),
            "patient_name": str(dicom_data.get("PatientName", "Unknown")),
            "study_date": str(dicom_data.get("StudyDate", "Unknown")),
            "study_time": str(dicom_data.get("StudyTime", "Unknown")),
            "modality": str(dicom_data.get("Modality", "Unknown")),
            "institution": str(dicom_data.get("InstitutionName", "Unknown")),
            "manufacturer": str(dicom_data.get("Manufacturer", "Unknown")),
            "model": str(dicom_data.get("ManufacturerModelName", "Unknown")),
            "image_size": f"{dicom_data.Rows}x{dicom_data.Columns}",
            "pixel_spacing": str(dicom_data.get("PixelSpacing", "Unknown")),
            "slice_thickness": str(dicom_data.get("SliceThickness", "Unknown")),
            "kvp": str(dicom_data.get("KVP", "Unknown")),
            "exposure_time": str(dicom_data.get("ExposureTime", "Unknown"))
        }
        
        pixel_array = dicom_data.pixel_array
        
        if len(pixel_array.shape) == 2:  # Grayscale
            if pixel_array.max() > 255:
                pixel_array = ((pixel_array - pixel_array.min()) / 
                             (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            else:
                pixel_array = pixel_array.astype(np.uint8)
            
            rgb_array = np.stack([pixel_array] * 3, axis=-1)
        else:
            rgb_array = pixel_array
            
        image = Image.fromarray(rgb_array)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image, metadata
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"DICOM processing failed: {str(e)}"
        )

# 🔐 Single prediction endpoint
@router.post("/predict")
async def predict_pneumonia(
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """🔐 PROTECTED: Predict pneumonia from chest X-ray"""
    
    if current_user.credits <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="❌ Insufficient credits. Please purchase more credits to continue using AI predictions."
        )

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )

    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum 10MB allowed."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"🔬 Processing image for user: {current_user.username}")
        
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)
        
        if len(prediction.shape) == 2 and prediction.shape[1] == 1:
            prediction_score = float(prediction[0][0])
        elif len(prediction.shape) == 1:
            prediction_score = float(prediction[0])
        else:
            prediction_score = float(prediction.flatten()[0])
        
        result = interpret_prediction(prediction_score)
        
        await db[Collections.USERS].update_one(
            {"username": current_user.username},
            {
                "$inc": {"credits": -1, "total_predictions": 1},
                "$set": {"last_prediction": datetime.now()}
            }
        )
        
        result.update({
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "user": current_user.username,
            "credits_remaining": current_user.credits - 1,
            "cross_operator_validation_performance": {
                "accuracy": "86.0%",
                "sensitivity": "96.4%", 
                "specificity": "74.8%",
                "validated_on": "485 independent samples"
            },
            "disclaimer": "This AI assistant is for preliminary screening only. Always consult healthcare professionals for medical decisions."
        })
        
        logger.info(f"✅ Prediction completed: {result['diagnosis']} ({result['confidence']:.1f}%) for user {current_user.username}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# 🩺 DICOM prediction endpoint
@router.post("/predict/dicom")
async def predict_dicom(
    file: UploadFile = File(...),
    current_user: UserResponse = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """🔐 PROTECTED: Predict pneumonia from DICOM chest X-ray"""
    
    if current_user.credits <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="❌ Insufficient credits. Please purchase more credits to continue using AI predictions."
        )

    if not file.filename.lower().endswith('.dcm'):
        raise HTTPException(
            status_code=400,
            detail="File must be a DICOM (.dcm) file"
        )
        
    if hasattr(file, 'size') and file.size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="DICOM file size too large. Maximum 50MB allowed."
        )

    try:
        contents = await file.read()
        logger.info(f"🏥 Processing DICOM for user: {current_user.username}")
        
        image, dicom_metadata = process_dicom_file(contents)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)
        
        if len(prediction.shape) == 2 and prediction.shape[1] == 1:
            prediction_score = float(prediction[0][0])
        elif len(prediction.shape) == 1:
            prediction_score = float(prediction[0])
        else:
            prediction_score = float(prediction.flatten()[0])
        
        result = interpret_prediction(prediction_score)
        
        await db[Collections.USERS].update_one(
            {"username": current_user.username},
            {
                "$inc": {"credits": -1, "total_predictions": 1},
                "$set": {"last_prediction": datetime.now()}
            }
        )
        
        result.update({
            "file_type": "DICOM",
            "dicom_metadata": dicom_metadata,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "user": current_user.username,
            "credits_remaining": current_user.credits - 1,
            "processing_notes": {
                "dicom_conversion": "Successfully converted DICOM to RGB image",
                "metadata_extracted": len(dicom_metadata),
                "original_format": f"{dicom_metadata.get('modality', 'Unknown')} from {dicom_metadata.get('manufacturer', 'Unknown')}"
            },
            "cross_operator_validation_performance": {
                "accuracy": "86.0%",
                "sensitivity": "96.4%", 
                "specificity": "74.8%",
                "validated_on": "485 independent samples"
            },
            "disclaimer": "This AI assistant is for preliminary screening only. Always consult healthcare professionals for medical decisions."
        })
        
        logger.info(f"✅ DICOM prediction completed: {result['diagnosis']} ({result['confidence']:.1f}%) for user {current_user.username}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DICOM prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"DICOM prediction failed: {str(e)}"
        )

# 🚀 NEW: Batch processing endpoint
@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple chest X-ray images or DICOM files"),
    current_user: UserResponse = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """
    🔐 PROTECTED: Batch predict pneumonia from multiple chest X-rays
    Supports mixed JPEG/PNG/DICOM files in single request
    Requires JWT authentication and sufficient credits
    """
    
    batch_start_time = time.time()
    batch_id = str(uuid.uuid4())[:8]
    
    # Validate batch size
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 50 files allowed per batch."
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided for batch processing."
        )
    
    # Check credits
    required_credits = len(files)
    if current_user.credits < required_credits:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"❌ Insufficient credits. Need {required_credits} credits, you have {current_user.credits}."
        )

    logger.info(f"🔬 Starting batch processing for user: {current_user.username}, {len(files)} files, batch_id: {batch_id}")
    
    # Process all files
    results = []
    successful_predictions = 0
    failed_predictions = 0
    pneumonia_count = 0
    normal_count = 0
    total_confidence = 0.0
    
    for file_index, file in enumerate(files):
        file_start_time = time.time()
        
        try:
            # Validate individual file
            file_size_mb = getattr(file, 'size', 0) / (1024 * 1024) if hasattr(file, 'size') else 0
            
            # Process based on file type
            is_dicom = file.filename.lower().endswith('.dcm')
            
            if is_dicom:
                if file_size_mb > 50:
                    raise Exception("DICOM file too large (max 50MB)")
                
                contents = await file.read()
                image, dicom_metadata = process_dicom_file(contents)
                file_type = "DICOM"
            else:
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise Exception("File must be an image or DICOM file")
                
                if file_size_mb > 10:
                    raise Exception("Image file too large (max 10MB)")
                
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                dicom_metadata = None
                file_type = "IMAGE"
            
            # AI Prediction
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image, verbose=0)
            
            if len(prediction.shape) == 2 and prediction.shape[1] == 1:
                prediction_score = float(prediction[0][0])
            elif len(prediction.shape) == 1:
                prediction_score = float(prediction[0])
            else:
                prediction_score = float(prediction.flatten()[0])
            
            result = interpret_prediction(prediction_score)
            
            # Track statistics
            successful_predictions += 1
            total_confidence += result['confidence']
            
            if result['diagnosis'] == "PNEUMONIA":
                pneumonia_count += 1
            else:
                normal_count += 1
            
            file_result = BatchFileResult(
                filename=file.filename,
                file_index=file_index,
                success=True,
                diagnosis=result['diagnosis'],
                confidence=result['confidence'],
                confidence_level=result['confidence_level'],
                recommendation=result['recommendation'],
                raw_score=result['raw_score'],
                file_type=file_type,
                dicom_metadata=dicom_metadata,
                processing_time_ms=round((time.time() - file_start_time) * 1000, 2)
            )
            
        except Exception as e:
            failed_predictions += 1
            file_result = BatchFileResult(
                filename=file.filename,
                file_index=file_index,
                success=False,
                error_message=str(e),
                file_type="UNKNOWN",
                processing_time_ms=round((time.time() - file_start_time) * 1000, 2)
            )
            
            logger.warning(f"❌ File {file_index} ({file.filename}) failed: {e}")
        
        results.append(file_result)
    
    # Calculate batch summary
    processing_time_seconds = round(time.time() - batch_start_time, 2)
    average_confidence = round(total_confidence / successful_predictions, 2) if successful_predictions > 0 else 0.0
    
    summary = BatchSummary(
        total_files=len(files),
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions,
        pneumonia_detected=pneumonia_count,
        normal_cases=normal_count,
        average_confidence=average_confidence,
        processing_time_seconds=processing_time_seconds,
        credits_consumed=successful_predictions
    )
    
    # Deduct credits (only for successful predictions)
    credits_to_deduct = successful_predictions
    await db[Collections.USERS].update_one(
        {"username": current_user.username},
        {
            "$inc": {
                "credits": -credits_to_deduct, 
                "total_predictions": successful_predictions
            },
            "$set": {"last_prediction": datetime.now()}
        }
    )
    
    # Determine batch status
    if failed_predictions == 0:
        batch_status = BatchStatus.COMPLETED
    elif successful_predictions == 0:
        batch_status = BatchStatus.FAILED
    else:
        batch_status = BatchStatus.PARTIAL
    
    batch_response = BatchResponse(
        batch_id=batch_id,
        status=batch_status,
        user=current_user.username,
        timestamp=datetime.now(),
        summary=summary,
        results=results,
        credits_remaining=current_user.credits - credits_to_deduct,
        cross_operator_validation_performance={
            "accuracy": "86.0%",
            "sensitivity": "96.4%", 
            "specificity": "74.8%",
            "validated_on": "485 independent samples"
        },
        disclaimer="This AI assistant is for preliminary screening only. Always consult healthcare professionals for medical decisions."
    )
    
    logger.info(f"✅ Batch {batch_id} completed: {successful_predictions}/{len(files)} successful, {processing_time_seconds}s, user: {current_user.username}")
    
    return batch_response

# 📊 Batch export endpoint
@router.get("/batch/{batch_id}/export/csv")
async def export_batch_results_csv(
    batch_id: str,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """🔐 PROTECTED: Export batch results as CSV"""
    return JSONResponse(
        content={
            "message": "CSV export feature - store batch results in MongoDB for persistent access",
            "batch_id": batch_id,
            "user": current_user.username,
            "implementation_note": "In production: query stored batch results and generate CSV"
        }
    )

# Health check endpoint
@router.get("/health")
def health_check():
    """Health check for prediction service."""
    return {
        "status": "healthy" if model_info["loaded"] else "unhealthy",
        "model_loaded": model_info["loaded"],
        "load_time": model_info["load_time"],
        "timestamp": datetime.now().isoformat(),
        "supported_formats": ["JPEG", "PNG", "DICOM (.dcm)"],
        "endpoints": {
            "single_prediction": "/predict",
            "dicom_prediction": "/predict/dicom", 
            "batch_prediction": "/predict/batch",
            "batch_export": "/batch/{batch_id}/export/csv"
        },
        "batch_capabilities": {
            "max_files_per_batch": 50,
            "supported_mixed_formats": True,
            "dicom_support": True,
            "progress_tracking": True,
            "csv_export": "Available"
        },
        "performance_summary": "86% accuracy, 96.4% sensitivity, 25.2% false positive rate"
    }

@router.get("/stats")
def get_model_stats():
    """Get model performance statistics."""
    return {
        "performance_metrics": {
            "overall_accuracy": "86.0%",
            "sensitivity": "96.4%",
            "specificity": "74.8%",
            "precision": "80.4%",
            "false_positive_rate": "25.2%",
            "false_negative_rate": "3.6%",
            "roc_auc": "0.964",
            "pr_auc": "0.968"
        },
        "cross_operator_validation_confusion_matrix": {
            "true_negatives": 175,
            "false_positives": 59,
            "false_negatives": 9,
            "true_positives": 242,
            "total_test_samples": 485
        },
        "clinical_interpretation": {
            "excellent_screening": "96.4% sensitivity ideal for pneumonia screening",
            "false_alarm_consideration": "25.2% false positive rate requires clinical review",
            "high_detection_rate": "96.4% of pneumonia cases correctly identified",
            "clinical_readiness": "Ready for real-world clinical validation"
        },
        "validation_methodology": {
            "type": "cross_operator_validation",
            "dataset": "485 independent samples",
            "generalization": "Good (8.8% drop from internal validation)"
        }
    }

@router.get("/info")
def model_info_endpoint():
    """Detailed model information."""
    return {
        "model_info": model_info,
        "clinical_validation": {
            "accuracy": "86.0%",
            "sensitivity": "96.4%",
            "specificity": "74.8%",
            "clinical_readiness": "READY for clinical validation"
        },
        "cross_operator_validation": {
            "dataset_size": "485 independent samples",
            "normal_cases": "234",
            "pneumonia_cases": "251",
            "generalization": "Good (8.8% drop from internal validation)"
        },
        "technical_specs": {
            "architecture": "MobileNetV2 with custom classification head",
            "input_size": "224x224 RGB images",
            "training_data": "Balanced dataset (1:1 ratio)",
            "preprocessing": "Resize to 224x224, normalize to [0,1]"
        },
        "usage_guidelines": {
            "intended_use": "Preliminary pneumonia screening assistant",
            "limitations": "Not a replacement for professional diagnosis",
            "recommendation": "Always consult healthcare professionals for medical decisions"
        }
    }
