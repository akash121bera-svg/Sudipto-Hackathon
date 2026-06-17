import uuid
import logging
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from backend.app.core.database import get_db
from backend.app.models.schemas import DocumentModel, HumanReviewModel, HumanReviewResponse, HumanReviewSubmit
from backend.app.agents.graph import agent_workflow

router = APIRouter(prefix="/review", tags=["review"])
logger = logging.getLogger("review_route")

@router.get("/", response_model=List[HumanReviewResponse])
def list_pending_reviews(db: Session = Depends(get_db)):
    """Lists all open human review tickets."""
    return db.query(HumanReviewModel).filter(HumanReviewModel.status == "pending").all()

@router.get("/{review_id}")
def get_review_details(review_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieves specific review ticket detail and its associated document details."""
    review = db.query(HumanReviewModel).filter(HumanReviewModel.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review ticket not found")
        
    doc = db.query(DocumentModel).filter(DocumentModel.id == review.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Associated document not found")
        
    return {
        "review_ticket": review,
        "document": {
            "id": doc.id,
            "filename": doc.filename,
            "confidence_score": doc.confidence_score,
            "extracted_data": doc.extracted_data,
            "validation_report": doc.validation_report,
            "reasoning_log": doc.reasoning_log
        }
    }

@router.post("/{review_id}/submit")
def submit_human_corrections(
    review_id: uuid.UUID,
    payload: HumanReviewSubmit,
    db: Session = Depends(get_db)
):
    """Submits corrected values, closes the ticket, and resumes the agent pipeline to record learning memory."""
    review = db.query(HumanReviewModel).filter(HumanReviewModel.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review ticket not found")
        
    if review.status == "resolved":
        raise HTTPException(status_code=400, detail="Review ticket already resolved")

    doc = db.query(DocumentModel).filter(DocumentModel.id == review.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Associated document not found")

    # 1. Update review ticket status
    review.corrected_data = payload.corrected_data
    review.reviewed_by = payload.reviewed_by
    review.status = "resolved"
    review.resolved_at = datetime.datetime.utcnow()
    
    # 2. Extract fields metadata from previous run to form the state input
    previous_fields_meta = {}
    if doc.extracted_data and "_fields_meta" in doc.extracted_data:
        previous_fields_meta = doc.extracted_data["_fields_meta"]
    else:
        # Fallback if fields meta was not stored
        previous_fields_meta = {k: {"value": v, "confidence": 0.5, "bbox": [0,0,10,10], "type": "printed_handwritten"} 
                                for k, v in (doc.extracted_data or {}).items() if not k.startswith("_")}

    # 3. Resume the LangGraph workflow from 'human_review' stage to run memory agent
    # We pass the corrections as human_corrections state input
    state_input = {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "file_path": doc.file_path,
        "preprocessed_path": doc.preprocessed_path or "",
        "current_stage": "human_review", # Supervisor knows to route to human_review, then memory
        "ocr_confidence": doc.confidence_score,
        "retry_count": doc.retry_count,
        "requires_review": False,
        "fields": previous_fields_meta,
        "validation_report": doc.validation_report or {"errors": [], "warnings": [], "is_valid": True},
        "reasoning_log": doc.reasoning_log or [],
        "preprocessing_params": {},
        "human_corrections": payload.corrected_data
    }
    
    try:
        # Invoke workflow to run memory persistence node and update long-term indices
        agent_workflow.invoke(state_input)
        db.commit()
        
        return {"status": "success", "message": "Corrections applied and learning memory saved successfully."}
    except Exception as e:
        db.rollback()
        logger.error(f"Error resuming pipeline with human corrections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process corrections: {str(e)}")
