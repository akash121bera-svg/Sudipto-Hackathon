import os
import uuid
import logging
import datetime
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.schemas import DocumentModel, DocumentResponse
from backend.app.agents.graph import agent_workflow

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger("documents_route")

def run_agent_pipeline(doc_id: uuid.UUID, file_path: str, filename: str):
    """Background task executing the LangGraph multi-agent flow."""
    db = next(get_db())
    try:
        logger.info(f"Starting multi-agent workflow for document {doc_id}")
        
        # Update status to processing
        doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        if doc:
            doc.status = "processing"
            db.commit()

        # Initialize shared agent state
        initial_state = {
            "document_id": str(doc_id),
            "filename": filename,
            "file_path": file_path,
            "preprocessed_path": "",
            "current_stage": "start",
            "ocr_confidence": 0.0,
            "retry_count": 0,
            "requires_review": False,
            "fields": {},
            "template_key": "",
            "validation_report": {"errors": [], "warnings": [], "is_valid": True},
            "reasoning_log": [],
            "preprocessing_params": {},
            "human_corrections": {}
        }
        
        # Execute LangGraph
        final_state = agent_workflow.invoke(initial_state)
        
        # Final database commits are handled inside Memory agent or HumanReview nodes,
        # but let's ensure status is updated in case of unexpected graph endings
        db.refresh(doc)
        if doc.status == "processing":
            if final_state.get("requires_review"):
                doc.status = "requires_review"
            else:
                doc.status = "completed"
            db.commit()
            
        logger.info(f"LangGraph execution finished for document {doc_id}. Status: {doc.status}")
        
    except Exception as e:
        logger.error(f"Error executing agent pipeline for document {doc_id}: {e}")
        # Mark as failed in DB
        doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        if doc:
            doc.status = "failed"
            doc.reasoning_log = (doc.reasoning_log or []) + [{
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "agent": "System",
                "message": f"Pipeline execution failed: {str(e)}",
                "level": "ERROR"
            }]
            db.commit()
    finally:
        db.close()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Uploads document image, creates DB entry, and schedules background multi-agent analysis."""
    # 1. Save uploaded file
    file_id = uuid.uuid4()
    ext = os.path.splitext(file.filename)[1]
    saved_filename = f"{file_id}{ext}"
    saved_path = os.path.join(settings.UPLOAD_DIR, saved_filename)
    
    try:
        with open(saved_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail=f"File save error: {e}")

    # 2. Register entry in PostgreSQL
    db_doc = DocumentModel(
        id=file_id,
        filename=file.filename,
        status="pending",
        file_path=saved_path,
        confidence_score=0.0,
        retry_count=0
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # 3. Trigger background LangGraph execution
    background_tasks.add_task(run_agent_pipeline, file_id, saved_path, file.filename)
    
    return db_doc

@router.get("/", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """Retrieves all documents sorted by upload time."""
    return db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()

@router.get("/{doc_id}")
def get_document_details(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    """Returns all metadata, extracted fields, bounding boxes, validation, and logs for a document."""
    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "file_path": doc.file_path,
        "preprocessed_path": doc.preprocessed_path,
        "confidence_score": doc.confidence_score,
        "retry_count": doc.retry_count,
        "extracted_data": doc.extracted_data,
        "validation_report": doc.validation_report,
        "reasoning_log": doc.reasoning_log,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at
    }

@router.post("/{doc_id}/reprocess", response_model=DocumentResponse)
def reprocess_document(
    doc_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Force re-runs the multi-agent pipeline on an existing document."""
    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    doc.status = "pending"
    doc.retry_count = 0
    db.commit()
    db.refresh(doc)
    
    background_tasks.add_task(run_agent_pipeline, doc.id, doc.file_path, doc.filename)
    return doc
