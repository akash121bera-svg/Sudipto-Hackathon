import logging
import uuid
from typing import Dict, Any
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import make_log_entry
from backend.app.core.database import SessionLocal
from backend.app.models.schemas import DocumentModel, HumanReviewModel

logger = logging.getLogger("human_review_agent")

def human_review_node(state: AgentState) -> Dict[str, Any]:
    """
    Human Review Agent Node.
    Saves document state and flags fields requiring human validation.
    Inserts a ticket into human_reviews table to hold manual corrections.
    """
    doc_id = state.get("document_id")
    fields = state.get("fields", {})
    validation = state.get("validation_report", {})
    errors = validation.get("errors", [])
    
    logs = []
    
    # Identify low confidence or errored fields
    flagged_fields = []
    for key, f_data in fields.items():
        conf = f_data.get("confidence", 0.0)
        field_has_error = any(key in err.lower().replace("_", " ") for err in errors)
        
        if conf < 0.80 or field_has_error:
            flagged_fields.append(key)
            
    if not flagged_fields:
        # Fallback if somehow triggered without flagged fields
        flagged_fields = list(fields.keys())
        
    logs.append(make_log_entry(
        "HumanReview", 
        f"Flagged fields for human operator correction: {flagged_fields}"
    ))

    # Persist the review request to PostgreSQL
    db = SessionLocal()
    try:
        # Check if review already exists
        existing_review = db.query(HumanReviewModel).filter(
            HumanReviewModel.document_id == uuid.UUID(doc_id),
            HumanReviewModel.status == "pending"
        ).first()
        
        if not existing_review:
            review_ticket = HumanReviewModel(
                id=uuid.uuid4(),
                document_id=uuid.UUID(doc_id),
                fields_flagged=flagged_fields,
                status="pending"
            )
            db.add(review_ticket)
            
        # Update document status to requires_review and save intermediate extraction state
        doc = db.query(DocumentModel).filter(DocumentModel.id == uuid.UUID(doc_id)).first()
        if doc:
            doc.status = "requires_review"
            doc.preprocessed_path = state.get("preprocessed_path")
            doc.extracted_data = {k: v.get("value", "") for k, v in fields.items()}
            # Format fields with full details for visual dashboard
            doc.extracted_data["_fields_meta"] = fields
            doc.confidence_score = state.get("ocr_confidence", 0.0)
            doc.retry_count = state.get("retry_count", 0)
            doc.reasoning_log = (doc.reasoning_log or []) + logs
            
        db.commit()
        logger.info(f"Human review ticket successfully registered for Document ID {doc_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to write human review ticket to database: {e}")
        logs.append(make_log_entry("HumanReview", f"Database write failed: {e}", level="ERROR"))
    finally:
        db.close()

    # We set requires_review to True and let supervisor halt/exit to wait for human API input
    return {
        "requires_review": True,
        "reasoning_log": logs
    }
