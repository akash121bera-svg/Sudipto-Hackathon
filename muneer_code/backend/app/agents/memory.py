import logging
import uuid
from typing import Dict, Any, List
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import make_log_entry
from backend.app.core.database import SessionLocal
from backend.app.models.schemas import DocumentModel, FeedbackMemoryModel
from backend.app.core.vector_store import store_embedding, search_embeddings

logger = logging.getLogger("memory_agent")

def retrieve_correction_from_memory(field_key: str, raw_value: str) -> str:
    """
    Searches Qdrant memory for similar past corrections.
    Returns the corrected value if a highly similar match is found, else returns empty string.
    """
    if not raw_value:
        return ""
        
    query_text = f"{field_key}: {raw_value.lower()}"
    try:
        results = search_embeddings("correction_memory", query_text, limit=1)
        if results and results[0]["score"] > 0.88:
            payload = results[0]["payload"]
            corrected = payload.get("corrected_value", "")
            logger.info(f"Memory hit! Match score {results[0]['score']:.2f} for '{query_text}'. Auto-correcting to '{corrected}'.")
            return corrected
    except Exception as e:
        logger.warning(f"Error querying memory index: {e}")
        
    return ""

def memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Learning Memory Agent Node.
    Saves document final state and commits human corrections to long-term database/vector memory.
    If running during normal pipeline, retrieves past corrections to improve OCR values.
    """
    doc_id = state.get("document_id")
    fields = state.get("fields", {}).copy()
    human_corrections = state.get("human_corrections", {})
    logs = []
    
    db = SessionLocal()
    try:
        # Scenario A: We are in the final save stage AFTER human review (human corrections present)
        if human_corrections:
            logs.append(make_log_entry("Memory", f"Persisting {len(human_corrections)} corrections into semantic vector memory."))
            
            for key, val in human_corrections.items():
                original_val = fields.get(key, {}).get("value", "")
                
                # Update local field values in state
                if key in fields:
                    fields[key]["value"] = val
                    fields[key]["confidence"] = 1.0  # Certified correct by human
                    fields[key]["engine"] = "human_operator"
                
                # Save to SQL memory
                db_mem = FeedbackMemoryModel(
                    document_id=uuid.UUID(doc_id),
                    field_key=key,
                    original_value=original_val,
                    corrected_value=val
                )
                db.add(db_mem)
                db.flush() # generate id
                
                # Save to Qdrant semantic search vector store
                # Store representation: e.g., "full_name: Jahn" -> payload {"corrected_value": "John"}
                q_id = str(uuid.uuid4())
                q_text = f"{key}: {original_val.lower() if original_val else ''}"
                q_payload = {
                    "field_key": key,
                    "original_value": original_val,
                    "corrected_value": val,
                    "db_memory_id": db_mem.id
                }
                store_embedding("correction_memory", q_id, q_text, q_payload)
            
            db.commit()
            logs.append(make_log_entry("Memory", "Successfully committed corrections to PostgreSQL and Qdrant."))
            
        # Scenario B: We are in the normal pipeline, check memory to AUTO-CORRECT low confidence fields
        else:
            corrections_applied = []
            for key, f_data in fields.items():
                val = f_data.get("value", "")
                conf = f_data.get("confidence", 0.0)
                
                # Try memory recall for low confidence OCR
                if conf < 0.85 and val:
                    recalled_val = retrieve_correction_from_memory(key, val)
                    if recalled_val and recalled_val != val:
                        fields[key]["value"] = recalled_val
                        fields[key]["confidence"] = 0.95 # Higher confidence from memory alignment
                        fields[key]["engine"] = "learning_memory_recall"
                        corrections_applied.append(f"{key}: '{val}' -> '{recalled_val}'")
                        
            if corrections_applied:
                logs.append(make_log_entry(
                    "Memory", 
                    f"Recalled and applied historical corrections: {', '.join(corrections_applied)}"
                ))
            else:
                logs.append(make_log_entry("Memory", "Queried long-term memory. No matching correction patterns found."))

        # Update final document record in PostgreSQL
        doc = db.query(DocumentModel).filter(DocumentModel.id == uuid.UUID(doc_id)).first()
        if doc:
            doc.preprocessed_path = state.get("preprocessed_path")
            doc.extracted_data = {k: v["value"] for k, v in fields.items()}
            # Format fields with full details for visual dashboard
            doc.extracted_data["_fields_meta"] = fields
            doc.confidence_score = state.get("ocr_confidence", 0.0)
            doc.status = "completed"
            
            # Append memory logs to DB reasoning trace
            doc.reasoning_log = (doc.reasoning_log or []) + logs
            db.commit()
            
    except Exception as e:
        db.rollback()
        logger.error(f"Error in Memory Agent node: {e}")
        logs.append(make_log_entry("Memory", f"Memory write failed: {e}", level="ERROR"))
    finally:
        db.close()
        
    return {
        "fields": fields,
        "reasoning_log": logs
    }
