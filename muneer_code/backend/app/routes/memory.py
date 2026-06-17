import logging
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from backend.app.core.database import get_db
from backend.app.models.schemas import FeedbackMemoryModel, MemoryEntryResponse
from backend.app.core.vector_store import search_embeddings

router = APIRouter(prefix="/memory", tags=["memory"])
logger = logging.getLogger("memory_route")

@router.get("/", response_model=List[MemoryEntryResponse])
def list_memory_logs(db: Session = Depends(get_db)):
    """Retrieves list of correction mappings stored in database memory."""
    return db.query(FeedbackMemoryModel).order_by(FeedbackMemoryModel.created_at.desc()).all()

@router.get("/search")
def search_vector_memory(
    query: str = Query(..., description="Query string to search in Qdrant memory"),
    limit: int = Query(3, description="Maximum number of memory matches to return")
):
    """Executes a semantic vector search query in the Qdrant correction memory collection."""
    try:
        results = search_embeddings("correction_memory", query, limit=limit)
        return {
            "query": query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Semantic search query failed: {e}")
        return {
            "query": query,
            "results": [],
            "error": str(e)
        }
