from typing import Dict, List, Any, TypedDict, Annotated
import operator

def append_log(existing_logs: List[Dict[str, Any]], new_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper to merge reasoning logs in the graph state."""
    return existing_logs + new_logs

class AgentState(TypedDict):
    document_id: str
    filename: str
    file_path: str
    preprocessed_path: str
    current_stage: str 
    ocr_confidence: float
    retry_count: int
    requires_review: bool
    fields: Dict[str, Dict[str, Any]]  # field_key -> {value, confidence, bbox, patch_url, engine}
    template_key: str  # detected form template key (e.g. "namibian_employment_application")
    validation_report: Dict[str, Any]  # {"errors": [], "warnings": [], "is_valid": bool}
    reasoning_log: Annotated[List[Dict[str, Any]], append_log]
    preprocessing_params: Dict[str, Any]
    human_corrections: Dict[str, Any]
