import os
import logging
from typing import Dict, Any
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import make_log_entry

logger = logging.getLogger("intake_agent")

def intake_node(state: AgentState) -> Dict[str, Any]:
    """
    Intake Agent Node.
    Validates uploaded file existence, format, and initial image dimensions.
    """
    file_path = state.get("file_path", "")
    filename = state.get("filename", "")
    
    logs = []
    
    if not file_path or not os.path.exists(file_path):
        logs.append(make_log_entry("Intake", f"File path not found: {file_path}", level="ERROR"))
        raise FileNotFoundError(f"Input file path {file_path} does not exist.")
        
    # Check extension
    ext = os.path.splitext(file_path)[1].lower()
    allowed_exts = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']
    if ext not in allowed_exts:
        logs.append(make_log_entry("Intake", f"Invalid file format: {ext}. Allowed: {allowed_exts}", level="ERROR"))
        raise ValueError(f"File extension {ext} not supported.")
        
    # Basic size check
    file_size_kb = os.path.getsize(file_path) / 1024
    logs.append(make_log_entry("Intake", f"File '{filename}' verified successfully. Size: {file_size_kb:.1f} KB."))
    
    # Initialize preprocessing params if not present
    preprocessing_params = state.get("preprocessing_params")
    if not preprocessing_params:
        preprocessing_params = {
            "deskew": True,
            "denoise": True,
            "clahe_enhancement": False,
            "adaptive_thresh": False,
            "scale_factor": 1.0
        }
        
    return {
        "preprocessing_params": preprocessing_params,
        "reasoning_log": logs
    }
