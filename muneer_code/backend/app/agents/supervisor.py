import datetime
import logging
from typing import Dict, Any, List
from backend.app.agents.state import AgentState

logger = logging.getLogger("supervisor_agent")

def make_log_entry(agent_name: str, message: str, level: str = "INFO") -> Dict[str, Any]:
    """Helper to format a reasoning log entry."""
    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "agent": agent_name,
        "message": message,
        "level": level
    }

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Agent Node.
    Analyzes the global state, records routing decisions in the reasoning log,
    and updates the current_stage.
    """
    stage = state.get("current_stage", "start")
    retry_count = state.get("retry_count", 0)
    ocr_confidence = state.get("ocr_confidence", 0.0)
    validation = state.get("validation_report", {})
    errors = validation.get("errors", [])
    
    logs = []
    next_stage = ""

    if stage == "start":
        logs.append(make_log_entry(
            "Supervisor", 
            "Initializing document processing lifecycle. Delegating file validation to Intake Agent."
        ))
        next_stage = "intake"
        
    elif stage == "intake":
        logs.append(make_log_entry(
            "Supervisor", 
            "File received and validated. Routing to Vision Preprocessing Agent for cleanup."
        ))
        next_stage = "preprocess"
        
    elif stage == "preprocess":
        logs.append(make_log_entry(
            "Supervisor", 
            "Image cleaning completed. Delegating field and structure detection to Layout Intelligence Agent."
        ))
        next_stage = "layout"
        
    elif stage == "layout":
        logs.append(make_log_entry(
            "Supervisor", 
            "Form structure and input boxes mapped. Triggering OCR Agent for hybrid printed/handwritten text extraction."
        ))
        next_stage = "ocr"
        
    elif stage == "ocr":
        logs.append(make_log_entry(
            "Supervisor", 
            "OCR text extraction finished. Routing text segments to Semantic Agent for schema normalization."
        ))
        next_stage = "semantic"
        
    elif stage == "semantic":
        logs.append(make_log_entry(
            "Supervisor", 
            "Text structured into canonical schema. Handing off to Validation & Reasoning Agent."
        ))
        next_stage = "validation"
        
    elif stage == "validation":
        logs.append(make_log_entry(
            "Supervisor", 
            "Business rule validations completed. Activating Confidence Agent to determine trust level."
        ))
        next_stage = "confidence"
        
    elif stage == "confidence":
        # Supervisor evaluates confidence & validation report to decide next phase
        has_errors = len(errors) > 0
        
        if ocr_confidence >= 0.85 and not has_errors:
            logs.append(make_log_entry(
                "Supervisor", 
                f"Confidence is high ({ocr_confidence:.2f}) and validations passed. Saving to Memory Agent and finalizing."
            ))
            next_stage = "memory"
        else:
            # Low confidence or validation failure trigger retry or escalation
            if retry_count < 2:
                new_retry = retry_count + 1
                
                # Adjust preprocessing params based on issues detected
                params = state.get("preprocessing_params", {}).copy()
                params["clahe_enhancement"] = True
                params["scale_factor"] = 1.5 if new_retry == 1 else 2.0
                params["adaptive_thresh"] = True if new_retry == 2 else False
                
                logs.append(make_log_entry(
                    "Supervisor", 
                    f"Low confidence ({ocr_confidence:.2f}) or validation errors detected. "
                    f"Triggering Retry {new_retry}/2. Updating preprocessing params: scale_factor={params['scale_factor']}, clahe=True. "
                    f"Routing back to Preprocess Agent.",
                    level="WARNING"
                ))
                
                return {
                    "current_stage": "preprocess",
                    "retry_count": new_retry,
                    "preprocessing_params": params,
                    "reasoning_log": logs
                }
            else:
                logs.append(make_log_entry(
                    "Supervisor", 
                    f"Processing failed after 2 retries (Confidence: {ocr_confidence:.2f}, Errors: {len(errors)}). "
                    "Escalating fields to Human Review Agent.",
                    level="ERROR"
                ))
                next_stage = "human_review"
                
    elif stage == "human_review":
        if state.get("requires_review"):
            logs.append(make_log_entry(
                "Supervisor", 
                "Document requires manual human review. Pausing pipeline and registering ticket."
            ))
            next_stage = "end"
        else:
            logs.append(make_log_entry(
                "Supervisor", 
                "Human corrections received. Routing to Memory Agent to store feedback and train the semantic index."
            ))
            next_stage = "memory"
        
    elif stage == "memory":
        logs.append(make_log_entry(
            "Supervisor", 
            "Long-term memory updated. Document processing successfully completed.",
        ))
        next_stage = "end"
 
    else:
        logger.error(f"Supervisor hit unknown stage: {stage}")
        next_stage = "end"
 
    return {
        "current_stage": next_stage,
        "reasoning_log": logs
    }
 
def supervisor_router(state: AgentState) -> str:
    """Conditional router function for LangGraph."""
    stage = state.get("current_stage", "end")
    if stage == "end":
        return "end"
    return stage
