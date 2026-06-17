import logging
from typing import Dict, Any
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import make_log_entry

logger = logging.getLogger("confidence_agent")

def confidence_node(state: AgentState) -> Dict[str, Any]:
    """
    Confidence Agent Node.
    Analyzes field confidences and validation reports to assign final confidence
    and set requires_review or retry flags.
    """
    fields = state.get("fields", {})
    validation = state.get("validation_report", {})
    errors = validation.get("errors", [])
    
    logs = []
    
    if not fields:
        return {
            "ocr_confidence": 0.0,
            "requires_review": True,
            "reasoning_log": [make_log_entry("Confidence", "No fields extracted. Flagging for review.", level="ERROR")]
        }
        
    # Calculate weighted confidence
    total_conf = 0.0
    count = 0
    
    adjusted_fields = fields.copy()
    
    for key, f_data in adjusted_fields.items():
        conf = f_data.get("confidence", 0.0)
        
        # Penalize confidence if there is a validation error associated with the field
        # E.g., if phone number fails length, set its confidence to 0
        field_has_error = any(key in err.lower().replace("_", " ") for err in errors)
        if field_has_error:
            conf = max(0.0, conf - 0.4) # Deduct 40% confidence for semantic errors
            f_data["confidence"] = round(conf, 3)
            logs.append(make_log_entry(
                "Confidence", 
                f"Penalized field '{key}' confidence due to validation errors.", 
                level="WARNING"
            ))
            
        total_conf += conf
        count += 1
        
    avg_confidence = total_conf / count if count > 0 else 0.0
    
    # Determine trust level
    if avg_confidence >= 0.85 and len(errors) == 0:
        trust_level = "high"
        requires_review = False
    elif avg_confidence >= 0.65 and len(errors) == 0:
        trust_level = "medium"
        requires_review = False
    else:
        trust_level = "low"
        requires_review = True
        
    logs.append(make_log_entry(
        "Confidence", 
        f"Aggregated Confidence Score: {avg_confidence:.2f}. Trust level determined: '{trust_level}'. "
        f"Requires review: {requires_review}"
    ))

    return {
        "ocr_confidence": round(avg_confidence, 3),
        "requires_review": requires_review,
        "fields": adjusted_fields,
        "reasoning_log": logs
    }
