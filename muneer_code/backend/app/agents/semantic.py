import json
import logging
import requests
import re
from typing import Dict, Any
from backend.app.core.config import settings
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import make_log_entry

logger = logging.getLogger("semantic_agent")

def normalize_date(date_str: str) -> str:
    """Normalizes various date formats to YYYY-MM-DD using regex fallbacks."""
    date_str = date_str.strip()
    # E.g. "12/04/1990" -> "1990-04-12" or "1990-12-04"
    # E.g. "1990/12/04" -> "1990-12-04"
    # Simple regex normalization
    date_str = re.sub(r'[\s\.\/]', '-', date_str)
    parts = date_str.split('-')
    if len(parts) == 3:
        # Check if year is first
        if len(parts[0]) == 4:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        # Check if year is last
        elif len(parts[2]) == 4:
            # Assume DD-MM-YYYY or MM-DD-YYYY, convert to YYYY-MM-DD
            # Let's standardize to YYYY-MM-DD. For standard US/European formats:
            # If parts[1] is > 12, then parts[1] is day, parts[0] is month
            # Let's write a simple helper
            p1, p2, p3 = parts
            if int(p1) > 12:
                # DD-MM-YYYY
                return f"{p3}-{p2.zfill(2)}-{p1.zfill(2)}"
            else:
                # Default to MM-DD-YYYY
                return f"{p3}-{p1.zfill(2)}-{p2.zfill(2)}"
    return date_str

def normalize_phone(phone_str: str) -> str:
    """Normalizes phone numbers to standard formats."""
    # Strip spaces and non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone_str)
    return cleaned

def clean_ocr_text(text: str) -> str:
    """Cleans up common OCR garbage characters."""
    return re.sub(r'[|\\_\[\]~`]', '', text).strip()

def run_llama_semantic_parser(raw_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Calls local Ollama Llama3 model to structure and normalize OCR outputs."""
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    
    prompt = f"""
    You are a document extraction parsing agent.
    Convert the following raw OCR key-value mappings into a clean, canonical JSON format.
    Correct spelling mistakes, resolve ambiguities, format dates to YYYY-MM-DD, and format phone numbers.
    
    Raw OCR Data:
    {json.dumps(raw_fields, indent=2)}
    
    Output JSON object ONLY. Do not write explanation, code blocks, or markdown. Output valid JSON.
    Example:
    {{
      "full_name": "John Doe",
      "date_of_birth": "1990-12-04",
      "phone_number": "+15550199",
      "email_address": "john.doe@example.com",
      "membership_type": "Premium"
    }}
    """
    
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=8.0)
        if response.status_code == 200:
            result = response.json()
            return json.loads(result.get("response", "{}"))
        else:
            raise RuntimeError(f"Ollama server returned code {response.status_code}")
    except Exception as e:
        logger.warning(f"Ollama Llama3 request failed: {e}. Falling back to rule-based normalizer.")
        raise e

def semantic_node(state: AgentState) -> Dict[str, Any]:
    """
    Semantic Understanding Agent Node.
    Maps raw OCR fields to structured, clean database values.
    """
    raw_fields = state.get("fields", {})
    logs = []
    
    # Flatten fields to pass to LLM or rule-based parser
    flat_data = {k: v.get("value", "") for k, v in raw_fields.items()}
    
    structured_data = {}
    normalized_log = []

    # Try LLM first if fallback is not active
    llm_success = False
    if not settings.ENABLE_MOCK_FALLBACK:
        try:
            structured_data = run_llama_semantic_parser(flat_data)
            llm_success = True
            logs.append(make_log_entry("Semantic", "Successfully parsed and structured text using Ollama (Llama3)."))
        except Exception:
            pass
            
    if not llm_success:
        # High quality rule-based fallback
        logs.append(make_log_entry("Semantic", "Running deterministic rule-based normalizer for values."))
        for key, val in flat_data.items():
            val_str = str(val)
            cleaned = clean_ocr_text(val_str)
            
            if "date" in key or "dob" in key or "birth" in key:
                cleaned = normalize_date(cleaned)
                normalized_log.append(f"Normalized DOB '{val}' -> '{cleaned}'")
            elif "phone" in key or "mobile" in key:
                cleaned = normalize_phone(cleaned)
                normalized_log.append(f"Normalized phone '{val}' -> '{cleaned}'")
            elif "email" in key:
                cleaned = cleaned.lower().replace(" ", "")
                normalized_log.append(f"Normalized email '{val}' -> '{cleaned}'")
            elif key == "membership_type":
                # Ensure it matches standard options
                cleaned = cleaned.capitalize()
                
            structured_data[key] = cleaned
            
        if normalized_log:
            logs.append(make_log_entry("Semantic", f"Applied normalizations: {'; '.join(normalized_log)[:200]}..."))

    # Update state fields with normalized values
    updated_fields = raw_fields.copy()
    for key, norm_val in structured_data.items():
        if key in updated_fields:
            updated_fields[key]["value"] = norm_val
            
    return {
        "fields": updated_fields,
        "reasoning_log": logs
    }
