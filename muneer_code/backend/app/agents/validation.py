import re
import datetime
import logging
from typing import Dict, Any, List
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import make_log_entry

logger = logging.getLogger("validation_agent")

def calculate_age(dob_str: str) -> int:
    """Calculates age from YYYY-MM-DD string."""
    try:
        birth_date = datetime.datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    except Exception:
        return -1

def validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Validation & Reasoning Agent Node.
    Inspects extracted fields, cross-checks logical rules, and outputs errors/warnings.
    """
    fields = state.get("fields", {})
    logs = []
    
    errors = []
    warnings = []
    
    # 1. Check Mandatory Fields
    mandatory_fields = ["full_name", "date_of_birth", "phone_number", "email_address"]
    
    for req in mandatory_fields:
        if req not in fields or not fields[req].get("value"):
            errors.append(f"Missing mandatory field: {req.replace('_', ' ').title()}")
            
    # Extract values for logical reasoning
    name_val = fields.get("full_name", {}).get("value", "")
    dob_val = fields.get("date_of_birth", {}).get("value", "")
    phone_val = fields.get("phone_number", {}).get("value", "")
    email_val = fields.get("email_address", {}).get("value", "")
    membership_val = fields.get("membership_type", {}).get("value", "")
    
    # 2. Name validation
    if name_val:
        if len(name_val) < 2:
            errors.append("Full Name is too short (minimum 2 characters).")
        elif not re.match(r'^[a-zA-Z\s\.\-\']+$', name_val):
            warnings.append("Full Name contains unusual characters (verify formatting).")

    # 3. Email validation
    if email_val:
        email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_pattern, email_val):
            errors.append(f"Invalid email address syntax: '{email_val}'")

    # 4. Phone validation
    if phone_val:
        cleaned_phone = re.sub(r'\D', '', phone_val)
        if len(cleaned_phone) < 7 or len(cleaned_phone) > 15:
            errors.append(f"Phone number digit length ({len(cleaned_phone)}) falls outside [7, 15] bounds.")

    # 5. Date of Birth & Age Cross-Check
    if dob_val:
        # Check if dob matches format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', dob_val):
            errors.append(f"Date of Birth '{dob_val}' must be in YYYY-MM-DD format.")
        else:
            age = calculate_age(dob_val)
            if age == -1:
                errors.append(f"Invalid Date of Birth value: '{dob_val}'")
            elif age < 18:
                # E.g. Under-age warning or error depending on form rules
                warnings.append(f"Applicant is under 18 years old (Age: {age}). Requires guardian consent.")
            elif age > 110:
                errors.append(f"Age calculated is logically impossible: {age} years.")

    # 6. Checkbox Selection Match
    if membership_val:
        allowed_memberships = ["Standard", "Premium", "Student", "Unchecked", "Checked"]
        if membership_val not in allowed_memberships:
            warnings.append(f"Membership Type '{membership_val}' doesn't match predefined options.")

    is_valid = len(errors) == 0
    
    report = {
        "errors": errors,
        "warnings": warnings,
        "is_valid": is_valid
    }
    
    if is_valid:
        logs.append(make_log_entry("Validation", "All field validations passed. Data matches logical schemas."))
    else:
        logs.append(make_log_entry(
            "Validation", 
            f"Validation errors detected: {', '.join(errors)}. Warnings: {len(warnings)}.", 
            level="WARNING"
        ))

    return {
        "validation_report": report,
        "reasoning_log": logs
    }
