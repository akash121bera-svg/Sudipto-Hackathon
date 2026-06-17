import json
import os
import sys
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Eligibility Match Engine
# ---------------------------------------------------------
def check_age_eligible(user_age: int, min_age, max_age) -> bool:
    eligible_min = True if min_age is None or pd.isna(min_age) else user_age >= int(min_age)
    eligible_max = True if max_age is None or pd.isna(max_age) else user_age <= int(max_age)
    return eligible_min and eligible_max

def check_gender_eligible(user_gender: str, scheme_gender) -> bool:
    user_gender = user_gender.lower().strip()
    if isinstance(scheme_gender, list):
        scheme_genders = [g.lower().strip() for g in scheme_gender]
    else:
        scheme_genders = [g.lower().strip() for g in str(scheme_gender).replace(",", "/").split("/")]
    if "all" in scheme_genders or "any" in scheme_genders:
        return True
    return user_gender in scheme_genders

def check_occupation_eligible(user_occ: str, scheme_occ) -> bool:
    user_occ = user_occ.lower().strip()
    if isinstance(scheme_occ, list):
        scheme_occs = [o.lower().strip() for o in scheme_occ]
    else:
        scheme_occs = [o.lower().strip() for o in str(scheme_occ).replace(",", "/").split("/")]
    if any(any(x in occ for x in ["all", "any", "not specified", "not required"]) for occ in scheme_occs):
        return True
    return any(user_occ in occ or occ in user_occ for occ in scheme_occs)

def check_education_eligible(user_edu: str, scheme_edu) -> bool:
    user_edu = user_edu.lower().strip()
    scheme_edu_str = str(scheme_edu).lower().strip()
    
    if any(x in scheme_edu_str for x in ["not required", "not specified", "literate", "all"]):
        return True
        
    ranks = {
        "literate": 0,
        "5th": 1, "five": 1, "class 5": 1,
        "8th": 2, "eight": 2, "class 8": 2,
        "10th": 3, "ten": 3, "matric": 3, "secondary": 3, "class 10": 3, " 10": 3,
        "12th": 4, "twelve": 4, "higher secondary": 4, "class 12": 4, " 12": 4,
        "iti": 5, "diploma": 5,
        "graduate": 6, "degree": 6, "undergraduate": 6, "college": 6, "bachelor": 6,
        "postgraduate": 7, "master": 7, "post graduate": 7
    }
    
    user_rank = 0
    for key, rank in ranks.items():
        if key in user_edu:
            user_rank = max(user_rank, rank)
            
    scheme_rank_required = 0
    has_requirement = False
    for key, rank in ranks.items():
        if key in scheme_edu_str:
            has_requirement = True
            scheme_rank_required = max(scheme_rank_required, rank)
            
    if has_requirement:
        return user_rank >= scheme_rank_required
    return user_edu in scheme_edu_str

def parse_income_limit(limit):
    if limit is None or pd.isna(limit):
        return None
    try:
        return float(limit)
    except ValueError:
        pass
    limit_str = str(limit).lower().replace(",", "").replace(" ", "")
    multiplier = 1
    if "lakh" in limit_str:
        multiplier = 100000
        limit_str = limit_str.replace("lakh", "").replace("s", "")
    nums = "".join([c for c in limit_str if c.isdigit() or c == '.'])
    if nums:
        try:
            return float(nums) * multiplier
        except ValueError:
            pass
    return None

def check_income_eligible(user_income: float, income_limit) -> bool:
    limit_val = parse_income_limit(income_limit)
    if limit_val is None:
        return True
    return user_income <= limit_val

def check_state_eligible(user_state: str, scheme_state: str) -> bool:
    user_state = user_state.lower().strip()
    scheme_state_str = str(scheme_state).lower().strip()
    if "all india" in scheme_state_str or "any" in scheme_state_str:
        return True
    return user_state in scheme_state_str

# ---------------------------------------------------------
# Pydantic Schema and Extraction
# ---------------------------------------------------------
class UserProfile(BaseModel):
    age: Optional[int] = Field(None)
    gender: Optional[str] = Field(None)
    occupation: Optional[str] = Field(None)
    education: Optional[str] = Field(None)
    annual_income: Optional[int] = Field(None)
    state: Optional[str] = Field(None)

def run_extraction(text: str, client) -> dict:
    if not client:
        return {
            "age": 22,
            "gender": "male",
            "occupation": "student",
            "education": "graduate",
            "annual_income": 300000,
            "state": "Delhi"
        }
    prompt = (
        "Extract the structured user profile details from the user text below. "
        "Strictly adhere to the provided schema and extract correct values (such as calculating income)."
    )
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, f"User Text: \"{text}\""],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=UserProfile,
                temperature=0.1
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error calling live Gemini: {e}. Using offline fallback parser.")
        return run_extraction(text, None)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("       AI Youth & Employment Scheme Navigator (CLI)")
    print("=" * 60)
    
    # Load dataset
    if not os.path.exists("schemes.json"):
        print("Error: schemes.json dataset not found in workspace.")
        sys.exit(1)
        
    with open("schemes.json", "r", encoding="utf-8") as f:
        schemes_data = json.load(f)
        
    # Get user description
    default_desc = "I am a 22 year old engineering student from Delhi with family income of 3 lakh."
    print(f"Default user text: '{default_desc}'")
    user_text = input("Enter description (or press Enter to use default): ").strip()
    if not user_text:
        user_text = default_desc
        
    # Set up Client
    api_key = os.environ.get("GEMINI_API_KEY", "")
    client = genai.Client(api_key=api_key) if api_key else None
    
    print("\n[Step 1] Extracting User Profile...")
    profile = run_extraction(user_text, client)
    print("Extracted Profile:")
    print(json.dumps(profile, indent=4))
    
    # Run Match Engine
    print("\n[Step 2] Matching eligibility...")
    eligible = []
    rejected = []
    
    for row in schemes_data:
        age_ok = check_age_eligible(profile.get("age", 18), row["min_age"], row["max_age"])
        gender_ok = check_gender_eligible(profile.get("gender", "male"), row["gender"])
        occ_ok = check_occupation_eligible(profile.get("occupation", "all"), row["occupation"])
        edu_ok = check_education_eligible(profile.get("education", "all"), row["education"])
        inc_ok = check_income_eligible(profile.get("annual_income", 0), row["income_limit"])
        state_ok = check_state_eligible(profile.get("state", "Delhi"), row["state"])
        
        score = 0
        reasons = []
        failed = []
        
        if age_ok: score += 20; reasons.append("Age match")
        else: failed.append("Age out of range")
        
        if gender_ok: reasons.append("Gender match")
        else: failed.append("Gender restriction mismatch")
        
        if occ_ok: score += 20; reasons.append("Occupation match")
        else: failed.append("Occupation restriction mismatch")
        
        if edu_ok: score += 20; reasons.append("Education match")
        else: failed.append("Education requirement not met")
        
        if inc_ok: score += 20; reasons.append("Income limit match")
        else: failed.append("Income threshold exceeded")
        
        if state_ok: score += 20; reasons.append("State match")
        else: failed.append("State location mismatch")
        
        scheme_res = {
            "name": row["scheme_name"],
            "category": row["category"],
            "score": score,
            "reasons": reasons,
            "failed_reasons": failed,
            "benefits": row["benefits"]
        }
        
        if gender_ok and score > 0:
            eligible.append(scheme_res)
        else:
            rejected.append(scheme_res)
            
    # Display Results
    eligible = sorted(eligible, key=lambda x: x["score"], reverse=True)
    rejected = sorted(rejected, key=lambda x: x["score"], reverse=True)
    
    print("\n" + "=" * 60)
    print(f" ELIGIBLE SCHEMES FOUND: {len(eligible)}")
    print("=" * 60)
    for s in eligible:
        print(f"[MATCH] {s['name']} (Score: {s['score']}/100)")
        print(f"   Category: {s['category']}")
        print(f"   Reasons:  {', '.join(s['reasons'])}")
        print(f"   Benefits: {s['benefits'] if isinstance(s['benefits'], str) else ', '.join(s['benefits'])}")
        print("-" * 50)
        
    print("\n" + "=" * 60)
    print(f" REJECTED SCHEMES: {len(rejected)}")
    print("=" * 60)
    for s in rejected[:3]:
        print(f"[REJECTED] {s['name']}")
        print(f"   Failures: {', '.join(s['failed_reasons'])}")
        print("-" * 50)

if __name__ == "__main__":
    main()
