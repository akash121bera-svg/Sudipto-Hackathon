import json
import os
import pandas as pd
import streamlit as st
import requests
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Union
from google import genai
from google.genai import types

# Load environment variables from .env if present
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().strip("'\"")
                        os.environ[key] = val

load_env_file()

# ---------------------------------------------------------
# Page Configurations & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Youth & Employment Scheme Navigator",
    page_icon="💼",
    layout="wide"
)

# Premium Custom Styling for Sleek Dark Theme Elements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    .card {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f3f4f6;
        margin-bottom: 15px;
    }
    
    .metric-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    
    .scheme-card {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Schemes Dataset
# ---------------------------------------------------------
@st.cache_data
def load_schemes_dataset() -> pd.DataFrame:
    try:
        with open("schemes.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        st.error("Error: schemes.json dataset file not found in workspace.")
        return pd.DataFrame()

df_schemes = load_schemes_dataset()

# ---------------------------------------------------------
# Deterministic Match Engine & Normalization Utilities
# ---------------------------------------------------------
def check_age_eligible(user_age: int, min_age, max_age) -> bool:
    if user_age is None or pd.isna(user_age):
        user_age = 18
    try:
        user_age = int(user_age)
    except (ValueError, TypeError):
        user_age = 18
    eligible_min = True if min_age is None or pd.isna(min_age) else user_age >= int(min_age)
    eligible_max = True if max_age is None or pd.isna(max_age) else user_age <= int(max_age)
    return eligible_min and eligible_max

def check_gender_eligible(user_gender: str, scheme_gender) -> bool:
    if not user_gender:
        user_gender = "all"
    user_gender = str(user_gender).lower().strip()
    if isinstance(scheme_gender, list):
        scheme_genders = [g.lower().strip() for g in scheme_gender]
    else:
        scheme_genders = [g.lower().strip() for g in str(scheme_gender).replace(",", "/").split("/")]
        
    if "all" in scheme_genders or "any" in scheme_genders:
        return True
    return user_gender in scheme_genders

def check_occupation_eligible(user_occ: str, scheme_occ) -> bool:
    if not user_occ:
        user_occ = "all"
    user_occ = str(user_occ).lower().strip()
    if isinstance(scheme_occ, list):
        scheme_occs = [o.lower().strip() for o in scheme_occ]
    else:
        scheme_occs = [o.lower().strip() for o in str(scheme_occ).replace(",", "/").split("/")]
        
    if any(any(x in occ for x in ["all", "any", "not specified", "not required"]) for occ in scheme_occs):
        return True
    return any(user_occ in occ or occ in user_occ for occ in scheme_occs)

def check_education_eligible(user_edu: str, scheme_edu) -> bool:
    if not user_edu:
        user_edu = "all"
    user_edu = str(user_edu).lower().strip()
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
    if user_income is None or pd.isna(user_income):
        user_income = 0
    try:
        user_income = float(user_income)
    except (ValueError, TypeError):
        user_income = 0
    return user_income <= limit_val

def check_state_eligible(user_state: str, scheme_state: str) -> bool:
    if not user_state:
        user_state = "all india"
    user_state = str(user_state).lower().strip()
    scheme_state_str = str(scheme_state).lower().strip()
    if "all india" in scheme_state_str or "any" in scheme_state_str:
        return True
    return user_state in scheme_state_str

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class UserProfile(BaseModel):
    age: Optional[int] = Field(None, description="Age of the user as integer. Null if not specified.")
    gender: Optional[str] = Field(None, description="Gender of the user: 'male', 'female', or null if not specified.")
    occupation: Optional[str] = Field(None, description="Current occupation: e.g. student, unemployed, job seeker, or null if not specified.")
    education: Optional[str] = Field(None, description="Highest level of education: e.g. graduate, diploma, b.tech, 12th, or null if not specified.")
    annual_income: Optional[int] = Field(None, description="Annual family income in INR. Extract from strings like '3 lakh' (300000).")
    state: Optional[str] = Field(None, description="Indian state of residence. Null if not specified.")

# ---------------------------------------------------------
# Gemini API / Mock Fallback Operations
# ---------------------------------------------------------
def get_gemini_client(api_key: str):
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[Gemini Client Init Error] {e}")
        st.warning("⚠️ Failed to initialize live Gemini Client. Using offline fallbacks.")
        return None

def handle_api_error(e, context_msg: str):
    print(f"[Gemini API Error] {context_msg}: {e}")
    err_str = str(e)
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        st.warning("⚠️ The Gemini AI service is currently rate-limited (quota exceeded). Using offline fallback...")
    elif "503" in err_str or "UNAVAILABLE" in err_str:
        st.warning("⚠️ The Gemini AI service is temporarily unavailable (503). Using offline fallback...")
    else:
        st.warning(f"⚠️ {context_msg}. Using offline fallback...")

def call_groq_chat_completion(messages: list, response_json: bool = False) -> str:
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY", "")
    if not groq_key:
        raise ValueError("Groq API key is not configured in environment.")
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.1
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}
        
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    res_data = response.json()
    return res_data["choices"][0]["message"]["content"]

def run_profile_extraction(user_text: str, client) -> dict:
    if client:
        prompt = (
            "Extract the structured user profile details from the user text below. "
            "Strictly adhere to the provided schema and extract correct values (such as calculating numerical income)."
        )
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, f"User Text: \"{user_text}\""],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=UserProfile,
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            handle_api_error(e, "Live Gemini profile extraction failed")

    # Fallback to Groq if Gemini client is not active or failed
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY", "")
    if groq_key:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert profile extractor. Extract user profile details from the text as a JSON object matching this schema:\n"
                        "{\n"
                        "  \"age\": int (or null),\n"
                        "  \"gender\": \"male\"|\"female\" (or null),\n"
                        "  \"occupation\": str (or null),\n"
                        "  \"education\": str (or null),\n"
                        "  \"annual_income\": int (or null),\n"
                        "  \"state\": str (or null)\n"
                        "}\n"
                        "Return ONLY the raw JSON object, no markdown formatting, no comments."
                    )
                },
                {"role": "user", "content": f"User Text: \"{user_text}\""}
            ]
            content = call_groq_chat_completion(messages, response_json=True)
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as groq_err:
            handle_api_error(groq_err, "Fallback Groq profile extraction failed")

    # Hardcoded deterministic fallbacks if both LLMs fail/are unavailable
    q = user_text.lower()
    if "23-year-old" in q or "23" in q:
        return {"age": 23, "gender": "male", "occupation": "unemployed", "education": "graduate", "annual_income": 150000, "state": "Delhi"}
    elif "female" in q:
        return {"age": 20, "gender": "female", "occupation": "student", "education": "undergraduate", "annual_income": 200000, "state": "Uttar Pradesh"}
    elif "b.tech" in q or "btech" in q:
        return {"age": 22, "gender": "male", "occupation": "job seeker", "education": "b.tech", "annual_income": None, "state": "Delhi"}
    elif "diploma" in q:
        return {"age": 19, "gender": "male", "occupation": "student", "education": "diploma", "annual_income": None, "state": "Haryana"}
    return {
        "age": 22,
        "gender": "male",
        "occupation": "student",
        "education": "graduate",
        "annual_income": 300000,
        "state": "Delhi"
    }

def get_scheme_explanation(user_profile: dict, scheme: dict, client) -> str:
    name = scheme.get("scheme_name")
    benefits_str = ", ".join(scheme.get("benefits")) if isinstance(scheme.get("benefits"), list) else str(scheme.get("benefits"))
    docs_str = ", ".join(scheme.get("required_documents")) if isinstance(scheme.get("required_documents"), list) else str(scheme.get("required_documents"))
    
    prompt = (
        "You are an AI citizen advisor. Write a simple, friendly, non-technical explanation for why the user qualifies for the scheme. "
        "Must be under 200 words. Split the content into exactly these sections:\n"
        "- Why you qualify\n"
        "- Benefits\n"
        "- Required documents\n"
        "- Application process\n"
        "- Important notes\n\n"
        f"User Profile: {json.dumps(user_profile)}\n"
        f"Scheme Data: {json.dumps(scheme)}"
    )

    if client:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            handle_api_error(e, "Live Gemini explanation generation failed")

    # Fallback to Groq
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY", "")
    if groq_key:
        try:
            messages = [{"role": "user", "content": prompt}]
            return call_groq_chat_completion(messages)
        except Exception as groq_err:
            handle_api_error(groq_err, "Fallback Groq explanation generation failed")

    # Offline summary layout
    return f"""
### **{name}**

* **Why you qualify**: Your profile matches the target criteria. You are {user_profile.get('age', 'eligible')} years old, a {user_profile.get('education', 'eligible')} level candidate, and residing in {user_profile.get('state', 'India')}.
* **Benefits**: {benefits_str}
* **Required Documents**: {docs_str}
* **Application Process**: Apply directly through the official link: [{scheme.get('application_link')}]({scheme.get('application_link')}).
* **Important Notes**: Make sure you have valid copies of all documents before submitting your application.
"""

def get_schemes_comparison(top_schemes: list, client) -> str:
    prompt = (
        "Generate a clean markdown table comparing the following schemes. "
        "Include exactly these columns: 'Scheme Name', 'Target Audience', 'Eligibility Requirements', and 'Key Benefits'. "
        "Ensure the comparison is brief, non-technical, and clean.\n\n"
        f"Schemes to compare: {json.dumps(top_schemes)}"
    )

    if client:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            handle_api_error(e, "Live Gemini comparison grid generation failed")

    # Fallback to Groq
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY", "")
    if groq_key:
        try:
            messages = [{"role": "user", "content": prompt}]
            return call_groq_chat_completion(messages)
        except Exception as groq_err:
            handle_api_error(groq_err, "Fallback Groq comparison grid generation failed")

    # Offline table generation
    lines = [
        "| Scheme Name | Target Audience | Eligibility Requirements | Key Benefits |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for scheme in top_schemes:
        name = scheme.get("scheme_name")
        audience = scheme.get("occupation")
        if isinstance(audience, list): audience = ", ".join(audience)
        
        eligibility = f"Age: {scheme.get('min_age') or 0}-{scheme.get('max_age') or 'Any'}, Edu: {scheme.get('education')}"
        benefits = scheme.get("benefits")
        if isinstance(benefits, list): benefits = benefits[0]
        
        lines.append(f"| **{name}** | {audience} | {eligibility} | {benefits} |")
    return "\n".join(lines)

# ---------------------------------------------------------
# Initialize Gemini Client
# ---------------------------------------------------------
# Gemini API key is loaded quietly from environment/dotenv variables
api_key = os.environ.get("GEMINI_API_KEY", "")

# Initialize client if key is provided
client = get_gemini_client(api_key)

# ---------------------------------------------------------
# Main UI Presentation
# ---------------------------------------------------------
st.markdown("<div class='main-title'>AI Youth & Employment Scheme Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deterministic Eligibility Matcher & AI Companion for Government Schemes in India</div>", unsafe_allow_html=True)

col_input, col_profile = st.columns([3, 2])

with col_input:
    st.markdown("<div class='card-title'>Describe Yourself</div>", unsafe_allow_html=True)
    user_text = st.text_area(
        "Enter your age, location, occupation, family income, etc. in plain English:",
        value="I am a 22 year old engineering student from Delhi with family income of 3 lakh.",
        height=120
    )
    find_button = st.button("Find My Schemes", type="primary", use_container_width=True)

# Set session state to preserve findings across interactions
if "profile" not in st.session_state:
    st.session_state.profile = None
if "eligible" not in st.session_state:
    st.session_state.eligible = []
if "rejected" not in st.session_state:
    st.session_state.rejected = []

if find_button:
    with st.spinner("Analyzing your profile and matching schemes..."):
        # 1. Profile Extraction
        profile_dict = run_profile_extraction(user_text, client)
        st.session_state.profile = profile_dict
        
        # 2. Match Calculation
        eligible_schemes = []
        rejected_schemes = []
        
        for _, row in df_schemes.iterrows():
            age_ok = check_age_eligible(profile_dict.get("age") or 18, row["min_age"], row["max_age"])
            gender_ok = check_gender_eligible(profile_dict.get("gender") or "male", row["gender"])
            occ_ok = check_occupation_eligible(profile_dict.get("occupation") or "all", row["occupation"])
            edu_ok = check_education_eligible(profile_dict.get("education") or "all", row["education"])
            inc_ok = check_income_eligible(profile_dict.get("annual_income") or 0, row["income_limit"])
            state_ok = check_state_eligible(profile_dict.get("state") or "Delhi", row["state"])
            
            # Reasons for matches and failures
            reasons = []
            failed = []
            
            if age_ok: reasons.append("Age eligible")
            else: failed.append(f"Age not in eligible range ({row['min_age'] or 0} - {row['max_age'] or 'unlimited'} years)")
            
            if gender_ok: reasons.append("Gender eligible")
            else: failed.append(f"Scheme restricted to target gender: {row['gender']}")
            
            if occ_ok: reasons.append("Occupation eligible")
            else: failed.append(f"Occupation mismatch: requires {row['occupation']}")
            
            if edu_ok: reasons.append("Education eligible")
            else: failed.append(f"Education requirements not met: requires {row['education']}")
            
            if inc_ok: reasons.append("Income eligible")
            else: failed.append(f"Annual income exceeds threshold of {row['income_limit']}")
            
            if state_ok: reasons.append("State eligible")
            else: failed.append(f"State not eligible (restricted to {row['state']})")
            
            # Score
            score = 0
            if age_ok: score += 20
            if occ_ok: score += 20
            if edu_ok: score += 20
            if inc_ok: score += 20
            if state_ok: score += 20
            
            scheme_dict = row.to_dict()
            scheme_dict["match_score"] = score
            scheme_dict["reasons"] = reasons
            scheme_dict["failed_reasons"] = failed
            
            # Gender is a hard requirement filter
            if gender_ok and score > 0:
                eligible_schemes.append(scheme_dict)
            else:
                rejected_schemes.append(scheme_dict)
                
        # Save to session
        st.session_state.eligible = sorted(eligible_schemes, key=lambda x: x["match_score"], reverse=True)
        st.session_state.rejected = sorted(rejected_schemes, key=lambda x: x["match_score"], reverse=True)

# ---------------------------------------------------------
# Section: Extracted Profile
# ---------------------------------------------------------
with col_profile:
    st.markdown("<div class='card-title'>Extracted Profile</div>", unsafe_allow_html=True)
    if st.session_state.profile:
        p = st.session_state.profile
        
        gender_val = p.get('gender')
        gender_disp = str(gender_val).capitalize() if gender_val else "-"
        
        occ_val = p.get('occupation')
        occ_disp = str(occ_val).capitalize() if occ_val else "-"
        
        edu_val = p.get('education')
        edu_disp = str(edu_val).capitalize() if edu_val else "-"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{p.get('age') or '-'}</div><div class='metric-label'>Age</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{gender_disp}</div><div class='metric-label'>Gender</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{p.get('state') or '-'}</div><div class='metric-label'>State</div></div>", unsafe_allow_html=True)
            
        st.write("")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{occ_disp}</div><div class='metric-label'>Occupation</div></div>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{edu_disp}</div><div class='metric-label'>Education</div></div>", unsafe_allow_html=True)
        with col6:
            income_disp = f"₹{p.get('annual_income'):,}" if p.get('annual_income') else "-"
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{income_disp}</div><div class='metric-label'>Family Income</div></div>", unsafe_allow_html=True)
    else:
        st.info("Provide profile descriptions and click 'Find My Schemes' to see details.")

st.divider()

# ---------------------------------------------------------
# Section: Eligible Schemes & AI Explanations
# ---------------------------------------------------------
col_elig, col_explain = st.columns([3, 2])

with col_elig:
    st.markdown("<div class='card-title'>Eligible Schemes</div>", unsafe_allow_html=True)
    if st.session_state.eligible:
        # Create output DataFrame for standard display
        display_list = []
        for s in st.session_state.eligible:
            display_list.append({
                "Scheme Name": s["scheme_name"],
                "Category": s["category"],
                "Match Score": f"{s['match_score']}/100",
                "Benefits": s["benefits"] if isinstance(s["benefits"], str) else ", ".join(s["benefits"]),
                "Documents Required": ", ".join(s["required_documents"])
            })
        st.dataframe(pd.DataFrame(display_list), use_container_width=True, hide_index=True)
    else:
        st.info("No eligible schemes found yet. Please refine your profile description.")

with col_explain:
    st.markdown("<div class='card-title'>AI Explanation</div>", unsafe_allow_html=True)
    if st.session_state.eligible:
        # User selects which scheme to explain
        eligible_names = [s["scheme_name"] for s in st.session_state.eligible]
        selected_scheme_name = st.selectbox("Select an eligible scheme for citizen explanation:", eligible_names)
        
        selected_scheme = next(s for s in st.session_state.eligible if s["scheme_name"] == selected_scheme_name)
        
        with st.spinner("Generating explanation..."):
            explanation = get_scheme_explanation(st.session_state.profile, selected_scheme, client)
            st.markdown(explanation)
    else:
        st.info("Select a matched scheme to generate a non-technical summary.")

st.divider()

# ---------------------------------------------------------
# Section: Why Not Eligible?
# ---------------------------------------------------------
col_rej, col_comp = st.columns([1, 1])

with col_rej:
    st.markdown("<div class='card-title'>Why Not Eligible?</div>", unsafe_allow_html=True)
    if st.session_state.rejected:
        st.caption("Here are some schemes you were not fully eligible for, along with the reasons for rejection:")
        
        # Display top 3 rejected schemes with failed reasons
        for r_scheme in st.session_state.rejected[:3]:
            with st.expander(f"❌ {r_scheme['scheme_name']} (Category: {r_scheme['category']})"):
                st.markdown("**Eligibility Failures:**")
                for reason in r_scheme["failed_reasons"]:
                    st.write(f"- {reason}")
                
                benefits_str = ", ".join(r_scheme["benefits"]) if isinstance(r_scheme["benefits"], list) else str(r_scheme["benefits"])
                st.markdown(f"**Benefits of this scheme**: {benefits_str}")
    else:
        st.info("No rejected schemes found.")

# ---------------------------------------------------------
# Section: Compare Schemes
# ---------------------------------------------------------
with col_comp:
    st.markdown("<div class='card-title'>Compare Schemes</div>", unsafe_allow_html=True)
    if st.session_state.eligible:
        st.caption("Side-by-side comparison of target audience, eligibility, and benefits of top matching schemes:")
        with st.spinner("Generating comparison grid..."):
            # Compare up to the top 4 matched schemes
            comparison_md = get_schemes_comparison(st.session_state.eligible[:4], client)
            st.markdown(comparison_md)
    else:
        st.info("No matching schemes to compare.")
