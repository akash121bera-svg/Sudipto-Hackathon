# Application Architecture & Flow Diagram Spec

This document describes the architectural pipeline of the **AI Youth & Employment Scheme Navigator**. 

You can paste the **Mermaid.js code block** or the **diagram prompt** below directly into ChatGPT (or any Mermaid renderer like [mermaid.live](https://mermaid.live)) to generate a clean visual flowchart of the application's lifecycle.

---

## 1. Mermaid.js Flowchart Code
*Paste this code block directly into ChatGPT or a Mermaid editor to render the graphical flow:*

```mermaid
graph TD
    %% Define Styles
    classDef user fill:#EBF8FF,stroke:#3182CE,stroke-width:2px;
    classDef py fill:#EDF2F7,stroke:#4A5568,stroke-width:2px;
    classDef gemini fill:#E2E8F0,stroke:#805AD5,stroke-width:2px,stroke-dasharray: 5 5;
    classDef groq fill:#FEFCBF,stroke:#B7791F,stroke-width:2px;
    classDef database fill:#EDFFF4,stroke:#38A169,stroke-width:2px;

    %% Elements
    A[User inputs profile description text]:::user
    B[Trigger: Find My Schemes]:::user
    
    %% API Extraction Phase
    C{Is Gemini Client Active?}:::py
    D[Call Gemini 2.5 Flash API]:::gemini
    E[Gemini Output: JSON UserProfile]:::gemini
    F{Did Gemini Fail or Quota Limit 429/503?}:::py
    
    G{Is Groq Client Active?}:::py
    H[Call Groq Llama 3.3 API]:::groq
    I[Groq Output: JSON UserProfile]:::groq
    J{Did Groq Fail?}:::py
    
    K[Load Local Hardcoded Mocks]:::py
    
    %% Match Engine Phase
    L[Consolidated Profile Object]:::py
    M[(schemes.json Dataset)]:::database
    N[Execute Deterministic Python Eligibility Engine]:::py
    O[Filter Schemes by age/gender/occupation/income/state/education]:::py
    
    %% Output Classification
    P[Output: Eligible Schemes List]:::py
    Q[Output: Rejected Schemes with failed_reasons]:::py
    
    %% Secondary RAG Generation Phase
    R[Selected Scheme for Explanation]:::user
    S[Call Gemini or Groq fallback or Local Offline summary template]:::py
    T[Display Custom Citizen-Friendly Explanation]:::user
    U[Generate Schemes Comparison Grid]:::py

    %% Connections
    A --> B
    B --> C
    
    C -- Yes --> D
    D --> E
    E --> L
    
    D -- Exception --> F
    C -- No --> F
    
    F -- Yes --> G
    G -- Yes --> H
    H --> I
    I --> L
    
    H -- Exception --> J
    G -- No --> J
    
    J -- Yes --> K
    K --> L
    
    L --> N
    M --> N
    N --> O
    O --> P
    O --> Q
    
    P --> R
    R --> S
    S --> T
    P --> U
```

---

## 2. ChatGPT Prompt for Graphical Diagram Generation
*If you want ChatGPT to draw an image or generate a structured visual architecture drawing, copy and paste the text below:*

```text
Please generate a polished, professional graphical block diagram for this Python-Streamlit AI application architecture. Use the following specifications:

1. User Input Layer:
   - Plain text input describing candidate age, occupation, income, caste, state, and education.

2. AI Extraction & Fallback Cascade Layer:
   - Primary: Gemini API (gemini-2.5-flash) with structured JSON response schema (UserProfile Pydantic model).
   - Secondary Fallback: Groq API (llama-3.3-70b-versatile) compatible chat completion.
   - Tertiary Fallback: Local deterministic keyword-based mock profiles.
   - Error Handling: Errors are captured silently, logged to console, and clean notices are displayed in Streamlit.

3. Rule-Based Eligibility Engine (Local Python Matcher):
   - Loads local 'schemes.json' dataset.
   - Deterministic matching functions compare UserProfile properties against scheme constraints (Age ranges, gender filters, education ranks, state residence, and parent income limits).

4. UI Output Presentation Layer:
   - Match Results: Displays "Eligible Schemes" list and reasons for match.
   - Rejections: Displays "Why Not Eligible?" with detailed failure causes.
   - Dynamic RAG Summary: Feeds chosen scheme details to LLM (Gemini -> Groq -> Local Template) to build a custom "citizen-friendly explanation."
   - Comparison Grid: Generates comparison table using LLM or local Markdown formatter.

Please render this flowchart visually or provide a clear structural diagram illustrating this pipeline.
```
