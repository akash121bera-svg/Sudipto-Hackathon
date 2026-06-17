# AI Youth & Employment Scheme Navigator

An intelligent, high-performance web companion that matches Indian citizens to youth development, internship, scholarship, and employment schemes. 

Built using **Streamlit**, the application uses a hybrid architecture combining structured Large Language Model (LLM) extraction with a deterministic local rule matching engine.

---

## 1. Application Flow & Architecture

Below is the technical flow of the application. The system implements a **three-tier LLM fallback cascade** for candidate profile extraction and a **secondary RAG cascade** for generating citizen-friendly explanations.

![Application Flow Diagram](flowdiagram/architecture_diagram.png)

---

## 2. Features
* 🔒 **Privacy-First**: No personal data is stored; matching is calculated entirely in-memory.
* ⚡ **High Reliability**: If the default Gemini API key encounters rate limits (`429 Resource Exhausted`) or server outages (`503 Unavailable`), the system automatically switches to the **Groq API** or offline fallback templates.
* 📋 **Matched Analysis**: Provides scores and direct links to active portals (e.g. Skill India Digital).
* ❌ **Transparent Rejections**: Lists exactly why a user did not qualify for a scheme (e.g. income limit exceeded or age out of boundaries).

---

## 3. Quick Start

### Prerequisites
* Python 3.10 or higher installed.
* A terminal setup with Git.

### Setup & Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/akash121bera-svg/Sudipto-Hackathon.git
   cd Sudipto-Hackathon
   ```

2. **Activate the Virtual Environment**:
   * **PowerShell**:
     ```powershell
     & .venv/Scripts/Activate.ps1
     ```
   * **Command Prompt**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **Git Bash / Linux / macOS**:
     ```bash
     source .venv/Scripts/activate
     ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root folder with your keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_gsk_api_key
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch the Web App**:
   ```bash
   streamlit run app.py
   ```

---

## 4. Testing Suite

The repository contains a testing suite inside the `testing/` directory:
* **[testing/queries.json](testing/queries.json)**: A JSON collection of 15 candidate profiles covering target matches and strict edge cases (underage checks, overage checks, invalid occupation checks).
* **[testing/test_queries.pdf](testing/test_queries.pdf)**: A professionally compiled PDF listing all test queries and expected matches.
* **[testing/README.md](testing/README.md)**: A user guide explaining how to execute and assert the test cases.
