# Testing Suite for Youth & Employment Scheme Navigator

This directory contains test cases to validate the deterministic matching logic and LLM extraction capabilities of the application.

## Test Queries

The test cases are saved in [queries.json](file:///e:/Hackathon/testing/queries.json). Each test case represents a unique candidate profile designed to trigger specific eligibility criteria, including caste, income, age, gender, education, and location constraints.

| Test ID | Scenario Description | Expected Primary Match | Key Verification |
| :--- | :--- | :--- | :--- |
| **TC-001** | SC caste student with low income | SC Post Matric Scholarship | Verify caste and income restrictions. |
| **TC-002** | Young male ITI diploma holder | NAPS & PMKVY | Verify apprenticeship eligibility. |
| **TC-003** | Unemployed BA graduate (22 years) | PM Internship Scheme (PMIS) | Verify age (21-24 limit) and education match. |
| **TC-004** | Female B.Tech student (low income) | AICTE Pragati Scholarship | Verify female gender constraint + tech degree. |
| **TC-005** | Older rural unskilled laborer | MGNREGA | Verify rural/manual work matching. |
| **TC-006** | 14-year-old girl in class 9 | NMMS & NSIGSE | Verify class 8/9 school boundaries and age. |
| **TC-007** | Rural BPL candidate from Odisha | DDU-GKY | Verify rural BPL matching and free lodging filter. |
| **TC-008** | High income engineering graduate | NCS (Only) | Verify income exclusion (excludes PMIS/Pragati). |
| **TC-009** | 26-year-old unemployed graduate | PMKVY & NCS | Verify PMIS age exclusion (fails since age > 24). |
| **TC-010** | 29-year-old vocational trainee | PMKVY | Verify vocational category matching. |
| **TC-011** | 9-year-old child (underage boundary) | None | Verify lower-limit age exclusion (matches no schemes). |
| **TC-012** | 68-year-old retired senior citizen | None | Verify upper-limit age exclusion (matches no schemes). |
| **TC-013** | Employed government office clerk | None | Verify occupation-based exclusion from benefits. |
| **TC-014** | Illiterate youth seeking B.Tech scholarship | PMKVY | Verify exclusion from technical degree scholarships. |
| **TC-015** | General candidate check | NCS | Verify general portal registry eligibility. |

---

## How to Test

### Manual UI Testing
1. Copy the text from the `"query"` field of any test case in `queries.json`.
2. Paste it into the **Describe Yourself** text area of the application.
3. Click **Find My Schemes**.
4. Verify that:
   - The **Extracted Profile** metrics match the properties in `"expected_profile"`.
   - The schemes listed under **Eligible Schemes** match those in `"expected_matches"`.
   - Rejections listed under **Why Not Eligible?** show correct failed criteria details.
