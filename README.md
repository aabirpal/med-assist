# ⚕️ Med-Assist : Medical Diagnosis Assistant

An evidence-grounded medical decision support agent for medical students and junior doctors. Built with LangGraph, ChromaDB RAG, and a domain-specific medical risk score calculator.

## What it does

Med-Assist accepts natural-language medical questions and:

- **Retrieves** grounded answers from a 12-document medical knowledge base (ChromaDB)
- **Calculates** validated medical risk scores deterministically (CURB-65, Wells PE, CHA₂DS₂-VASc)
- **Remembers** conversation context across multi-turn consultations (MemorySaver + sliding window)
- **Self-reflects** via a faithfulness evaluator that retries hallucinated answers (threshold 0.80)
- **Declines** out-of-scope questions gracefully

---

## Architecture

```
User → memory → router → [retrieve | skip | tool] → answer → eval → [retry | save] → User
```

| Node | File | Role |
|---|---|---|
| `memory` | `agent.py` | Appends question; 6-message sliding window |
| `router` | `agent.py` | Classifies as `retrieve` / `memory_only` / `tool` |
| `retrieve` | `agent.py` | ChromaDB top-5 chunk lookup |
| `skip` | `agent.py` | No-op for memory_only route |
| `tool` | `agent.py` | LLM extracts params → deterministic calculator |
| `answer` | `agent.py` | LLM call with strict grounding rules |
| `eval` | `agent.py` | Faithfulness gate; triggers retry if < 0.80 |
| `save` | `agent.py` | Appends answer to history |

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/aabirpal/med-assist.git
cd med-assist

# 2. Create and activate virtual environment
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# edit .env → GROQ_API_KEY=...

# 5. Run the UI
streamlit run capstone_streamlit.py
```

---

## Medical Calculator Tool

Ask in plain English with patient parameters:

```
CURB-65: age 70, confused, urea 9, RR 32, BP 110/70
Wells PE score: DVT signs present, HR 115, immobile for 5 days
CHA2DS2-VASc: female, 74 years old, hypertension and diabetes
```

---

> ⚠️ All medical decisions must be verified by a qualified senior medician.
