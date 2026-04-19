from dotenv import load_dotenv
load_dotenv()

"""
config.py — Med-Assist configuration
"""

# ── LLM ───────────────────────────────────────────────────
MODEL_NAME   = "llama-3.3-70b-versatile"
TEMPERATURE  = 0

# ── Retrieval ─────────────────────────────────────────────
N_RESULTS    = 5          # top-k chunks returned by ChromaDB
EMBED_MODEL  = "all-MiniLM-L6-v2"
COLLECTION   = "med_assist_kb"

# ── Eval / reflection loop ────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.8   # below this → retry answer node
MAX_EVAL_RETRIES       = 2     # safety valve: never retry more than this

# ── Memory ────────────────────────────────────────────────
WINDOW_SIZE  = 6          # max messages kept (3 turns × 2 roles)
