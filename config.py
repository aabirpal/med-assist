"""
config.py — Med-Assist configuration
Single source of truth. Import constants from here; never hardcode them elsewhere.
"""

from dotenv import load_dotenv
load_dotenv()

# ── LLM ───────────────────────────────────────────────────
MODEL_NAME   = "llama-3.3-70b-versatile"
TEMPERATURE  = 0

# ── Retrieval ─────────────────────────────────────────────
N_RESULTS    = 6          # top-k chunks returned by ChromaDB (raised from 5
                          # because chunked docs are smaller — need more chunks
                          # to cover the same information)
EMBED_MODEL  = "all-MiniLM-L6-v2"
COLLECTION   = "med_assist_kb"

# ── Persistence ───────────────────────────────────────────
CHROMA_PATH  = "./chroma_db"   # folder where ChromaDB persists to disk
                               # delete this folder to force a full rebuild
                               # after editing knowledge base documents (e.g. manifest.json or text files)

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE    = 350   # target words per chunk
CHUNK_OVERLAP = 60    # word overlap between adjacent chunks — prevents
                      # answers being cut mid-sentence at a boundary

# ── Eval / reflection loop ────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.8
MAX_EVAL_RETRIES       = 2

# ── Memory ────────────────────────────────────────────────
WINDOW_SIZE  = 6   # max messages kept (3 turns × 2 roles)
