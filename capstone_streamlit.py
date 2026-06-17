"""
capstone_streamlit.py — Med-Assist Clinical Diagnosis Assistant
Streaming Streamlit UI. All logic lives in agent.py and knowledge_base/.

Run: streamlit run capstone_streamlit.py
"""

import uuid
import streamlit as st

from agent import ask_stream
from knowledge_base import DOCUMENTS, collection

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Med-Assist — Clinical Diagnosis Assistant",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("⚕️ Med-Assist — Clinical Diagnosis Assistant")
st.caption("Evidence-based clinical decision support for medical students and junior doctors")
st.success(f"✅ Knowledge base loaded — {collection.count()} chunks from {len(DOCUMENTS)} documents")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚕️ Med-Assist")
    st.write("Evidence-based clinical decision support for medical students and junior doctors.")
    st.divider()
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    st.divider()

    st.write("**📚 Clinical Topics Covered:**")
    for doc in DOCUMENTS:
        st.write(f"• {doc['topic']}")
    st.divider()

    st.write("**🧮 Clinical Calculator Tool:**")
    st.info(
        "Provide patient parameters and ask for a score:\n\n"
        "- *CURB-65: age 70, confused, urea 9, RR 32, BP 110/70*\n"
        "- *Wells PE: DVT signs, HR 115, immobile 5 days*\n"
        "- *CHA₂DS₂-VASc: female, 74y, hypertension, diabetes*"
    )
    st.divider()
    st.caption("⚠️ For educational use only.\nAlways verify with a senior clinician.")

    if st.button("🗑️ New Consultation"):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a clinical question or request a risk score..."):

    # Show user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        meta        = {}

        try:
            with st.spinner("Consulting knowledge base..."):
                for chunk in ask_stream(prompt, thread_id=st.session_state.thread_id):
                    if isinstance(chunk, dict) and chunk.get("__meta__"):
                        meta = chunk
                    else:
                        full_answer += chunk
                        # Show the growing answer with a blinking cursor
                        placeholder.markdown(full_answer + "▌")

            # Final render without cursor
            placeholder.markdown(full_answer)

            # Metadata caption
            if meta:
                faith   = meta.get("faithfulness", 0.0)
                route   = meta.get("route", "?")
                sources = meta.get("sources", [])
                caption = f"🔀 Route: `{route}` | 🎯 Faithfulness: `{faith:.2f}`"
                if sources:
                    caption += f" | 📄 {', '.join(sources[:2])}"
                st.caption(caption)

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
        except Exception as e:
            placeholder.empty()
            st.error(f"⚠️ Clinical Diagnosis Assistant error: {e}")
