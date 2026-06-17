"""
agent.py — Med-Assist / Asclepius Clinical Diagnosis Assistant

Public exports:
    app        — compiled LangGraph app (MemorySaver)
    ask()      — non-streaming: ask(question, thread_id) -> dict
    ask_stream() — streaming: ask_stream(question, thread_id) -> generator
    CapstoneState
"""

import json
import re
from typing import TypedDict, List, Generator

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import (
    MODEL_NAME, TEMPERATURE,
    FAITHFULNESS_THRESHOLD, MAX_EVAL_RETRIES,
    WINDOW_SIZE, N_RESULTS,
)
from knowledge_base import embedder, collection

load_dotenv()
llm        = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class CapstoneState(TypedDict):
    question:        str
    messages:        List[dict]
    route:           str
    retrieved:       str
    sources:         List[str]
    tool_result:     str
    risk_score_type: str
    answer:          str
    faithfulness:    float
    eval_retries:    int


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL RISK SCORE CALCULATORS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_curb65(params: dict) -> str:
    score, breakdown = 0, []
    if params.get("confusion"):                             score += 1; breakdown.append("Confusion: +1")
    if params.get("urea_gt_7"):                             score += 1; breakdown.append("Urea >7 mmol/L: +1")
    if params.get("rr_ge_30"):                              score += 1; breakdown.append("RR ≥30/min: +1")
    if params.get("sbp_lt_90") or params.get("dbp_le_60"): score += 1; breakdown.append("Low BP: +1")
    if params.get("age_ge_65"):                             score += 1; breakdown.append("Age ≥65: +1")

    if score <= 1:   severity = "Low — consider home treatment"
    elif score == 2: severity = "Moderate — consider hospital admission"
    else:            severity = "High — hospitalise; consider ITU if ≥4"

    return (f"CURB-65 Score: {score}/5\n"
            f"Breakdown: {', '.join(breakdown) or 'No criteria met'}\n"
            f"Severity: {severity}")


def calculate_wells_pe(params: dict) -> str:
    score, breakdown = 0.0, []
    if params.get("dvt_signs"):      score += 3;   breakdown.append("DVT signs: +3")
    if params.get("pe_likely"):      score += 3;   breakdown.append("PE most likely dx: +3")
    if params.get("hr_gt_100"):      score += 1.5; breakdown.append("HR >100: +1.5")
    if params.get("immobilisation"): score += 1.5; breakdown.append("Immobilisation/surgery: +1.5")
    if params.get("prev_dvt_pe"):    score += 1.5; breakdown.append("Previous DVT/PE: +1.5")
    if params.get("haemoptysis"):    score += 1;   breakdown.append("Haemoptysis: +1")
    if params.get("malignancy"):     score += 1;   breakdown.append("Malignancy: +1")

    probability = ("Low/Intermediate — check D-dimer first" if score <= 4
                   else "High — proceed to CTPA without D-dimer")

    return (f"Wells PE Score: {score}/12.5\n"
            f"Breakdown: {', '.join(breakdown) or 'No criteria met'}\n"
            f"Probability: {probability}")


def calculate_cha2ds2_vasc(params: dict) -> str:
    score, breakdown = 0, []
    if params.get("heart_failure"):    score += 1; breakdown.append("Heart failure: +1")
    if params.get("hypertension"):     score += 1; breakdown.append("Hypertension: +1")
    if params.get("age_ge_75"):        score += 2; breakdown.append("Age ≥75: +2")
    elif params.get("age_65_74"):      score += 1; breakdown.append("Age 65-74: +1")
    if params.get("diabetes"):         score += 1; breakdown.append("Diabetes: +1")
    if params.get("prior_stroke_tia"): score += 2; breakdown.append("Prior Stroke/TIA: +2")
    if params.get("vascular_disease"): score += 1; breakdown.append("Vascular disease: +1")
    if params.get("female"):           score += 1; breakdown.append("Female sex: +1")

    if score == 0:   rec = "No anticoagulation needed"
    elif score == 1: rec = "Consider anticoagulation (especially if male)"
    else:            rec = "Anticoagulation recommended (DOAC preferred)"

    return (f"CHA\u2082DS\u2082-VASc Score: {score}/9\n"
            f"Breakdown: {', '.join(breakdown) or 'No risk factors'}\n"
            f"Recommendation: {rec}")


# ══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > WINDOW_SIZE:
        msgs = msgs[-WINDOW_SIZE:]
    return {"messages": msgs}


def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(
        f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for a Clinical Diagnosis Assistant for medical students and junior doctors.

Available options:
- retrieve: search the knowledge base for clinical information about diseases, symptoms, investigations, or treatments
- memory_only: answer from conversation history alone (e.g. 'what did you just say?', 'what diagnosis did you mention?')
- tool: use the clinical risk score calculator (CURB-65, Wells Score, or CHA2DS2-VASc) — use this when the user provides specific patient parameters and asks for a risk score or severity score

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    decision = llm.invoke(prompt).content.strip().lower()
    if "memory" in decision:   decision = "memory_only"
    elif "tool" in decision:   decision = "tool"
    else:                      decision = "retrieve"

    print(f"  [router] → {decision}")
    return {"route": decision}


def retrieval_node(state: CapstoneState) -> dict:
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=N_RESULTS)
    chunks  = results["documents"][0]
    metas   = results["metadatas"][0]
    topics  = [m["topic"] for m in metas]

    # Deduplicate topics for display while preserving order
    seen = set()
    unique_topics = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            unique_topics.append(t)

    context = "\n\n---\n\n".join(
        f"[{metas[i]['topic']} | chunk {metas[i]['chunk']+1}/{metas[i]['n_chunks']}]\n{chunks[i]}"
        for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": unique_topics}


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    question = state["question"]

    extraction_prompt = f"""You are a clinical assistant. The user wants a clinical risk score calculated.
Extract the score type and patient parameters from the question below.

Return ONLY valid JSON with no preamble or markdown fences.

For CURB-65:
{{"score_type": "curb65", "params": {{"confusion": bool, "urea_gt_7": bool, "rr_ge_30": bool, "sbp_lt_90": bool, "dbp_le_60": bool, "age_ge_65": bool}}}}

For Wells PE:
{{"score_type": "wells_pe", "params": {{"dvt_signs": bool, "pe_likely": bool, "hr_gt_100": bool, "immobilisation": bool, "prev_dvt_pe": bool, "haemoptysis": bool, "malignancy": bool}}}}

For CHA2DS2-VASc:
{{"score_type": "cha2ds2", "params": {{"heart_failure": bool, "hypertension": bool, "age_ge_75": bool, "age_65_74": bool, "diabetes": bool, "prior_stroke_tia": bool, "vascular_disease": bool, "female": bool}}}}

If unclear, return: {{"score_type": "unknown", "params": {{}}}}

Question: {question}
JSON:"""

    try:
        raw        = llm.invoke(extraction_prompt).content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        raw_json   = json_match.group(0) if json_match else raw
        data       = json.loads(raw_json)
        score_type = data.get("score_type", "unknown")
        params     = data.get("params", {})

        if score_type == "curb65":     result = calculate_curb65(params)
        elif score_type == "wells_pe": result = calculate_wells_pe(params)
        elif score_type == "cha2ds2":  result = calculate_cha2ds2_vasc(params)
        else:
            result = ("I can calculate: CURB-65 (pneumonia severity), "
                      "Wells Score (PE probability), or CHA\u2082DS\u2082-VASc (AF stroke risk). "
                      "Please provide the relevant patient parameters.")
    except Exception as e:
        score_type = "error"
        result     = f"Clinical calculator error: {e}. Please rephrase with explicit patient parameters."

    return {"tool_result": result, "risk_score_type": score_type}


def _build_system_prompt(state: CapstoneState, is_retry: bool) -> str:
    """Shared system prompt builder used by both answer and stream nodes."""
    retrieved   = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")

    context_parts = []
    if retrieved:   context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result: context_parts.append(f"CLINICAL CALCULATOR RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    retry_note = (
        "\n\nPREVIOUS ANSWER FAILED QUALITY GATE — every claim must be "
        "traceable to an explicit sentence in the KNOWLEDGE BASE above. "
        "If uncertain, omit it."
    ) if is_retry else ""

    if context:
        return f"""You are a Clinical Diagnosis Assistant for medical students and junior doctors.

GROUNDING RULES:
1. Answer using ONLY facts explicitly present in the KNOWLEDGE BASE below.
2. If a specific value is NOT in the context, write [not in knowledge base] rather than guessing.
3. Do NOT draw on your general medical training beyond what is written below.
4. Every bullet point must be directly traceable to the context.
5. For out-of-scope questions (not clinical medicine), politely decline.{retry_note}

{context}"""
    else:
        return ("You are a Clinical Diagnosis Assistant. Answer based on the "
                "conversation history only. Remind the user to verify all "
                "decisions with a senior clinician.")


def answer_node(state: CapstoneState) -> dict:
    """Non-streaming answer node — used by the LangGraph graph."""
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    is_retry     = eval_retries > 0

    system_content = _build_system_prompt(state, is_retry)

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
    lc_msgs.append(HumanMessage(content=state["question"]))

    return {"answer": llm.invoke(lc_msgs).content}


def eval_node(state: CapstoneState) -> dict:
    answer  = state.get("answer", "")
    context = state.get("retrieved", "")
    retries = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    # Out-of-scope refusals are correctly unfaithful to retrieved context
    refusal_signals = ["outside the scope", "only assist with medical",
                       "not able to assist", "cannot help with this"]
    if any(s in answer.lower() for s in refusal_signals):
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""You are a medical fact-checker. Score faithfulness of the ANSWER against the CONTEXT.
Output ONLY a single number: 0.0, 0.2, 0.4, 0.6, 0.8, or 1.0

1.0 = every claim traceable to the context
0.8 = nearly all claims grounded, 1-2 minor additions
0.6 = some claims go beyond the context
0.4 = several claims not supported
0.0 = ignores or contradicts the context

CONTEXT:
{context}

ANSWER:
{answer}

NUMBER:"""

    result = llm.invoke(prompt).content.strip()
    try:
        match = re.search(r"\b(0\.0|0\.2|0\.4|0\.6|0\.8|1\.0|0|1)\b", result)
        if match:
            score = float(match.group(1).replace(",", "."))
        else:
            match_any = re.search(r"\d+\.\d+|\d+", result)
            score = float(match_any.group(0).replace(",", ".")) if match_any else 0.5
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5

    gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️"
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    return {"messages": messages + [{"role": "assistant", "content": state["answer"]}]}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def route_decision(state: CapstoneState) -> str:
    r = state.get("route", "retrieve")
    if r == "tool":        return "tool"
    if r == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    if (state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD
            or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES):
        return "save"
    return "answer"


def _build_graph():
    graph = StateGraph(CapstoneState)

    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")

    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    return graph.compile(checkpointer=MemorySaver())


app = _build_graph()
print("✅ Agent compiled and ready")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ask(question: str, thread_id: str = "default") -> dict:
    """
    Non-streaming invocation. Returns the full state dict when complete.
    Key fields: answer, faithfulness, route, sources.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"question": question}, config=config)


def ask_stream(question: str, thread_id: str = "default") -> Generator:
    """
    Streaming invocation. Runs the full pipeline (memory → router →
    retrieve/tool → eval) non-streaming, then simulates streaming of the final
    verified answer token-by-token.

    Yields:
        str tokens as they arrive.
        The final item yielded is a dict with metadata:
        {"__meta__": True, "faithfulness": float, "route": str, "sources": list}

    Usage in Streamlit:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            meta = {}
            for chunk in ask_stream(prompt, thread_id):
                if isinstance(chunk, dict):
                    meta = chunk
                else:
                    full += chunk
                    placeholder.markdown(full + "▌")
            placeholder.markdown(full)
    """
    import time
    config = {"configurable": {"thread_id": thread_id}}

    # Run the entire graph. The graph computes the verified answer.
    partial = app.invoke({"question": question}, config=config)
    route      = partial.get("route", "retrieve")
    sources    = partial.get("sources", [])
    faith      = partial.get("faithfulness", 1.0)
    answer     = partial.get("answer", "")

    # Yield the verified answer in small chunks to simulate streaming
    # Split by spaces to preserve word boundaries
    words = answer.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield chunk
        time.sleep(0.015)  # Natural reading delay

    # Final metadata token
    yield {"__meta__": True, "faithfulness": faith, "route": route, "sources": sources}
