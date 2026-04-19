"""
agent.py — Med-Assist Medical Diagnosis Assistant
Single-file agent module. Imports from knowledge_base/ and config.py.

Public exports:
    app       — compiled LangGraph app (MemorySaver checkpointer)
    ask()     — convenience wrapper: ask(question, thread_id) -> dict
    CapstoneState — TypedDict (useful for type hints in the notebook)

Usage:
    from agent import app, ask
    result = ask("What is the CURB-65 score?", thread_id="session-1")
    print(result["answer"])
"""

import json
import os
from typing import TypedDict, List

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
llm = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — State TypedDict
# Every field a node reads or writes declared here.
# ══════════════════════════════════════════════════════════════════════════════

class CapstoneState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    question:        str          # user's current medical question

    # ── Memory ─────────────────────────────────────────────
    messages:        List[dict]   # conversation history (sliding window)

    # ── Routing ────────────────────────────────────────────
    route:           str          # "retrieve" | "memory_only" | "tool"

    # ── RAG ────────────────────────────────────────────────
    retrieved:       str          # ChromaDB context chunks (joined string)
    sources:         List[str]    # topic names of retrieved documents

    # ── Tool ───────────────────────────────────────────────
    tool_result:     str          # output from medical calculator
    risk_score_type: str          # curb65 | wells_pe | cha2ds2 | unknown | error

    # ── Answer ─────────────────────────────────────────────
    answer:          str          # final LLM response

    # ── Quality control ────────────────────────────────────
    faithfulness:    float        # eval score 0.0–1.0
    eval_retries:    int          # safety valve counter


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Medical Risk Score Calculator (Tool)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_curb65(params: dict) -> str:
    """CURB-65 score for pneumonia severity."""
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
    """Wells Score for PE pre-test probability."""
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
    """CHA₂DS₂-VASc score for stroke risk in AF."""
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
# PART 3 — Node Functions
# ══════════════════════════════════════════════════════════════════════════════

def memory_node(state: CapstoneState) -> dict:
    """Append question to history; apply sliding window."""
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > WINDOW_SIZE:
        msgs = msgs[-WINDOW_SIZE:]
    return {"messages": msgs}


def router_node(state: CapstoneState) -> dict:
    """Classify the question as retrieve / memory_only / tool."""
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(
        f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for a Medical Diagnosis Assistant for medical students and junior doctors.

Available options:
- retrieve: search the knowledge base for medical information about diseases, symptoms, investigations, or treatments
- memory_only: answer from conversation history alone (e.g. 'what did you just say?', 'what diagnosis did you mention?')
- tool: use the medical risk score calculator (CURB-65, Wells Score, or CHA2DS2-VASc) — use this when the user provides specific patient parameters and asks for a risk score or severity score

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
    """Query ChromaDB and return top-N chunks as a joined context string."""
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=N_RESULTS)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state: CapstoneState) -> dict:
    """No-op: used on the memory_only route."""
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    """Extract patient parameters via LLM → run deterministic calculator."""
    question = state["question"]

    extraction_prompt = f"""You are a medical assistant. The user wants a medical risk score calculated.
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
        raw        = raw.strip("```json").strip("```").strip()
        data       = json.loads(raw)
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
        result     = f"Medical calculator error: {e}. Please rephrase with explicit patient parameters."

    return {"tool_result": result, "risk_score_type": score_type}


def answer_node(state: CapstoneState) -> dict:
    """Build prompt from context + history, call LLM, return answer."""
    question     = state["question"]
    retrieved    = state.get("retrieved", "")
    tool_result  = state.get("tool_result", "")
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)

    context_parts = []
    if retrieved:   context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result: context_parts.append(f"CLINICAL CALCULATOR RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    retry_note = (
        "\n\nPREVIOUS ANSWER FAILED QUALITY GATE — it contained information not "
        "present in the context. This time, every single claim must be traceable "
        "to an explicit sentence in the KNOWLEDGE BASE above. If uncertain, omit it."
    ) if eval_retries > 0 else ""

    if context:
        system_content = f"""You are a Medical Diagnosis Assistant for medical students and junior doctors.

GROUNDING RULES — follow these exactly:
1. Answer using ONLY the facts, numbers, and terminology that appear verbatim in the KNOWLEDGE BASE below.
2. If a specific value, drug dose, or criterion is NOT in the context, write: "[not in knowledge base]" rather than guessing.
3. Do NOT draw on your general medical training. Treat yourself as a reader who knows ONLY what is written below.
4. Every bullet point or sentence in your answer must be directly traceable to the context.
5. For out-of-scope questions (not medical medicine), politely decline.
{retry_note}

{context}"""
    else:
        system_content = (
            "You are a Medical Diagnosis Assistant. Answer based on the conversation history only. "
            "Remind the user to verify all decisions with a senior medician."
        )

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
    lc_msgs.append(HumanMessage(content=question))

    return {"answer": llm.invoke(lc_msgs).content}


def eval_node(state: CapstoneState) -> dict:
    """Score faithfulness; trigger retry if below threshold."""
    answer  = state.get("answer", "")
    context = state.get("retrieved", "")[:3000]
    retries = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}
    
        # Out-of-scope refusals are intentionally unfaithful to retrieved context
    refusal_signals = ["outside the scope", "not able to assist", "only provide information related to medical", "cannot help with"]
    if any(signal in answer.lower() for signal in refusal_signals):
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""You are a medical fact-checker auditing an AI-generated medical answer for hallucination.

TASK: For each factual claim in the ANSWER, decide whether it is explicitly supported by the CONTEXT.
Then return a single faithfulness score.

SCORING:
- 1.0 = every claim is directly traceable to the context
- 0.8 = 1-2 minor additions but all medical facts are grounded
- 0.6 = some medical facts added from outside the context
- 0.4 = several facts not in the context
- 0.0 = answer largely ignores or contradicts the context

CONTEXT:
{context}

ANSWER:
{answer[:600]}

Reply with ONLY a decimal number between 0.0 and 1.0. No explanation."""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5

    gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️"
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    """Append assistant answer to conversation history."""
    messages = state.get("messages", [])
    return {"messages": messages + [{"role": "assistant", "content": state["answer"]}]}


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Graph Assembly
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
print("✅ Med-Assist agent compiled and ready")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Public Helper
# ══════════════════════════════════════════════════════════════════════════════

def ask(question: str, thread_id: str = "default") -> dict:
    """
    Run the agent for one question.

    Args:
        question:  The medical question.
        thread_id: Session identifier. Reuse the same thread_id across turns
                   to preserve conversation memory.
    Returns:
        Full state dict. Key fields: answer, faithfulness, route, sources.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"question": question}, config=config)
