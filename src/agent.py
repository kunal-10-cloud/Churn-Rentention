"""Agentic workflow for churn retention strategy planning using LangGraph.

Builds a four-node StateGraph that takes customer data and a churn
probability, then produces a risk level, contributing factors, a
retention strategy, and supporting reasoning/disclaimers.

Nodes (executed in order):
    analyze_risk → identify_factors → generate_strategy → add_disclaimers
"""

import json
from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END

from src.llm_client import get_llm_response
from src.fallback import generate_fallback_strategy

import logging

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ChurnAgentState(TypedDict, total=False):
    customer_data: Dict
    churn_probability: float
    shap_explanation: List[Dict]
    risk_level: str
    contributing_factors: List[str]
    retention_strategy: str
    sources: List[str]
    structured_report: dict


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def analyze_risk(state: ChurnAgentState) -> dict:
    """Classify churn probability into Low / Medium / High risk using the LLM."""

    prob = state["churn_probability"]

    prompt = (
        f"A customer has a churn probability of {prob:.2%}.\n\n"
        "Classify this into exactly one of the following risk levels:\n"
        "- Low (probability ≤ 0.3)\n"
        "- Medium (probability between 0.3 and 0.7)\n"
        "- High (probability > 0.7)\n\n"
        "Respond with ONLY the single word: Low, Medium, or High."
    )

    system = (
        "You are a risk-classification assistant. "
        "Return exactly one word — Low, Medium, or High — with no extra text."
    )

    raw = get_llm_response(prompt, system_prompt=system).strip().rstrip(".")

    # Normalise the LLM response to one of the three canonical labels
    normalised = raw.capitalize()
    if normalised not in {"Low", "Medium", "High"}:
        # Fallback: deterministic rule if the LLM drifts
        if prob > 0.7:
            normalised = "High"
        elif prob > 0.3:
            normalised = "Medium"
        else:
            normalised = "Low"

    return {"risk_level": normalised}


def identify_factors(state: ChurnAgentState) -> dict:
    """Identify the top contributing factors for potential churn."""

    customer_json = json.dumps(state["customer_data"], indent=2, default=str)

    shap_block = ""
    shap_explanation = state.get("shap_explanation") or []
    if shap_explanation:
        from src.explainability import format_shap_for_prompt
        shap_block = (
            "\nSHAP PER-CUSTOMER ATTRIBUTION (top features driving this "
            "customer's prediction):\n"
            f"{format_shap_for_prompt(shap_explanation)}\n"
            "Prefer factors that appear in the SHAP list, since they are "
            "computed from this specific customer's prediction.\n"
        )

    prompt = (
        f"CUSTOMER PROFILE:\n{customer_json}\n\n"
        f"RISK LEVEL: {state['risk_level']}\n"
        f"{shap_block}\n"
        "Based on the information above, identify the top 3-5 factors "
        "that most likely contribute to this customer's churn risk.\n\n"
        "Rules:\n"
        "- Each factor must reference a specific field and value from the profile.\n"
        "- If SHAP attributions are provided, cite the SHAP feature name and sign.\n"
        "- Explain in one sentence why that field/value raises or lowers churn risk.\n"
        "- Do not invent data that is not in the profile.\n\n"
        "Return ONLY a JSON array of strings. Example:\n"
        '["Month-to-month contract increases flexibility to leave", '
        '"High monthly charges of $95 exceed segment average"]'
    )

    system = (
        "You are a customer-churn analyst. "
        "Return a JSON array of concise factor strings — nothing else."
    )

    raw = get_llm_response(prompt, system_prompt=system)

    # Parse the JSON array from the LLM response
    try:
        factors = json.loads(raw)
        if not isinstance(factors, list):
            raise ValueError
        factors = [str(f) for f in factors]
    except (json.JSONDecodeError, ValueError):
        # Best-effort: split by newlines and clean up
        factors = [
            line.lstrip("- •0123456789.").strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    return {"contributing_factors": factors}


def generate_strategy(state: ChurnAgentState) -> dict:
    """Generate an actionable retention strategy based on all preceding analysis."""

    customer_json = json.dumps(state["customer_data"], indent=2, default=str)
    factors_text = "\n".join(f"- {f}" for f in state["contributing_factors"])

    # RAG Retrieval
    try:
        from src.rag import retrieve
        query_text = (
            f"Risk Level: {state['risk_level']}\n"
            f"Factors: {factors_text}\n"
            f"Profile: {customer_json}"
        )
        contexts = retrieve(query_text)
        context_str = "\n---\n".join(contexts) if contexts else "No relevant knowledge base strategies found."
    except Exception as e:
        _logger.warning(f"RAG retrieval failed: {e}")
        context_str = "RAG unavailable."

    prompt = (
        f"CUSTOMER PROFILE:\n{customer_json}\n\n"
        f"RISK LEVEL: {state['risk_level']}\n\n"
        f"CONTRIBUTING FACTORS:\n{factors_text}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        "Based on the information above, propose a concrete retention strategy "
        "consisting of 2-4 actionable steps the business can take to retain this customer.\n\n"
        "Rules:\n"
        "- Each step must directly address one of the contributing factors.\n"
        "- Prefer low-cost, measurable, and ethical actions.\n"
        "- Be specific (e.g., 'offer a 12-month contract at a 10 % discount') rather than vague.\n"
        "- If the risk level is Low, focus on monitoring and lightweight engagement.\n"
        "- Incorporate insights from the CONTEXT if relevant and applicable.\n\n"
        "Return ONLY valid JSON matching this exact structure:\n"
        "{\n"
        '  "risk_summary": "...",\n'
        '  "risk_level": "...",\n'
        '  "churn_probability": ' + str(state['churn_probability']) + ',\n'
        '  "contributing_factors": [...],\n'
        '  "recommended_actions": [\n'
        '    {"action": "...", "rationale": "...", "priority": "High/Medium/Low"}\n'
        '  ],\n'
        '  "sources": [...],\n'
        '  "disclaimers": [...]\n'
        "}\n\n"
        "Do NOT return any text outside of the JSON."
    )

    system = (
        "You are a customer retention strategist for a telecom company. "
        "Be concise, practical, and ground every recommendation in the data provided. "
        "You must output only valid, parseable JSON."
    )

    raw_response = get_llm_response(prompt, system_prompt=system)

    from src.models import RetentionReport
    try:
        report = RetentionReport.model_validate_json(raw_response)
        return {"structured_report": report.model_dump()}
    except Exception as e:
        _logger.warning(f"Failed to parse main response, retrying: {e}")
        # Retry once
        retry_prompt = f"Previous JSON parsing failed. Return valid JSON only:\n\n{prompt}"
        raw_retry = get_llm_response(retry_prompt, system_prompt=system)
        report = RetentionReport.model_validate_json(raw_retry)
        return {"structured_report": report.model_dump()}


def add_disclaimers(state: ChurnAgentState) -> dict:
    """Provide reasoning sources and disclaimers behind the decisions."""

    prompt = (
        f"RISK LEVEL: {state['risk_level']}\n\n"
        f"CONTRIBUTING FACTORS:\n"
        + "\n".join(f"- {f}" for f in state["contributing_factors"])
        + f"\n\nRETENTION STRATEGY:\n{state['retention_strategy']}\n\n"
        "Provide 3-5 brief supporting reasons or references that justify the "
        "retention strategy above. Include a final disclaimer noting that these "
        "are AI-generated suggestions and should be reviewed by a human.\n\n"
        "Return ONLY a JSON array of strings. Example:\n"
        '["Reason 1", "Reason 2", "Disclaimer: ..."]'
    )

    system = (
        "You are a compliance-aware AI assistant. "
        "Return a JSON array of concise reasoning strings — nothing else."
    )

    raw = get_llm_response(prompt, system_prompt=system)

    try:
        sources = json.loads(raw)
        if not isinstance(sources, list):
            raise ValueError
        sources = [str(s) for s in sources]
    except (json.JSONDecodeError, ValueError):
        sources = [
            line.lstrip("- •0123456789.").strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    # Always ensure a disclaimer is present
    has_disclaimer = any("disclaimer" in s.lower() for s in sources)
    if not has_disclaimer:
        sources.append(
            "Disclaimer: These suggestions are AI-generated and must be "
            "reviewed by a human before being acted upon."
        )

    return {"sources": sources}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph workflow."""

    graph = StateGraph(ChurnAgentState)

    # Add nodes
    graph.add_node("analyze_risk", analyze_risk)
    graph.add_node("identify_factors", identify_factors)
    graph.add_node("generate_strategy", generate_strategy)

    # Set entry point
    graph.set_entry_point("analyze_risk")

    # Connect nodes in sequence
    graph.add_edge("analyze_risk", "identify_factors")
    graph.add_edge("identify_factors", "generate_strategy")
    graph.add_edge("generate_strategy", END)

    return graph.compile()


# Pre-compile the graph once at module level
_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(
    customer_data: dict,
    churn_probability: float,
    shap_explanation: Optional[List[Dict]] = None,
) -> dict:
    """Run the churn-retention agent and return structured output.

    If the LLM is unavailable (after retries), the function falls back
    to a deterministic rule-based strategy so the caller always receives
    a valid result.

    Args:
        customer_data: Dictionary of customer attributes (e.g. tenure,
            MonthlyCharges, Contract, etc.).
        churn_probability: Float between 0 and 1 representing the ML
            model's predicted churn probability.

    Returns:
        Dictionary with keys:
            - risk_level (str)
            - contributing_factors (list[str])
            - retention_strategy (str)
            - sources (list[str])
            - is_fallback (bool)
    """

    initial_state: ChurnAgentState = {
        "customer_data": customer_data,
        "churn_probability": churn_probability,
        "shap_explanation": shap_explanation or [],
        "risk_level": "",
        "contributing_factors": [],
        "retention_strategy": "",
        "sources": [],
    }

    try:
        final_state = _compiled_graph.invoke(initial_state)

        if "structured_report" in final_state:
            report_dict = final_state["structured_report"]
            report_dict["is_fallback"] = False
            return report_dict
        else:
            raise ValueError("No structured report found")
    except Exception as exc:
        _logger.warning("LLM agent failed, using rule-based fallback: %s", exc)
        return generate_fallback_strategy(customer_data, churn_probability)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
    }

    result = run_agent(sample_customer, churn_probability=0.82)

    print("=" * 60)
    print(f"Risk Level        : {result.get('risk_level')}")
    print(f"Summary           : {result.get('risk_summary')}")
    print(f"Contributing Factors:")
    for f in result.get("contributing_factors", []):
        print(f"  • {f}")
    print(f"\nRetention Strategy:")
    for a in result.get("recommended_actions", []):
        print(f"  • [{a.get('priority', 'N/A')}] {a.get('action')}: {a.get('rationale')}")
    print(f"\nSources / Reasoning:")
    for s in result.get("sources", []):
        print(f"  • {s}")
    print("=" * 60)
