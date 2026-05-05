"""Rule-based fallback for retention strategy when the LLM is unavailable.

This module provides a deterministic, zero-dependency alternative to the
LLM-powered agent so the dashboard never crashes even if the AI service
is down.
"""

from typing import Dict, List


def _classify_risk(churn_probability: float) -> str:
    """Deterministic risk classification."""
    if churn_probability > 0.7:
        return "High"
    elif churn_probability > 0.3:
        return "Medium"
    return "Low"


def _identify_factors(customer_data: dict) -> List[str]:
    """Identify contributing factors using simple business rules."""
    factors: List[str] = []

    # Contract type
    contract = str(customer_data.get("Contract", "")).lower()
    if "month" in contract:
        factors.append(
            f"Month-to-month contract ('{customer_data.get('Contract')}') "
            "makes it easy for the customer to leave at any time."
        )

    # Monthly charges
    monthly = customer_data.get("MonthlyCharges")
    if monthly is not None:
        try:
            monthly = float(monthly)
            if monthly > 70:
                factors.append(
                    f"High monthly charges (${monthly:.2f}) increase price sensitivity "
                    "and churn likelihood."
                )
        except (ValueError, TypeError):
            pass

    # Tenure
    tenure = customer_data.get("tenure")
    if tenure is not None:
        try:
            tenure = int(tenure)
            if tenure < 12:
                factors.append(
                    f"Low tenure ({tenure} months) — newer customers are statistically "
                    "more likely to churn."
                )
        except (ValueError, TypeError):
            pass

    # Internet service type
    internet = str(customer_data.get("InternetService", "")).lower()
    if "fiber" in internet:
        factors.append(
            "Fiber optic internet service is associated with higher churn, "
            "possibly due to service quality or pricing concerns."
        )

    # Tech support
    tech = str(customer_data.get("TechSupport", "")).lower()
    if tech == "no":
        factors.append(
            "No tech support subscription — customers without support are more "
            "likely to leave after negative experiences."
        )

    # Online security
    security = str(customer_data.get("OnlineSecurity", "")).lower()
    if security == "no":
        factors.append(
            "No online security add-on — lack of value-added services reduces "
            "switching cost."
        )

    # Payment method
    payment = str(customer_data.get("PaymentMethod", "")).lower()
    if "electronic check" in payment:
        factors.append(
            "Electronic check payment method is correlated with higher churn "
            "compared to automatic payments."
        )

    # Paperless billing
    paperless = str(customer_data.get("PaperlessBilling", "")).lower()
    if paperless == "yes":
        factors.append(
            "Paperless billing may indicate less engagement with the provider."
        )

    # Return top 5 at most
    return factors[:5] if factors else ["No strong churn indicators detected in the available data."]


def _build_strategy(factors: List[str], risk_level: str, customer_data: dict) -> str:
    """Build a numbered retention strategy from the identified factors."""
    steps: List[str] = []

    factors_text = " ".join(factors).lower()

    if "month-to-month" in factors_text:
        steps.append(
            "Offer an incentive to upgrade to a 1-year or 2-year contract "
            "(e.g., 10-15% discount on monthly charges for the first year)."
        )

    if "monthly charges" in factors_text or "price" in factors_text:
        steps.append(
            "Review pricing against competitors and consider offering a "
            "loyalty discount or service bundle to improve perceived value."
        )

    if "tenure" in factors_text or "newer" in factors_text:
        steps.append(
            "Improve the onboarding experience with a dedicated welcome call, "
            "a quick-start guide, and a 90-day check-in to address early concerns."
        )

    if "fiber" in factors_text:
        steps.append(
            "Proactively improve fiber optic service quality and offer "
            "complimentary tech support for the first 6 months."
        )

    if "tech support" in factors_text:
        steps.append(
            "Offer a free or discounted tech support add-on to reduce "
            "frustration-driven churn."
        )

    if "security" in factors_text:
        steps.append(
            "Bundle online security with the current plan at no extra cost "
            "for the next billing cycle to increase stickiness."
        )

    if "electronic check" in factors_text:
        steps.append(
            "Encourage switching to automatic bank transfer or credit card "
            "payment with a small one-time incentive."
        )

    if risk_level == "Low":
        steps.append(
            "Continue monitoring this customer with lightweight engagement "
            "(e.g., quarterly satisfaction surveys)."
        )

    if not steps:
        steps.append("Schedule a proactive outreach call to understand the customer's needs.")
        steps.append("Consider offering a loyalty reward or personalised discount.")

    return steps[:4]


def generate_fallback_strategy(customer_data: dict, churn_probability: float) -> dict:
    """Generate a complete retention strategy using rule-based logic.

    Returns the same structure as structured report so it can be used as a
    drop-in replacement.

    Args:
        customer_data: Customer profile dictionary.
        churn_probability: ML model's predicted churn probability (0-1).

    Returns:
        dict with keys matching the RetentionReport schema, plus is_fallback.
    """
    risk_level = _classify_risk(churn_probability)
    factors = _identify_factors(customer_data)
    strategy_steps = _build_strategy(factors, risk_level, customer_data)

    sources = [
        "Risk classification based on standard probability thresholds (Low ≤ 0.3, Medium ≤ 0.7, High > 0.7).",
        "Contributing factors derived from well-known telecom churn indicators.",
        "Retention strategies based on industry best practices for customer retention."
    ]
    
    disclaimers = [
        "Disclaimer: These are rule-based suggestions generated without AI. "
        "They should be reviewed by a domain expert before implementation."
    ]

    actions = [
        {"action": step, "rationale": "Derived from rule-based fallback logic.", "priority": "Medium"}
        for step in strategy_steps
    ]

    return {
        "risk_summary": f"Customer represents a {risk_level.lower()} churn risk based on rule heuristics.",
        "risk_level": risk_level,
        "churn_probability": churn_probability,
        "contributing_factors": factors,
        "recommended_actions": actions,
        "sources": sources,
        "disclaimers": disclaimers,
        "is_fallback": True,
    }
