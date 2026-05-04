"""Prompt templates for churn risk analysis and retention strategy generation.

Each template is intentionally strict about:
- structured output (markdown sections or JSON),
- grounding every claim in the provided customer data,
- flagging unknowns instead of inventing facts.
"""

RISK_ANALYSIS_SYSTEM = """You are a senior customer retention analyst for a telecom company.
Your job is to read a single customer's profile and churn probability and explain, in plain language,
what is driving the risk. Ground every statement in the fields provided. Do not invent data.
If a field is missing, say 'not available' rather than guessing."""

RISK_ANALYSIS_USER = """Analyze the following customer and their predicted churn probability.

CUSTOMER PROFILE:
{customer_profile}

CHURN PROBABILITY: {churn_probability}
RISK LEVEL: {risk_level}

Return the response in the following markdown format:

### Summary
One or two sentences describing the customer and the overall risk.

### Top Contributing Factors
A bulleted list of up to 4 factors from the profile that plausibly raise or lower churn risk.
For each factor, quote the field name and value, then explain the link in one sentence.

### Unknowns / Low-Confidence Areas
Bullet list of fields that are missing, noisy, or insufficient to draw a strong conclusion.

Do not speculate beyond the provided fields. Do not recommend actions here — only analyze."""


STRATEGY_SYSTEM = """You are a retention strategist. Propose concrete retention actions
grounded strictly in the provided risk analysis. Each action must reference the factor it addresses.
Do not propose actions that rely on information not present in the analysis.
Prefer actions that are low-cost, measurable, and ethical."""

STRATEGY_USER = """Given the risk analysis below, propose retention actions.

RISK ANALYSIS:
{risk_analysis}

CUSTOMER RISK LEVEL: {risk_level}

Return the response as valid JSON with this exact shape:

{{
  "actions": [
    {{
      "title": "short action name",
      "description": "1-2 sentence description of what to do",
      "addresses_factor": "the specific factor from the risk analysis this targets",
      "reasoning": "why this action should help, referencing the factor",
      "effort": "low | medium | high",
      "expected_impact": "low | medium | high"
    }}
  ],
  "notes": "optional caveats, or empty string"
}}

Rules:
- Return between 2 and 4 actions.
- Every action must cite a factor that appears in the risk analysis.
- If the risk level is Low, bias toward lightweight / monitoring actions.
- Do not mention competitors by name.
- Do not promise specific discounts or legal terms."""


def build_risk_analysis_prompt(customer_profile: str, churn_probability: float, risk_level: str):
    return RISK_ANALYSIS_SYSTEM, RISK_ANALYSIS_USER.format(
        customer_profile=customer_profile,
        churn_probability=f"{churn_probability:.2%}",
        risk_level=risk_level,
    )


def build_strategy_prompt(risk_analysis: str, risk_level: str):
    return STRATEGY_SYSTEM, STRATEGY_USER.format(
        risk_analysis=risk_analysis,
        risk_level=risk_level,
    )


DISCLAIMER_SYSTEM = """You append short, neutral disclaimers to AI-generated retention recommendations."""

DISCLAIMER_USER = """Append a disclaimer block to the following retention recommendations.

RECOMMENDATIONS:
{recommendations}

Return the original recommendations unchanged, followed by a markdown section:

---
### Disclaimer
- These suggestions are model-generated and must be reviewed by a human before being acted on.
- They are based only on the provided customer fields and may not reflect the full context.
- Any pricing, contract, or legal action requires approval from the relevant team.
- Do not use these outputs to make automated decisions that materially affect a customer."""


def build_disclaimer_prompt(recommendations: str):
    return DISCLAIMER_SYSTEM, DISCLAIMER_USER.format(recommendations=recommendations)
