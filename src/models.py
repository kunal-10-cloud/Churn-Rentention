from pydantic import BaseModel
from typing import List, Dict

class RetentionReport(BaseModel):
    risk_summary: str
    risk_level: str
    churn_probability: float
    contributing_factors: List[str]
    recommended_actions: List[Dict]  # {action, rationale, priority}
    sources: List[str]
    disclaimers: List[str]
