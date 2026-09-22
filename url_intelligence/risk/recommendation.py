from typing import Dict, Any

class URLRecommendationEngine:
    """
    Decoupled Action Recommendation Engine.
    Maps risk levels and evidence severity to actionable policy decisions:
    LOW      -> ALLOW
    MEDIUM   -> WARN
    HIGH     -> WARN / QUARANTINE
    CRITICAL -> BLOCK
    """
    @staticmethod
    def get_recommendation(severity: str, risk_score: int) -> Dict[str, str]:
        if severity == "LOW":
            action = "ALLOW"
            instruction = "URL appears benign. Safe to proceed with normal operational monitoring."
        elif severity == "MEDIUM":
            action = "WARN"
            instruction = "Suspicious structural or keyword anomalies detected. Advise user caution before submitting data."
        elif severity == "HIGH":
            action = "QUARANTINE"
            instruction = "High probability of malicious intent. Isolate or block direct click-through pending manual review."
        else: # CRITICAL
            action = "BLOCK"
            instruction = "Active credential harvesting or phishing characteristics detected. Immediate automated block recommended."

        return {
            'action': action,
            'guidance': instruction
        }
