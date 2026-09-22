from typing import Dict, Any, Tuple
import numpy as np

class URLRiskEngine:
    """
    Risk Score Computation and Severity Tiering Engine.
    Converts calibrated malicious probability into 0-100 risk score and 4-tier severity.
    """
    def __init__(self, low_thresh: int = 25, med_thresh: int = 50, high_thresh: int = 80):
        self.low_thresh = low_thresh
        self.med_thresh = med_thresh
        self.high_thresh = high_thresh

    def compute_risk(self, calibrated_prob: float) -> Tuple[int, str]:
        """
        Maps probability [0.0, 1.0] -> Risk Score [0, 100] & Severity.
        0-24   -> LOW
        25-49  -> MEDIUM
        50-79  -> HIGH
        80-100 -> CRITICAL
        """
        score = int(round(float(calibrated_prob) * 100))
        score = max(0, min(100, score))

        if score < self.low_thresh:
            severity = "LOW"
        elif score < self.med_thresh:
            severity = "MEDIUM"
        elif score < self.high_thresh:
            severity = "HIGH"
        else:
            severity = "CRITICAL"

        return score, severity
