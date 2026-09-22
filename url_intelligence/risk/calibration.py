import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Any, Tuple

class URLRiskCalibrator:
    """
    Probability Calibration Engine.
    Employs Isotonic Regression with clipping to convert raw model probabilities
    into calibrated posterior probabilities with minimal Expected Calibration Error (ECE).
    """
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.is_fitted = False
        self.metrics = {}

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        raw_probs = np.array(raw_probs, dtype=np.float64).ravel()
        true_labels = np.array(true_labels, dtype=np.float64).ravel()
        
        self.calibrator.fit(raw_probs, true_labels)
        self.is_fitted = True
        
        calibrated_probs = self.predict(raw_probs)
        raw_ece = self.compute_ece(raw_probs, true_labels)
        cal_ece = self.compute_ece(calibrated_probs, true_labels)
        brier = float(np.mean((calibrated_probs - true_labels) ** 2))
        
        self.metrics = {
            'raw_ece': round(raw_ece, 4),
            'calibrated_ece': round(cal_ece, 4),
            'brier_score': round(brier, 4)
        }
        return self.metrics

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.clip(raw_probs, 0.0, 1.0)
        return np.clip(self.calibrator.predict(raw_probs), 0.0, 1.0)

    @staticmethod
    def compute_ece(probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for i in range(n_bins):
            lower, upper = bins[i], bins[i+1]
            mask = (probs >= lower) & (probs < upper) if i < n_bins - 1 else (probs >= lower) & (probs <= upper)
            bin_size = np.sum(mask)
            if bin_size > 0:
                bin_acc = np.mean(true_labels[mask])
                bin_conf = np.mean(probs[mask])
                ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
        return float(ece)
