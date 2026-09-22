import os
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression

class RiskEngine:
    """
    Calibrated Risk Engine & Severity Classifier.
    Transforms raw model output probabilities into calibrated 0-100 risk scores,
    calculates Expected Calibration Error (ECE), and assigns 4-tier severity levels & security actions.
    """
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.is_fitted = False
        self.calibration_metrics = {}

    def fit_calibration(self, raw_probs, true_labels):
        """
        Fits Isotonic Regression calibration on validation set probabilities.
        """
        raw_probs = np.array(raw_probs, dtype=np.float64).ravel()
        true_labels = np.array(true_labels, dtype=np.float64).ravel()
        
        # Sort and fit
        self.calibrator.fit(raw_probs, true_labels)
        self.is_fitted = True
        
        # Compute Expected Calibration Error (ECE) before and after
        ece_raw = self.compute_ece(raw_probs, true_labels, n_bins=10)
        calibrated_probs = self.calibrator.predict(raw_probs)
        ece_calibrated = self.compute_ece(calibrated_probs, true_labels, n_bins=10)
        
        self.calibration_metrics = {
            'ece_raw': round(float(ece_raw), 4),
            'ece_calibrated': round(float(ece_calibrated), 4),
            'brier_score': round(float(np.mean((calibrated_probs - true_labels) ** 2)), 4)
        }
        
        print(f"Isotonic Risk Calibration fitted successfully.")
        print(f"  Raw ECE: {self.calibration_metrics['ece_raw']*100:.2f}% -> Calibrated ECE: {self.calibration_metrics['ece_calibrated']*100:.2f}%")
        print(f"  Calibrated Brier Score: {self.calibration_metrics['brier_score']:.4f}")
        return self.calibration_metrics

    @staticmethod
    def compute_ece(probs, true_labels, n_bins=10):
        """
        Calculates Expected Calibration Error across equal-width confidence bins.
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        
        for i in range(n_bins):
            bin_lower, bin_upper = bins[i], bins[i+1]
            mask = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
            bin_size = np.sum(mask)
            if bin_size > 0:
                bin_acc = np.mean(true_labels[mask])
                bin_conf = np.mean(probs[mask])
                ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
        return ece

    def calculate_risk(self, raw_prob):
        """
        Converts a raw model probability into a calibrated risk score (0-100) and 4-tier severity band.
        """
        raw_prob = float(raw_prob)
        raw_prob = max(0.0, min(1.0, raw_prob))
        
        if self.is_fitted:
            calibrated_prob = float(self.calibrator.predict([raw_prob])[0])
            calibrated_prob = max(0.0, min(1.0, calibrated_prob))
        else:
            calibrated_prob = raw_prob
            
        risk_score = int(round(calibrated_prob * 100))
        risk_score = max(0, min(100, risk_score))
        
        # 4-Tier Severity & Security Action Mapping
        if risk_score <= 24:
            severity = "LOW"
            action = "ALLOW"
            status = "LEGITIMATE"
            color = "#10B981" # Green
        elif risk_score <= 49:
            severity = "MEDIUM"
            action = "FLAG_SUSPICIOUS"
            status = "SUSPICIOUS"
            color = "#F59E0B" # Amber
        elif risk_score <= 79:
            severity = "HIGH"
            action = "BLOCK_LINK"
            status = "PHISHING"
            color = "#F97316" # Orange
        else:
            severity = "CRITICAL"
            action = "QUARANTINE"
            status = "PHISHING"
            color = "#EF4444" # Red
            
        return {
            'raw_prob': round(raw_prob, 4),
            'calibrated_prob': round(calibrated_prob, 4),
            'risk_score': risk_score,
            'severity': severity,
            'action': action,
            'status': status,
            'color': color
        }

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'calibrator': self.calibrator,
                'is_fitted': self.is_fitted,
                'calibration_metrics': self.calibration_metrics
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.calibrator = data['calibrator']
            self.is_fitted = data['is_fitted']
            self.calibration_metrics = data.get('calibration_metrics', {})

if __name__ == '__main__':
    engine = RiskEngine()
    dummy_probs = np.random.uniform(0, 1, 500)
    dummy_labels = (dummy_probs > 0.45).astype(int)
    engine.fit_calibration(dummy_probs, dummy_labels)
    
    test_probs = [0.05, 0.35, 0.68, 0.94]
    print("\nRisk Engine Test Results:")
    for p in test_probs:
        res = engine.calculate_risk(p)
        print(f" Raw Prob: {p:.2f} -> Calibrated: {res['calibrated_prob']:.2f} | Risk Score: {res['risk_score']:3d}/100 | Severity: {res['severity']:8s} | Action: {res['action']}")
