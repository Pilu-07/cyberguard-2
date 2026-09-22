import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import torch

from url_intelligence.preprocessing import URLCanonicalizer
from url_intelligence.url_features import URLFeatureExtractor
from url_intelligence.tokenizer import CharacterURLTokenizer
from url_intelligence.model import HybridURLIntelligenceModel
from url_intelligence.risk.calibration import URLRiskCalibrator
from url_intelligence.risk.risk_engine import URLRiskEngine
from url_intelligence.risk.recommendation import URLRecommendationEngine
from url_intelligence.xai.explainer import URLExplainer
from url_intelligence.qr.decoder import QRDecoder

DEFAULT_BUNDLE_PATH = r'e:\project 2\saved_models\url_intelligence_bundle.pkl'

class URLIntelligenceEngine:
    """
    Unified URL & QR Intelligence Production Pipeline.
    Direct URL or Decoded QR -> Canonicalization -> Multi-Modal Feature Extraction ->
    Model Inference -> Calibration -> Risk Scoring (0-100) -> XAI -> Recommendation.
    """
    def __init__(self, bundle_path: str = DEFAULT_BUNDLE_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = CharacterURLTokenizer(max_length=150)
        self.feature_extractor = URLFeatureExtractor()
        self.risk_engine = URLRiskEngine()
        self.calibrator = URLRiskCalibrator()
        self.qr_decoder = QRDecoder()
        
        self.model = None
        self.scaler = None
        self.bundle_loaded = False
        
        if os.path.exists(bundle_path):
            self.load_bundle(bundle_path)
        else:
            print(f"[URLIntelligenceEngine] No pre-trained bundle at {bundle_path}. Ready for training.")

    def load_bundle(self, bundle_path: str):
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
            
        self.scaler = bundle['scaler']
        self.calibrator = bundle['calibrator']
        
        if bundle.get('model_type') == 'neural':
            cfg = bundle['model_config']
            self.model = HybridURLIntelligenceModel(
                vocab_size=cfg['vocab_size'],
                embed_dim=cfg['embed_dim'],
                char_out_dim=cfg['char_out_dim'],
                num_engineered_feats=cfg['num_engineered_feats'],
                mlp_out_dim=cfg['mlp_out_dim'],
                hidden_dim=cfg['hidden_dim'],
                num_classes=cfg['num_classes'],
                dropout=cfg['dropout']
            ).to(self.device)
            self.model.load_state_dict(bundle['model_state_dict'])
            self.model.eval()
            self.model_type = 'neural'
        else:
            self.model = bundle['model']
            self.model_type = 'tree'
            
        self.bundle_loaded = True
        print(f"[URLIntelligenceEngine] Loaded model bundle ({self.model_type}) from {bundle_path}")

    def analyze_url(self, raw_url: str, source: str = "direct_url") -> Dict[str, Any]:
        """
        End-to-end evaluation of a single URL.
        """
        canonical = URLCanonicalizer.canonicalize(raw_url)
        norm_url = canonical['normalized_url']
        domain = canonical['domain']
        root_domain = URLCanonicalizer.extract_root_domain(domain)
        
        # 1. Feature extraction
        feats_dict = self.feature_extractor.extract_features(canonical)
        feat_vector = np.array([feats_dict[name] for name in self.feature_extractor.FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
        
        # 2. Model prediction
        if self.bundle_loaded and self.model is not None:
            if self.scaler is not None:
                feat_norm = self.scaler.transform(feat_vector)
            else:
                feat_norm = feat_vector

            if self.model_type == 'neural':
                char_tokens = self.tokenizer.encode_batch([norm_url])
                char_t = torch.tensor(char_tokens, dtype=torch.long, device=self.device)
                feat_t = torch.tensor(feat_norm, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    logits = self.model(char_t, feat_t)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    raw_p_malicious = float(probs[1])
            else: # Tree model (XGBoost / LightGBM)
                probs = self.model.predict_proba(feat_norm)[0]
                raw_p_malicious = float(probs[1])
        else:
            # Deterministic heuristic fallback if bundle not yet trained
            heuristic_score = 0.05
            if feats_dict['has_ip_host'] == 1.0: heuristic_score += 0.45
            if feats_dict['has_suspicious_tld'] == 1.0: heuristic_score += 0.25
            if feats_dict['has_shortener'] == 1.0: heuristic_score += 0.20
            if feats_dict['has_login_kw'] == 1.0: heuristic_score += 0.20
            if feats_dict['has_brand_kw'] == 1.0: heuristic_score += 0.30
            raw_p_malicious = min(0.99, heuristic_score)

        # 3. Probability Calibration
        calibrated_p = float(self.calibrator.predict(np.array([raw_p_malicious]))[0])

        # 4. Risk Engine
        risk_score, severity = self.risk_engine.compute_risk(calibrated_p)

        # 5. XAI Explanation
        xai_res = URLExplainer.explain(raw_url, risk_score, severity, feats_dict)

        # 6. Recommendation Engine
        rec = URLRecommendationEngine.get_recommendation(severity, risk_score)

        return {
            'original_url': canonical['original_url'],
            'normalized_url': norm_url,
            'domain': domain,
            'root_domain': root_domain,
            'source': source,
            'raw_p_malicious': round(raw_p_malicious, 4),
            'calibrated_p_malicious': round(calibrated_p, 4),
            'risk_score': risk_score,
            'severity': severity,
            'action': rec['action'],
            'guidance': rec['guidance'],
            'indicators': xai_res['indicators'],
            'suspicious_tokens': xai_res['suspicious_tokens'],
            'explanation': xai_res['explanation_summary']
        }

    def analyze_qr_image(self, image_input) -> Dict[str, Any]:
        """
        Decodes QR image then passes to the exact same URL Intelligence engine.
        """
        decode_res = self.qr_decoder.decode_image(image_input)
        if not decode_res['success']:
            return {
                'success': False,
                'error': decode_res['error'],
                'risk_score': 0,
                'severity': 'UNKNOWN',
                'action': 'ERROR'
            }

        url_content = decode_res['raw_content']
        analysis = self.analyze_url(url_content, source="qr_decoded")
        analysis['success'] = True
        analysis['decoded_text'] = url_content
        return analysis
