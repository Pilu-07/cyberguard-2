import re
import torch
import numpy as np

class DualXAIEngine:
    """
    Dual-Source Explainable AI (XAI) Engine.
    Combines:
      1. Model Token Attribution Saliency (gradient-based token importance).
      2. Deterministic Cybersecurity Signals (programmatic rule verification).
    Produces transparent, non-memorized, dual-source explanations and security recommendations.
    """
    
    def __init__(self, top_k_tokens=5):
        self.top_k_tokens = top_k_tokens

    def compute_token_saliency(self, text, model, vectorizer, cyber_features_tensor, device=torch.device('cpu')):
        """
        Computes token-level importance attribution using model gradient saliency on text representation.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return []
            
        # Transform text into vectorizer representation with grad
        text_vec_np = vectorizer.transform([text]).toarray()
        text_tensor = torch.tensor(text_vec_np, dtype=torch.float32, device=device, requires_grad=True)
        
        model.eval()
        logits = model(text_embeddings=text_tensor, cyber_features=cyber_features_tensor)
        phishing_logit = logits[0, 1]
        
        # Compute gradient w.r.t text representation
        model.zero_grad()
        phishing_logit.backward()
        
        grad = text_tensor.grad.detach().cpu().numpy()[0]
        feature_importance = grad * text_vec_np[0]
        
        # Map vocabulary weights back to individual words in the input text
        vocab = vectorizer.vocabulary_
        word_scores = {}
        
        for w in set(words):
            if len(w) < 2:
                continue
            score = 0.0
            if w in vocab:
                idx = vocab[w]
                score += float(feature_importance[idx])
            # Check 2-grams containing w
            word_scores[w] = max(0.0, score)
            
        # Sort words by attribution score
        sorted_tokens = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        # Filter top tokens with positive attribution
        top_tokens = [{"word": w, "importance": round(float(s), 4)} for w, s in sorted_tokens[:self.top_k_tokens] if s > 0.0001]
        return top_tokens

    def generate_explanation(self, text, cyber_dict, risk_result, token_saliency=None):
        """
        Synthesizes token-level attributions with deterministic cyber rules.
        """
        rule_indicators = []
        
        # Rule-based Deterministic Cyber Signals
        if cyber_dict.get('urgency_score', 0) > 0:
            rule_indicators.append("High-pressure urgency or immediate deadline detected")
            
        if cyber_dict.get('account_threat', 0) > 0:
            rule_indicators.append("Account suspension, blockage, or unauthorized breach threat")
            
        if cyber_dict.get('credential_request', 0) > 0:
            rule_indicators.append("Credential harvesting or password verification prompt")
            
        if cyber_dict.get('otp_request', 0) > 0:
            rule_indicators.append("One-Time Password (OTP) / 2FA secret code interception request")
            
        if cyber_dict.get('payment_request', 0) > 0:
            rule_indicators.append("Financial transaction, lottery prize, or payment claim prompt")
            
        if cyber_dict.get('impersonation_flag', 0) > 0:
            rule_indicators.append("Brand or financial institution impersonation indicators")
            
        if cyber_dict.get('has_shortened_url', 0) > 0:
            rule_indicators.append("Shortened or obfuscated external URL (masks destination host)")
        elif cyber_dict.get('has_ip_url', 0) > 0:
            rule_indicators.append("Raw IP address host detected in link (bypasses domain reputation)")
        elif cyber_dict.get('has_http_only', 0) > 0:
            rule_indicators.append("Insecure HTTP link detected (lacks SSL/TLS encryption)")
        elif cyber_dict.get('has_url', 0) > 0:
            rule_indicators.append("External embedded web URL present in message")
            
        if cyber_dict.get('uppercase_ratio', 0) > 0.25:
            rule_indicators.append("Abnormal uppercase character density (social engineering pressure)")

        # Fallback if no specific rules fired
        if not rule_indicators and risk_result['severity'] == "LOW":
            rule_indicators.append("Normal language structure with no malicious threat indicators")
        elif not rule_indicators:
            rule_indicators.append("Statistical phishing patterns detected by neural semantic encoder")

        # Security Action and Recommendation
        sev = risk_result['severity']
        if sev == "CRITICAL":
            recommendation = "QUARANTINE: Do not click any links or enter credentials. Immediately report and delete message."
            explanation_summary = "Critical phishing threat: Multiple high-confidence social engineering and credential theft indicators identified."
        elif sev == "HIGH":
            recommendation = "BLOCK_LINK: Avoid clicking embedded links. Verify the sender through official channels."
            explanation_summary = "High risk phishing message: Contains suspicious links and coercion tactics characteristic of active phishing campaigns."
        elif sev == "MEDIUM":
            recommendation = "FLAG_SUSPICIOUS: Exercise caution. Verify sender authenticity before responding."
            explanation_summary = "Medium risk: Contains promotional or urgency wording that requires user caution."
        else:
            recommendation = "ALLOW: Message verified safe to process."
            explanation_summary = "Low risk: Clean message with standard communication characteristics."

        # Compile Dual XAI Results
        top_words = [t['word'] for t in (token_saliency or [])]
        
        return {
            'rule_indicators': rule_indicators,
            'token_attributions': token_saliency or [],
            'suspicious_tokens': top_words,
            'recommendation': recommendation,
            'explanation_summary': explanation_summary
        }

if __name__ == '__main__':
    from cyberguard.features import CyberFeatureExtractor
    from cyberguard.risk_engine import RiskEngine
    
    ext = CyberFeatureExtractor()
    risk_eng = RiskEngine()
    xai = DualXAIEngine()
    
    sample_text = "URGENT: Your SBI account is blocked! Verify password now at http://bit.ly/3x8q or call 88600."
    cyber_dict = ext.extract_dict(sample_text, "SMS")
    risk_res = risk_eng.calculate_risk(0.92)
    xai_res = xai.generate_explanation(sample_text, cyber_dict, risk_res, token_saliency=[
        {"word": "blocked", "importance": 0.421},
        {"word": "verify", "importance": 0.385},
        {"word": "urgent", "importance": 0.354}
    ])
    
    print("Dual XAI Engine Output:")
    print("  Rule Indicators:", xai_res['rule_indicators'])
    print("  Token Saliency:", xai_res['token_attributions'])
    print("  Recommendation:", xai_res['recommendation'])
