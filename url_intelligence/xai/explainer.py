from typing import Dict, Any, List
import re
from url_intelligence.preprocessing import URLCanonicalizer
from url_intelligence.url_features import URLFeatureExtractor

class URLExplainer:
    """
    Explainable AI (XAI) Engine for URL Intelligence.
    Combines deterministic security indicator auditing with character/structural token highlights.
    Provides human-interpretable rationale for why a URL was flagged.
    """
    @classmethod
    def explain(cls, raw_url: str, risk_score: int, severity: str, feats: Dict[str, float]) -> Dict[str, Any]:
        info = URLCanonicalizer.canonicalize(raw_url)
        norm_url = info['normalized_url']
        domain = info['domain']
        path = info['path']
        
        indicators = []
        token_highlights = []

        # 1. Deterministic Security Alerts
        if feats.get('has_ip_host', 0.0) == 1.0:
            indicators.append({
                'level': 'CRITICAL',
                'title': 'Direct IP Address Host',
                'description': f'Host is a raw numeric IP ({domain}) rather than an authenticated registered domain name.'
            })
            token_highlights.append(domain)

        if feats.get('has_shortener', 0.0) == 1.0:
            indicators.append({
                'level': 'HIGH',
                'title': 'URL Shortener Redirection Service',
                'description': 'URL uses a shortener service often deployed to obscure malicious destination payloads.'
            })

        if feats.get('has_suspicious_tld', 0.0) == 1.0:
            tld = domain.split('.')[-1]
            indicators.append({
                'level': 'HIGH',
                'title': f'High-Risk TLD (.{tld})',
                'description': f'The domain uses top-level domain .{tld} frequently associated with low-reputation disposable phishing infrastructure.'
            })
            token_highlights.append(f'.{tld}')

        if feats.get('has_punycode', 0.0) == 1.0:
            indicators.append({
                'level': 'HIGH',
                'title': 'Punycode / Homograph Attack Structure',
                'description': 'URL contains xn-- prefix indicative of character spoofing or internationalized domain homoglyph deception.'
            })

        if feats.get('subdomain_count', 0.0) >= 3:
            indicators.append({
                'level': 'MEDIUM',
                'title': 'Deep Subdomain Hierarchy',
                'description': f'Domain has excessive subdomain nesting ({int(feats.get("subdomain_count", 0))} levels) common in deceptive hostname tunneling.'
            })

        if feats.get('url_entropy', 0.0) > 4.5:
            indicators.append({
                'level': 'MEDIUM',
                'title': 'High Character Entropy / Random String',
                'description': f'Unusually high randomness (Entropy: {feats.get("url_entropy", 0.0):.2f}) indicating DGA or obfuscated payload slugs.'
            })

        if feats.get('is_https', 1.0) == 0.0:
            indicators.append({
                'level': 'MEDIUM',
                'title': 'Unencrypted Protocol (HTTP)',
                'description': 'URL transmits credentials over plaintext unencrypted HTTP.'
            })

        if feats.get('consecutive_hyphens', 0.0) == 1.0:
            indicators.append({
                'level': 'MEDIUM',
                'title': 'Hyphenated Brand Squatting Pattern',
                'description': 'Repeated consecutive hyphens detected in domain or path, typical of typosquatting.'
            })

        # Keyword checks
        for kw in URLFeatureExtractor.PHISHING_KEYWORDS:
            if kw in norm_url.lower():
                token_highlights.append(kw)

        if feats.get('has_login_kw', 0.0) == 1.0:
            indicators.append({
                'level': 'MEDIUM',
                'title': 'Authentication Credential Trap Detected',
                'description': 'Presence of login/signin/auth keywords in non-standard domain context.'
            })

        if feats.get('has_brand_kw', 0.0) == 1.0:
            indicators.append({
                'level': 'HIGH',
                'title': 'Targeted Brand Mimicry Signal',
                'description': 'Prominent enterprise/banking brand name detected within URL structure.'
            })

        if not indicators and severity == "LOW":
            indicators.append({
                'level': 'INFO',
                'title': 'Clean Lexical & Infrastructure Profile',
                'description': 'Standard domain hierarchy, valid HTTPS scheme, and absence of deceptive obfuscation patterns.'
            })

        # Summary text
        if severity in ("HIGH", "CRITICAL"):
            summary = f"Flagged as {severity} Risk ({risk_score}/100) due to {len(indicators)} structural anomalies and phishing indicators."
        elif severity == "MEDIUM":
            summary = f"Moderate suspicion ({risk_score}/100). Exercise caution before interacting with this destination."
        else:
            summary = f"Clean URL ({risk_score}/100). Characteristics align with normal benign web traffic."

        return {
            'risk_score': risk_score,
            'severity': severity,
            'indicators': indicators,
            'suspicious_tokens': list(set(token_highlights))[:8],
            'explanation_summary': summary
        }
