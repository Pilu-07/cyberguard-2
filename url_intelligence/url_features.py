import re
import math
import numpy as np
from typing import Dict, Any, List
import urllib.parse
from url_intelligence.preprocessing import URLCanonicalizer

class URLFeatureExtractor:
    """
    Deterministic 42-Dimensional Security, Lexical, and Structural Feature Extractor
    for Phishing & Malicious URL Classification.
    """

    SUSPICIOUS_TLDS = {
        'xyz', 'top', 'club', 'work', 'site', 'loan', 'click', 'cfd',
        'buzz', 'gq', 'ml', 'cf', 'tk', 'ga', 'fit', 'kim', 'country',
        'science', 'party', 'stream', 'trade', 'racing', 'mom', 'date',
        'zip', 'mov', 'cam', 'rest', 'link', 'tokyo', 'men', 'win', 'bid'
    }

    PHISHING_KEYWORDS = [
        'login', 'verify', 'verification', 'secure', 'account', 'update',
        'password', 'signin', 'bank', 'wallet', 'confirm', 'unlock',
        'auth', 'kyc', 'support', 'recover', 'billing', 'validation',
        'service', 'free', 'bonus', 'claim', 'security', 'suspended',
        'appleid', 'paypal', 'netflix', 'amazon', 'microsoft', 'chase'
    ]

    SHORTENER_DOMAINS = {
        'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'is.gd', 'buff.ly',
        'ow.ly', 'tiny.cc', 'rb.gy', 'cutt.ly', '000webhostapp.com',
        'linktr.ee', 's.id', 'shorturl.at', 'bl.ink', 'qr.ae', 'v.gd',
        'tr.im', 'rebrand.ly', 'bit.do', 'shorte.st'
    }

    FEATURE_NAMES = [
        # Lexical (10)
        'url_length', 'hostname_length', 'path_length', 'query_length', 'fragment_length',
        'digit_count', 'letter_count', 'special_char_count', 'letter_ratio', 'digit_ratio',
        # Structural (10)
        'subdomain_count', 'dot_count', 'slash_count', 'hyphen_count', 'underscore_count',
        'equal_count', 'ampersand_count', 'percent_count', 'question_mark_count', 'at_symbol_count',
        # Entropy (2)
        'url_entropy', 'hostname_entropy',
        # Security & Infrastructure Patterns (10)
        'is_https', 'has_ip_host', 'has_non_standard_port', 'has_punycode', 'has_double_slash_in_path',
        'has_shortener', 'has_suspicious_tld', 'consecutive_hyphens', 'hex_encoded_char_count', 'tld_length',
        # Semantic Phishing Signals (10)
        'keyword_count', 'has_login_kw', 'has_verify_kw', 'has_account_kw', 'has_security_kw',
        'has_payment_kw', 'has_brand_kw', 'keyword_in_subdomain', 'keyword_in_path', 'keyword_density'
    ]

    @staticmethod
    def shannon_entropy(s: str) -> float:
        """Calculates Shannon entropy of string to detect random/DGA strings."""
        if not s:
            return 0.0
        prob = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    @classmethod
    def extract_features(cls, url_or_canonical: Any) -> Dict[str, float]:
        """
        Extracts 42 numerical features from URL string or canonicalized dict.
        """
        if isinstance(url_or_canonical, str):
            info = URLCanonicalizer.canonicalize(url_or_canonical)
        elif isinstance(url_or_canonical, dict):
            info = url_or_canonical
        else:
            info = URLCanonicalizer.canonicalize("")

        orig_url = info.get('original_url', '')
        norm_url = info.get('normalized_url', '')
        scheme = info.get('scheme', 'http')
        domain = info.get('domain', '')
        path = info.get('path', '')
        query = info.get('query', '')
        fragment = info.get('fragment', '')

        # 1. Lexical features
        url_len = len(norm_url)
        host_len = len(domain)
        path_len = len(path)
        query_len = len(query)
        frag_len = len(fragment)

        digits = sum(c.isdigit() for c in norm_url)
        letters = sum(c.isalpha() for c in norm_url)
        special = sum(not c.isalnum() for c in norm_url)
        letter_ratio = (letters / url_len) if url_len > 0 else 0.0
        digit_ratio = (digits / url_len) if url_len > 0 else 0.0

        # 2. Structural features
        dot_count = norm_url.count('.')
        slash_count = norm_url.count('/')
        hyphen_count = norm_url.count('-')
        underscore_count = norm_url.count('_')
        equal_count = norm_url.count('=')
        ampersand_count = norm_url.count('&')
        percent_count = norm_url.count('%')
        question_count = norm_url.count('?')
        at_count = norm_url.count('@')

        # Subdomain count
        root_domain = URLCanonicalizer.extract_root_domain(domain)
        if root_domain and root_domain != domain:
            subdomain_part = domain[:-len(root_domain)].rstrip('.')
            subdomain_count = len(subdomain_part.split('.')) if subdomain_part else 0
        else:
            subdomain_count = 0

        # 3. Entropy
        url_entropy = cls.shannon_entropy(norm_url)
        host_entropy = cls.shannon_entropy(domain)

        # 4. Security Patterns
        is_https = 1.0 if scheme == 'https' else 0.0
        is_ip = 1.0 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0.0
        
        # Check port
        has_non_standard_port = 0.0
        if ':' in domain:
            port_str = domain.split(':')[-1]
            if port_str.isdigit() and int(port_str) not in (80, 443):
                has_non_standard_port = 1.0

        has_punycode = 1.0 if 'xn--' in norm_url.lower() else 0.0
        has_double_slash_in_path = 1.0 if '//' in path else 0.0
        has_shortener = 1.0 if root_domain.lower() in cls.SHORTENER_DOMAINS else 0.0

        tld = domain.split('.')[-1].lower() if '.' in domain else ''
        tld_length = float(len(tld))
        has_suspicious_tld = 1.0 if tld in cls.SUSPICIOUS_TLDS else 0.0
        consecutive_hyphens = 1.0 if '--' in norm_url else 0.0
        hex_enc_count = float(len(re.findall(r'%[0-9a-fA-F]{2}', orig_url)))

        # 5. Semantic / Keyword Features
        norm_url_lower = norm_url.lower()
        path_lower = path.lower()
        subdomain_part = domain[:-len(root_domain)].lower() if root_domain else ''

        kw_matches = [kw for kw in cls.PHISHING_KEYWORDS if kw in norm_url_lower]
        keyword_count = float(len(kw_matches))
        has_login_kw = 1.0 if any(k in norm_url_lower for k in ['login', 'signin', 'auth']) else 0.0
        has_verify_kw = 1.0 if any(k in norm_url_lower for k in ['verify', 'verification', 'validation', 'confirm', 'kyc']) else 0.0
        has_account_kw = 1.0 if any(k in norm_url_lower for k in ['account', 'update', 'billing', 'recover']) else 0.0
        has_security_kw = 1.0 if any(k in norm_url_lower for k in ['secure', 'security', 'suspended', 'unlock']) else 0.0
        has_payment_kw = 1.0 if any(k in norm_url_lower for k in ['bank', 'wallet', 'claim', 'bonus', 'free']) else 0.0
        has_brand_kw = 1.0 if any(b in norm_url_lower for b in ['appleid', 'paypal', 'netflix', 'amazon', 'microsoft', 'chase']) else 0.0

        kw_in_subdomain = 1.0 if any(kw in subdomain_part for kw in cls.PHISHING_KEYWORDS) else 0.0
        kw_in_path = 1.0 if any(kw in path_lower for kw in cls.PHISHING_KEYWORDS) else 0.0
        keyword_density = (keyword_count / (url_len / 10.0)) if url_len > 0 else 0.0

        return {
            'url_length': float(url_len),
            'hostname_length': float(host_len),
            'path_length': float(path_len),
            'query_length': float(query_len),
            'fragment_length': float(frag_len),
            'digit_count': float(digits),
            'letter_count': float(letters),
            'special_char_count': float(special),
            'letter_ratio': float(letter_ratio),
            'digit_ratio': float(digit_ratio),
            'subdomain_count': float(subdomain_count),
            'dot_count': float(dot_count),
            'slash_count': float(slash_count),
            'hyphen_count': float(hyphen_count),
            'underscore_count': float(underscore_count),
            'equal_count': float(equal_count),
            'ampersand_count': float(ampersand_count),
            'percent_count': float(percent_count),
            'question_mark_count': float(question_count),
            'at_symbol_count': float(at_count),
            'url_entropy': float(url_entropy),
            'hostname_entropy': float(host_entropy),
            'is_https': is_https,
            'has_ip_host': is_ip,
            'has_non_standard_port': has_non_standard_port,
            'has_punycode': has_punycode,
            'has_double_slash_in_path': has_double_slash_in_path,
            'has_shortener': has_shortener,
            'has_suspicious_tld': has_suspicious_tld,
            'consecutive_hyphens': consecutive_hyphens,
            'hex_encoded_char_count': hex_enc_count,
            'tld_length': tld_length,
            'keyword_count': keyword_count,
            'has_login_kw': has_login_kw,
            'has_verify_kw': has_verify_kw,
            'has_account_kw': has_account_kw,
            'has_security_kw': has_security_kw,
            'has_payment_kw': has_payment_kw,
            'has_brand_kw': has_brand_kw,
            'keyword_in_subdomain': kw_in_subdomain,
            'keyword_in_path': kw_in_path,
            'keyword_density': float(keyword_density)
        }

    @classmethod
    def extract_vector(cls, url: str) -> np.ndarray:
        feats = cls.extract_features(url)
        return np.array([feats[name] for name in cls.FEATURE_NAMES], dtype=np.float32)
