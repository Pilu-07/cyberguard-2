import re
import math
import numpy as np
import pandas as pd

class CyberFeatureExtractor:
    """
    Deterministic Cybersecurity Signal Extractor.
    Extracts 21 message-level, URL-level, and social-engineering indicators.
    """
    
    URGENCY_KEYWORDS = [
        r'\burgent\b', r'\bimmediately\b', r'\bwithin \d+ hours?\b', r'\baction required\b',
        r'\bexpire[s]? today\b', r'\blast chance\b', r'\bnow\b', r'\binstant\b', r'\basap\b',
        r'\bsuspended within\b', r'\baccount closure\b', r'\bdeadline\b', r'\bterminate[d]?\b',
        r'\bfinal warning\b', r'\bimmediate attention\b', r'\bact fast\b'
    ]
    
    ACCOUNT_THREAT_KEYWORDS = [
        r'\bsuspend(ed)?\b', r'\bblock(ed)?\b', r'\block(ed)?\b', r'\brestrict(ed)?\b',
        r'\bunauthorized\b', r'\bsecurity alert\b', r'\bcompromise(d)?\b', r'\bflagged\b',
        r'\bdeactivat(e|ed)\b', r'\btermination\b', r'\baccess revoked\b', r'\bunusual activity\b',
        r'\bbreach(ed)?\b', r'\blogin attempt\b', r'\bsecurity risk\b'
    ]
    
    CREDENTIAL_KEYWORDS = [
        r'\bverify\b', r'\bconfirm\b', r'\bpassword\b', r'\bpin\b', r'\blogin\b',
        r'\bcredentials\b', r'\bsecurity details\b', r'\bidentity\b', r'\bpasscode\b',
        r'\bupdate details\b', r'\bvalidate\b', r'\breset password\b', r'\bkyc\b'
    ]
    
    OTP_KEYWORDS = [
        r'\botp\b', r'\bverification code\b', r'\bpasscode\b', r'\bone-time\b',
        r'\bsecurity code\b', r'\bauth code\b', r'\b2fa\b', r'\bmfa\b', r'\bsecret pin\b'
    ]
    
    PAYMENT_KEYWORDS = [
        r'\brefund\b', r'\breward\b', r'\bprize\b', r'\bclaim\b', r'\bbanking\b',
        r'\bwire transfer\b', r'\bpayment\b', r'\bcredit card\b', r'\bdebit\b',
        r'\bbill\b', r'\binvoice\b', r'\btransaction\b', r'\bwon\b', r'\bcash prize\b',
        r'\bcrypto\b', r'\bbitcoin\b', r'\bbonus\b', r'\blottery\b'
    ]
    
    IMPERSONATION_BRANDS = [
        r'\bamazon\b', r'\bpaypal\b', r'\bnetflix\b', r'\bapple\b', r'\bgoogle\b',
        r'\bmicrosoft\b', r'\bchase\b', r'\bwell[s]? fargo\b', r'\bbank of america\b',
        r'\bsbi\b', r'\bhdfc\b', r'\bicici\b', r'\baxis\b', r'\bwhatsapp\b', r'\bfacebook\b',
        r'\busps\b', r'\bfedex\b', r'\bdhl\b', r'\bups\b', r'\bmeta\b', r'\binstagram\b'
    ]

    SHORTENER_PATTERNS = [
        r'bit\.ly', r'tinyurl\.com', r'goo\.gl', r't\.co', r'is\.gd', r'buff\.ly',
        r'ow\.ly', r'tiny\.cc', r'rb\.gy', r'cutt\.ly', r'000webhostapp\.com',
        r'linktr\.ee', r's\.id', r'shorturl\.at'
    ]

    FEATURE_NAMES = [
        'message_length', 'word_count', 'sentence_count', 'uppercase_ratio',
        'digit_ratio', 'special_char_ratio', 'exclamation_count', 'question_count',
        'has_url', 'url_count', 'domain_count', 'has_ip_url', 'has_shortened_url',
        'has_http_only', 'url_length_max', 'urgency_score', 'account_threat',
        'credential_request', 'otp_request', 'payment_request', 'impersonation_flag'
    ]

    def extract_dict(self, text, channel="SMS"):
        if not isinstance(text, str):
            text = str(text or "")
            
        msg_len = len(text)
        words = text.split()
        word_cnt = len(words)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        sent_cnt = max(len(sentences), 1)
        
        uppercase_cnt = sum(1 for c in text if c.isupper())
        uppercase_ratio = round(uppercase_cnt / max(msg_len, 1), 4)
        
        digit_cnt = sum(1 for c in text if c.isdigit())
        digit_ratio = round(digit_cnt / max(msg_len, 1), 4)
        
        special_cnt = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = round(special_cnt / max(msg_len, 1), 4)
        
        exclamation_cnt = text.count('!')
        question_cnt = text.count('?')
        
        # URL extraction
        urls = re.findall(r'https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}/\S*', text)
        has_url = 1.0 if len(urls) > 0 else 0.0
        url_cnt = float(len(urls))
        
        domains = set()
        has_ip = 0.0
        has_shortener = 0.0
        has_http_only = 0.0
        max_url_len = 0.0
        
        for u in urls:
            max_url_len = max(max_url_len, float(len(u)))
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', u):
                has_ip = 1.0
            if any(re.search(pat, u, re.I) for pat in self.SHORTENER_PATTERNS):
                has_shortener = 1.0
            if u.startswith('http://') and not u.startswith('https://'):
                has_http_only = 1.0
            
            # Extract domain
            dom_match = re.search(r'https?://([^/]+)|www\.([^/]+)|([^/]+)', u)
            if dom_match:
                dom = dom_match.group(0)
                domains.add(dom)
                
        dom_cnt = float(len(domains))
        
        # Social Engineering Flags
        text_lower = text.lower()
        urgency_score = float(sum(1.0 for pat in self.URGENCY_KEYWORDS if re.search(pat, text_lower)))
        account_threat = 1.0 if any(re.search(pat, text_lower) for pat in self.ACCOUNT_THREAT_KEYWORDS) else 0.0
        credential_request = 1.0 if any(re.search(pat, text_lower) for pat in self.CREDENTIAL_KEYWORDS) else 0.0
        otp_request = 1.0 if any(re.search(pat, text_lower) for pat in self.OTP_KEYWORDS) else 0.0
        payment_request = 1.0 if any(re.search(pat, text_lower) for pat in self.PAYMENT_KEYWORDS) else 0.0
        impersonation_flag = 1.0 if any(re.search(pat, text_lower) for pat in self.IMPERSONATION_BRANDS) else 0.0
        
        return {
            'message_length': float(msg_len),
            'word_count': float(word_cnt),
            'sentence_count': float(sent_cnt),
            'uppercase_ratio': float(uppercase_ratio),
            'digit_ratio': float(digit_ratio),
            'special_char_ratio': float(special_ratio),
            'exclamation_count': float(exclamation_cnt),
            'question_count': float(question_cnt),
            'has_url': float(has_url),
            'url_count': float(url_cnt),
            'domain_count': float(dom_cnt),
            'has_ip_url': float(has_ip),
            'has_shortened_url': float(has_shortener),
            'has_http_only': float(has_http_only),
            'url_length_max': float(max_url_len),
            'urgency_score': float(urgency_score),
            'account_threat': float(account_threat),
            'credential_request': float(credential_request),
            'otp_request': float(otp_request),
            'payment_request': float(payment_request),
            'impersonation_flag': float(impersonation_flag)
        }

    def extract_features_array(self, df):
        """
        Transforms a DataFrame or list of dicts with 'cleaned_text' and optional 'channel'
        into a 21-dimensional numpy feature matrix.
        """
        if isinstance(df, pd.DataFrame):
            records = df.to_dict('records')
        elif isinstance(df, list):
            records = df
        else:
            raise TypeError("Expected pandas.DataFrame or list of records.")
            
        features_list = []
        for row in records:
            t = row.get('cleaned_text', '')
            ch = row.get('channel', 'SMS')
            f_dict = self.extract_dict(t, ch)
            feat_vec = [f_dict[k] for k in self.FEATURE_NAMES]
            features_list.append(feat_vec)
            
        return np.array(features_list, dtype=np.float32)

if __name__ == '__main__':
    extractor = CyberFeatureExtractor()
    sample = "URGENT: Your SBI account is blocked! Verify password now at http://bit.ly/3x8q or call 88600."
    feats = extractor.extract_dict(sample, "SMS")
    print(f"Sample Cyber Signals Extracted ({len(feats)} total):")
    for k, v in feats.items():
        print(f"  {k:22s}: {v}")
