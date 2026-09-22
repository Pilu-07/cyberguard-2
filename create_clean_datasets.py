import json
import re
import html
import os
import pandas as pd
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    t = html.unescape(text)
    
    # Fix unicode non-breaking spaces and common corruption symbols
    t = t.replace('\xa0', ' ').replace('\xa0', ' ').replace('\r', '\n')
    t = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', t)
    
    # Collapse excess blank lines / spaces while keeping single newline if relevant
    lines = [line.strip() for line in t.split('\n')]
    lines = [l for l in lines if l]
    t_clean = ' '.join(lines)
    
    return t_clean

def is_raw_url(text):
    t = text.strip()
    # Direct URLs
    if t.startswith(('http://', 'https://', 'www.')):
        return True
    # Single token paths / domain strings with no spaces
    if len(t) < 200 and '/' in t and '.' in t and not ' ' in t:
        return True
    return False

def is_html_webpage(text):
    return bool(re.search(r'<!doctype|<html|<body|<div|<head|<title>', text, re.I))

def classify_channel(text):
    t = text.strip()
    
    # Email detection (headers, length, email specific syntax)
    if re.search(r'^(Subject:|From:|To:|MIME-Version:|X-Mailer:|Received:)', t, re.I | re.M) or 'mailto:' in t:
        return 'EMAIL'
    
    # Social / Chat detection (@handles, #hashtags, chat platform links, emojis)
    if re.search(r'(@\w+|#\w+|http\S*t\.co|discord|whatsapp|telegram|dm me|snapchat|insta|tweet)', t, re.I):
        return 'SOCIAL_CHAT'
    
    # SMS detection (short length, mobile spam phrases, shortcodes)
    if len(t) <= 300:
        if re.search(r'(\bstop\b|\btxt\b|\bcall\b|\bfree\b|\bclaim\b|\bwin\b|\burgent\b|\bcontact\b|\bwon\b|\bprize\b|\bmsg\b|\btext\b|\bhandset\b|\bmobile\b|\boptout\b)', t, re.I) or re.search(r'\b\d{5}\b', t):
            return 'SMS'
        return 'SMS' # Default short text to SMS
    
    # Long text defaults to EMAIL
    return 'EMAIL'

def process_dataset(json_path, output_dir):
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    print(f"Total raw records: {len(raw_data)}")
    
    processed_records = []
    purged_url_count = 0
    purged_html_count = 0
    purged_empty_count = 0
    
    for idx, item in enumerate(raw_data):
        raw_t = str(item.get('text', ''))
        lbl = item.get('label')
        
        # Binary label check
        if lbl not in (0, 1):
            continue
            
        cleaned_t = clean_text(raw_t)
        
        if not cleaned_t or len(cleaned_t) < 3:
            purged_empty_count += 1
            continue
            
        if is_raw_url(raw_t):
            purged_url_count += 1
            continue
            
        if is_html_webpage(raw_t):
            purged_html_count += 1
            continue
            
        channel = classify_channel(cleaned_t)
        
        # Structural Feature Extraction
        msg_len = len(cleaned_t)
        words = cleaned_t.split()
        word_cnt = len(words)
        
        has_url = bool(re.search(r'https?://\S+|www\.\S+', cleaned_t))
        url_cnt = len(re.findall(r'https?://\S+|www\.\S+', cleaned_t))
        has_phone = bool(re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}|\b\d{5,6}\b', cleaned_t))
        has_email_hdr = bool(re.search(r'^(Subject:|From:|To:)', cleaned_t, re.I | re.M))
        
        # Case & Symbol Ratios
        uppercase_chars = sum(1 for c in cleaned_t if c.isupper())
        uppercase_ratio = round(uppercase_chars / max(msg_len, 1), 4)
        special_chars = sum(1 for c in cleaned_t if not c.isalnum() and not c.isspace())
        special_ratio = round(special_chars / max(msg_len, 1), 4)
        
        processed_records.append({
            'channel': channel,
            'raw_text': raw_t,
            'cleaned_text': cleaned_t,
            'label': int(lbl),
            'label_name': 'Phishing' if int(lbl) == 1 else 'Legitimate',
            'message_length': msg_len,
            'word_count': word_cnt,
            'has_url': has_url,
            'url_count': url_cnt,
            'has_phone': has_phone,
            'has_email_header': has_email_hdr,
            'uppercase_ratio': uppercase_ratio,
            'special_char_ratio': special_ratio
        })
        
    print(f"Purged Raw URLs: {purged_url_count}")
    print(f"Purged HTML Webpages: {purged_html_count}")
    print(f"Purged Empty/Corrupt: {purged_empty_count}")
    print(f"Valid Natural Language Records: {len(processed_records)}")
    
    df = pd.DataFrame(processed_records)
    
    # Deduplication based on cleaned_text
    initial_count = len(df)
    df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
    dedup_count = initial_count - len(df)
    print(f"Deduplicated records removed: {dedup_count}")
    
    # Add unique message_id
    df.insert(0, 'message_id', [f"MSG_{i+1:06d}" for i in range(len(df))])
    
    # Reorder columns
    cols = [
        'message_id', 'channel', 'label', 'label_name', 'cleaned_text', 
        'raw_text', 'message_length', 'word_count', 'has_url', 'url_count', 
        'has_phone', 'has_email_header', 'uppercase_ratio', 'special_char_ratio'
    ]
    df = df[cols]
    
    # Save Master CSV
    master_path = os.path.join(output_dir, 'unified_messages_clean.csv')
    df.to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved Master CSV -> {master_path} ({len(df)} rows)")
    
    # Save Channel Specific CSVs
    email_df = df[df['channel'] == 'EMAIL'].reset_index(drop=True)
    sms_df = df[df['channel'] == 'SMS'].reset_index(drop=True)
    social_df = df[df['channel'] == 'SOCIAL_CHAT'].reset_index(drop=True)
    
    email_path = os.path.join(output_dir, 'email_messages_clean.csv')
    sms_path = os.path.join(output_dir, 'sms_messages_clean.csv')
    social_path = os.path.join(output_dir, 'social_chat_messages_clean.csv')
    
    email_df.to_csv(email_path, index=False, encoding='utf-8-sig')
    sms_df.to_csv(sms_path, index=False, encoding='utf-8-sig')
    social_df.to_csv(social_path, index=False, encoding='utf-8-sig')
    
    print(f"Saved EMAIL CSV -> {email_path} ({len(email_df)} rows)")
    print(f"Saved SMS CSV -> {sms_path} ({len(sms_df)} rows)")
    print(f"Saved SOCIAL/CHAT CSV -> {social_path} ({len(social_df)} rows)")
    
    # Data Quality Metrics Verification
    total_valid = len(df)
    null_count = df['cleaned_text'].isnull().sum()
    duplicate_count = df.duplicated(subset=['cleaned_text']).sum()
    valid_text_count = (df['message_length'] >= 3).sum()
    valid_channel_count = df['channel'].isin(['EMAIL', 'SMS', 'SOCIAL_CHAT']).sum()
    
    completeness_score = (1 - (null_count / total_valid)) * 100
    uniqueness_score = (1 - (duplicate_count / total_valid)) * 100
    validity_score = (valid_text_count / total_valid) * 100
    channel_validity_score = (valid_channel_count / total_valid) * 100
    
    overall_quality_score = (completeness_score + uniqueness_score + validity_score + channel_validity_score) / 4.0
    
    print("\n================ DATA QUALITY AUDIT REPORT ================")
    print(f" Completeness Score: {completeness_score:.2f}%")
    print(f" Uniqueness Score:   {uniqueness_score:.2f}%")
    print(f" Validity Score:     {validity_score:.2f}%")
    print(f" Channel Tag Score:  {channel_validity_score:.2f}%")
    print(f" ---------------------------------------------------------")
    print(f" OVERALL DATA QUALITY SCORE: {overall_quality_score:.2f}% (Target: 95%+ PASS)")
    print("===========================================================")

if __name__ == '__main__':
    process_dataset(r'e:\project 2\combined_reduced.json', r'e:\project 2')
