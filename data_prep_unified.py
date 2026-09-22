import os
import sys
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from url_intelligence.preprocessing import URLCanonicalizer

DATA_DIR = r'e:\project 2\data'
URL_DATASET_PATH = r'e:\project 2\url\clean url_cleaned.csv'
QR_BENIGN_PATH = r'e:\project 2\qr_decoding_output\benign_decoded.csv'
QR_MALICIOUS_PATH = r'e:\project 2\qr_decoding_output\malicious_decoded.csv'

def prepare():
    print("[1/6] Loading Existing URL Dataset via fast chunked stream...", flush=True)
    # Stream top 25k benign and 25k malicious directly from existing url dataset
    b_urls = []
    m_urls = []
    
    with open(URL_DATASET_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find index of URL and label
        url_idx = header.index('URL') if 'URL' in header else 0
        lbl_idx = header.index('label') if 'label' in header else -2
        
        for row in reader:
            if len(row) > max(url_idx, lbl_idx):
                u = row[url_idx].strip()
                l = row[lbl_idx].strip()
                if l == '0' and len(b_urls) < 15000:
                    b_urls.append({'raw_url': u, 'label': 0, 'source': 'existing_url'})
                elif l == '1' and len(m_urls) < 15000:
                    m_urls.append({'raw_url': u, 'label': 1, 'source': 'existing_url'})
                if len(b_urls) >= 15000 and len(m_urls) >= 15000:
                    break

    print(f"Loaded existing URLs: {len(b_urls)} Benign, {len(m_urls)} Malicious", flush=True)

    print("[2/6] Loading QR-Decoded URLs...", flush=True)
    qr_b_rows = []
    with open(QR_BENIGN_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            u = r.get('decoded_content') or r.get('normalized_url')
            if u:
                qr_b_rows.append({'raw_url': u.strip(), 'label': 0, 'source': 'qr_decoded'})
            if len(qr_b_rows) >= 15000:
                break

    qr_m_rows = []
    with open(QR_MALICIOUS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            u = r.get('decoded_content') or r.get('normalized_url')
            if u:
                qr_m_rows.append({'raw_url': u.strip(), 'label': 1, 'source': 'qr_decoded'})
            if len(qr_m_rows) >= 15000:
                break

    print(f"Loaded QR URLs: {len(qr_b_rows)} Benign, {len(qr_m_rows)} Malicious", flush=True)

    all_rows = b_urls + m_urls + qr_b_rows + qr_m_rows
    print(f"[3/6] Total raw consolidated items: {len(all_rows)}", flush=True)

    print("[4/6] Canonicalizing and extracting root domains...", flush=True)
    clean_data = []
    seen = set()
    for item in all_rows:
        canon = URLCanonicalizer.canonicalize(item['raw_url'])
        norm = canon['normalized_url']
        if not norm or norm in seen:
            continue
        seen.add(norm)
        root = URLCanonicalizer.extract_root_domain(canon['domain'])
        clean_data.append({
            'raw_url': item['raw_url'],
            'normalized_url': norm,
            'domain': canon['domain'],
            'root_domain': root,
            'label': item['label'],
            'source': item['source']
        })

    df = pd.DataFrame(clean_data)
    print(f"Clean deduplicated corpus: {len(df)} rows. Labels: {df['label'].value_counts().to_dict()}", flush=True)
    df.to_csv(os.path.join(DATA_DIR, 'unified_url_clean.csv'), index=False)

    print("[5/6] Performing Zero-Leakage Domain Partitioning...", flush=True)
    unique_domains = df['root_domain'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_domains)

    # 15% domains for Test 2 (Unseen-Domain Test)
    n_unseen = int(len(unique_domains) * 0.15)
    unseen_dom_set = set(unique_domains[:n_unseen])
    train_dom_set = set(unique_domains[n_unseen:])

    df_unseen_dom = df[df['root_domain'].isin(unseen_dom_set)].copy()
    df_trainable = df[df['root_domain'].isin(train_dom_set)].copy()

    # QR unseen test set (Test 3)
    qr_pool = df_trainable[df_trainable['source'] == 'qr_decoded']
    non_qr_pool = df_trainable[df_trainable['source'] != 'qr_decoded']

    qr_train, qr_test = train_test_split(qr_pool, test_size=0.25, random_state=42, stratify=qr_pool['label'])
    
    train_pool = pd.concat([non_qr_pool, qr_train], ignore_index=True)
    train_df, test_random = train_test_split(train_pool, test_size=0.15, random_state=42, stratify=train_pool['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['label'])

    # Test 4: Temporal / High-Entropy Campaign Shift test
    # Sample long/complex campaign URLs equally from benign and malicious
    b_long = test_random[(test_random['label'] == 0) & (test_random['normalized_url'].str.len() > 35)]
    m_long = test_random[(test_random['label'] == 1) & (test_random['normalized_url'].str.len() > 35)]
    n_sample_each = min(len(b_long), len(m_long), 250)
    test_temporal = pd.concat([b_long.sample(n_sample_each, random_state=42), m_long.sample(n_sample_each, random_state=42)], ignore_index=True)

    print("[6/6] Writing Split Artifacts to disk...", flush=True)
    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'validation.csv'), index=False)
    test_random.to_csv(os.path.join(DATA_DIR, 'test_random.csv'), index=False)
    df_unseen_dom.to_csv(os.path.join(DATA_DIR, 'test_unseen_domain.csv'), index=False)
    qr_test.to_csv(os.path.join(DATA_DIR, 'test_qr_unseen.csv'), index=False)
    test_temporal.to_csv(os.path.join(DATA_DIR, 'test_temporal.csv'), index=False)

    print("\n" + "="*50)
    print("Zero-Leakage Multi-Set Partitioning Complete!")
    print("="*50)
    print(f"Train set:           {len(train_df)} rows")
    print(f"Validation set:      {len(val_df)} rows")
    print(f"Test 1 (Random):     {len(test_random)} rows")
    print(f"Test 2 (Unseen Dom): {len(df_unseen_dom)} rows")
    print(f"Test 3 (QR Unseen):  {len(qr_test)} rows")
    print(f"Test 4 (Temporal):   {len(test_temporal)} rows")

    # Mathematical assertion
    overlap = set(train_df['root_domain']).intersection(set(df_unseen_dom['root_domain']))
    print(f"Domain overlap verification: {len(overlap)} (Must be 0)")
    assert len(overlap) == 0

if __name__ == '__main__':
    prepare()
