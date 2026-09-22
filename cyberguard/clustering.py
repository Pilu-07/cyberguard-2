import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict

def extract_template_skeleton(text):
    """
    Replaces dynamic entities (URLs, numbers, handles, emails, dates) with generic placeholders
    to compute structural template similarity and group near-duplicates into template clusters.
    """
    t = str(text or "").lower()
    # Replace URLs
    t = re.sub(r'https?://\S+|www\.\S+', '[URL]', t)
    # Replace emails
    t = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '[EMAIL]', t)
    # Replace handles
    t = re.sub(r'@\w+', '[USER]', t)
    # Replace numbers/amounts/phones
    t = re.sub(r'\b\d+[\d,.-]*\b', '[NUM]', t)
    # Collapse multiple whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t

class TemplateClusterSplitter:
    """
    Template-Aware Cluster Split Engine.
    Groups near-duplicate message templates to prevent template leakage between train and Test B.
    Also handles ingestion of external out-of-distribution Test C.
    """
    def __init__(self, df, text_col='cleaned_text', label_col='label', channel_col='channel'):
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.channel_col = channel_col

    def cluster_templates(self):
        """
        Groups rows into Template Family Clusters based on structural skeleton hashing.
        """
        skeletons = self.df[self.text_col].apply(extract_template_skeleton)
        self.df['template_skeleton'] = skeletons
        
        # Unique template IDs
        unique_templates = {sk: idx for idx, sk in enumerate(skeletons.unique())}
        self.df['template_id'] = skeletons.map(unique_templates)
        
        print(f"Total master rows: {len(self.df)} -> Identified Unique Template Families: {len(unique_templates)}")
        return self.df

    def create_anti_leakage_splits(self, test_a_ratio=0.10, test_b_ratio=0.10, val_ratio=0.10, seed=42):
        """
        Generates strict anti-leakage splits:
        - Train set (~70%)
        - Val set (~10%)
        - Test A (Standard Stratified Split) (~10%)
        - Test B (Template-Separated Split) (~10% - completely unseen template clusters)
        """
        np.random.seed(seed)
        df_clustered = self.cluster_templates()
        
        # Get template-level statistics
        template_stats = df_clustered.groupby('template_id').agg(
            size=(self.label_col, 'count'),
            label=(self.label_col, 'max'),
            channel=(self.channel_col, 'first')
        ).reset_index()
        
        # Shuffle template IDs
        template_stats = template_stats.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        total_rows = len(df_clustered)
        target_test_b_rows = int(total_rows * test_b_ratio)
        
        test_b_template_ids = set()
        accumulated_rows = 0
        
        for _, row in template_stats.iterrows():
            if accumulated_rows < target_test_b_rows:
                test_b_template_ids.add(row['template_id'])
                accumulated_rows += row['size']
            else:
                break
                
        # Split dataframe into Test B and Remaining
        is_test_b = df_clustered['template_id'].isin(test_b_template_ids)
        test_b_df = df_clustered[is_test_b].copy().reset_index(drop=True)
        remaining_df = df_clustered[~is_test_b].copy().reset_index(drop=True)
        
        print(f"Test B (Unseen Templates) rows: {len(test_b_df)} ({len(test_b_df)/total_rows*100:.1f}%)")
        print(f"Remaining rows for Train/Val/Test A: {len(remaining_df)}")
        
        # Stratified split on remaining for Train (~70%), Val (~10%), Test A (~10%)
        rem_total = len(remaining_df)
        test_a_frac = (total_rows * test_a_ratio) / rem_total
        val_frac = (total_rows * val_ratio) / rem_total
        
        # Stratification key
        remaining_df['strat_key'] = remaining_df[self.channel_col].astype(str) + "_" + remaining_df[self.label_col].astype(str)
        
        test_a_indices = []
        val_indices = []
        train_indices = []
        
        for key, group in remaining_df.groupby('strat_key'):
            shuffled_idx = group.index.to_numpy().copy()
            np.random.shuffle(shuffled_idx)
            
            n_group = len(shuffled_idx)
            n_test_a = int(n_group * test_a_frac)
            n_val = int(n_group * val_frac)
            
            test_a_indices.extend(shuffled_idx[:n_test_a])
            val_indices.extend(shuffled_idx[n_test_a:n_test_a+n_val])
            train_indices.extend(shuffled_idx[n_test_a+n_val:])
            
        train_df = remaining_df.loc[train_indices].drop(columns=['strat_key']).reset_index(drop=True)
        val_df = remaining_df.loc[val_indices].drop(columns=['strat_key']).reset_index(drop=True)
        test_a_df = remaining_df.loc[test_a_indices].drop(columns=['strat_key']).reset_index(drop=True)
        
        print("\nFinal Dataset Split Counts:")
        print(f"  Train Set:  {len(train_df):5d} rows ({len(train_df)/total_rows*100:.1f}%)")
        print(f"  Val Set:    {len(val_df):5d} rows ({len(val_df)/total_rows*100:.1f}%)")
        print(f"  Test A:     {len(test_a_df):5d} rows ({len(test_a_df)/total_rows*100:.1f}%)")
        print(f"  Test B:     {len(test_b_df):5d} rows ({len(test_b_df)/total_rows*100:.1f}%)")
        
        return train_df, val_df, test_a_df, test_b_df

def load_external_test_c(external_csv_path=r'e:\project 2\message cleaned_data.csv', unified_df=None):
    """
    Loads external holdout dataset for Test C (External Unseen Test).
    Filters out any texts that overlap with the master training dataset.
    Maps labels:
      - 'ham' -> 0 (Legitimate)
      - 'smishing' / 'spam' -> 1 (Phishing)
    """
    if not os.path.exists(external_csv_path):
        raise FileNotFoundError(f"External dataset not found at {external_csv_path}")
        
    ext_df = pd.read_csv(external_csv_path)
    
    # Filter out empty or null texts
    ext_df = ext_df.dropna(subset=['cleaned_text']).copy()
    ext_df['cleaned_text'] = ext_df['cleaned_text'].astype(str).str.strip()
    ext_df = ext_df[ext_df['cleaned_text'].str.len() >= 3].copy()
    
    # Remove duplicates
    ext_df = ext_df.drop_duplicates(subset=['cleaned_text']).copy()
    
    # Exclude overlap with unified dataset if provided
    if unified_df is not None:
        master_texts = set(unified_df['cleaned_text'].astype(str).str.strip())
        ext_df = ext_df[~ext_df['cleaned_text'].isin(master_texts)].copy()
        
    # Map label
    label_map = {'ham': 0, 'smishing': 1, 'spam': 1, 0: 0, 1: 1}
    if 'final_label' in ext_df.columns:
        ext_df['label'] = ext_df['final_label'].map(label_map)
    elif 'label' in ext_df.columns:
        ext_df['label'] = ext_df['label'].map(label_map)
    else:
        raise ValueError("Neither 'final_label' nor 'label' found in external dataset.")
        
    ext_df = ext_df.dropna(subset=['label']).copy()
    ext_df['label'] = ext_df['label'].astype(int)
    ext_df['label_name'] = ext_df['label'].map({0: 'Legitimate', 1: 'Phishing'})
    
    # Channel assignment: default to SMS/Chat given mobile spam nature
    if 'channel' not in ext_df.columns:
        ext_df['channel'] = 'SMS'
        
    ext_df = ext_df.reset_index(drop=True)
    print(f"\nLoaded Test C (External Unseen Dataset): {len(ext_df)} rows")
    print(f"  Distribution: Legitimate = {(ext_df['label']==0).sum()} | Phishing = {(ext_df['label']==1).sum()}")
    
    return ext_df

if __name__ == '__main__':
    df = pd.read_csv(r'e:\project 2\unified_messages_clean.csv')
    splitter = TemplateClusterSplitter(df)
    train_df, val_df, test_a_df, test_b_df = splitter.create_anti_leakage_splits()
    test_c_df = load_external_test_c(unified_df=df)
