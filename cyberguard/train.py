import os
import sys
import time
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from cyberguard.features import CyberFeatureExtractor
from cyberguard.clustering import TemplateClusterSplitter, load_external_test_c
from cyberguard.model import CyberGuardHybridModel, CyberGuardSemanticBaselineModel
from cyberguard.risk_engine import RiskEngine

MODEL_DIR = r'e:\project 2\saved_models'
BUNDLE_PATH = os.path.join(MODEL_DIR, 'cyberguard_bundle.pt')

def evaluate_predictions(y_true, y_pred):
    return {
        'acc': round(accuracy_score(y_true, y_pred) * 100, 2),
        'f1': round(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2),
        'prec': round(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2),
        'rec': round(recall_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)
    }

def train_stage1_baseline(train_df, val_df, test_a_df, test_b_df):
    """
    Stage 1 Baseline: TF-IDF + Logistic Regression.
    """
    print("\n" + "="*70)
    print("      STAGE 1 BASELINE: TF-IDF + LOGISTIC REGRESSION")
    print("="*70)
    
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    # Prepend channel token
    train_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(train_df['channel'], train_df['cleaned_text'])]
    val_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(val_df['channel'], val_df['cleaned_text'])]
    test_a_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(test_a_df['channel'], test_a_df['cleaned_text'])]
    test_b_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(test_b_df['channel'], test_b_df['cleaned_text'])]
    
    X_train = vec.fit_transform(train_texts)
    y_train = train_df['label'].values
    
    X_val = vec.transform(val_texts)
    y_val = val_df['label'].values
    
    X_test_a = vec.transform(test_a_texts)
    y_test_a = test_a_df['label'].values
    
    X_test_b = vec.transform(test_b_texts)
    y_test_b = test_b_df['label'].values
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    
    m_val = evaluate_predictions(y_val, clf.predict(X_val))
    m_a = evaluate_predictions(y_test_a, clf.predict(X_test_a))
    m_b = evaluate_predictions(y_test_b, clf.predict(X_test_b))
    
    print(f" Stage 1 Val:    Acc: {m_val['acc']}% | F1: {m_val['f1']}%")
    print(f" Stage 1 Test A: Acc: {m_a['acc']}% | F1: {m_a['f1']}% | Prec: {m_a['prec']}% | Rec: {m_a['rec']}%")
    print(f" Stage 1 Test B: Acc: {m_b['acc']}% | F1: {m_b['f1']}% | Prec: {m_b['prec']}% | Rec: {m_b['rec']}%")
    
    return {
        'model': clf,
        'vectorizer': vec,
        'val_metrics': m_val,
        'test_a_metrics': m_a,
        'test_b_metrics': m_b
    }

def train_stage2_baseline(train_text_t, val_text_t, test_a_text_t, test_b_text_t, 
                          y_train, y_val, y_test_a, y_test_b, epochs=3, batch_size=64, lr=1e-3, device=torch.device('cpu')):
    """
    Stage 2 Baseline: Semantic Representation Only (Text + Channel without Cyber Signals).
    """
    print("\n" + "="*70)
    print("      STAGE 2 BASELINE: SEMANTIC BRANCH ONLY (NEURAL BASELINE)")
    print("="*70)
    
    model = CyberGuardSemanticBaselineModel(text_dim=768, hidden_dim=256, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    ds_train = TensorDataset(train_text_t, y_train)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b_text, b_y in loader_train:
            b_text, b_y = b_text.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_text)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_y)
            
        train_loss = total_loss / len(train_text_t)
        
        # Val
        model.eval()
        with torch.no_grad():
            v_logits = model(val_text_t.to(device))
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
            v_m = evaluate_predictions(y_val.numpy(), v_preds)
        print(f" Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {v_m['acc']}% | Val F1: {v_m['f1']}%")
        
    # Evaluate Test A and Test B
    model.eval()
    with torch.no_grad():
        a_logits = model(test_a_text_t.to(device))
        a_preds = torch.argmax(a_logits, dim=1).cpu().numpy()
        m_a = evaluate_predictions(y_test_a.numpy(), a_preds)
        
        b_logits = model(test_b_text_t.to(device))
        b_preds = torch.argmax(b_logits, dim=1).cpu().numpy()
        m_b = evaluate_predictions(y_test_b.numpy(), b_preds)
        
    print(f" Stage 2 Test A: Acc: {m_a['acc']}% | F1: {m_a['f1']}% | Prec: {m_a['prec']}% | Rec: {m_a['rec']}%")
    print(f" Stage 2 Test B: Acc: {m_b['acc']}% | F1: {m_b['f1']}% | Prec: {m_b['prec']}% | Rec: {m_b['rec']}%")
    
    return {
        'model': model,
        'test_a_metrics': m_a,
        'test_b_metrics': m_b
    }

def train_stage3_hybrid(train_df, val_df, test_a_df, test_b_df, 
                        epochs=4, batch_size=64, lr=8e-4, device=torch.device('cpu')):
    """
    Stage 3: CYBERGUARD Hybrid Architecture (768-d Semantic + 21-d Cyber Signal MLP -> 832-d Fusion).
    """
    print("\n" + "="*70)
    print("      STAGE 3: CYBERGUARD HYBRID MODEL (SEMANTIC + CYBER SIGNALS)")
    print("="*70)
    
    extractor = CyberFeatureExtractor()
    
    # 1. Extract and normalize cyber features
    print("Extracting 21 deterministic cybersecurity signals across splits...")
    X_train_cyber_raw = extractor.extract_features_array(train_df)
    X_val_cyber_raw = extractor.extract_features_array(val_df)
    X_test_a_cyber_raw = extractor.extract_features_array(test_a_df)
    X_test_b_cyber_raw = extractor.extract_features_array(test_b_df)
    
    scaler = StandardScaler()
    X_train_cyber = scaler.fit_transform(X_train_cyber_raw)
    X_val_cyber = scaler.transform(X_val_cyber_raw)
    X_test_a_cyber = scaler.transform(X_test_a_cyber_raw)
    X_test_b_cyber = scaler.transform(X_test_b_cyber_raw)
    
    # 2. Text Representation Encoder (768-d semantic representation)
    print("Fitting 768-dimensional Semantic Text Representation...")
    train_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(train_df['channel'], train_df['cleaned_text'])]
    val_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(val_df['channel'], val_df['cleaned_text'])]
    test_a_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(test_a_df['channel'], test_a_df['cleaned_text'])]
    test_b_texts = [f"[CHANNEL: {ch}] {t}" for ch, t in zip(test_b_df['channel'], test_b_df['cleaned_text'])]
    
    vec = TfidfVectorizer(max_features=768, ngram_range=(1, 2), sublinear_tf=True)
    X_train_text = torch.tensor(vec.fit_transform(train_texts).toarray(), dtype=torch.float32)
    X_val_text = torch.tensor(vec.transform(val_texts).toarray(), dtype=torch.float32)
    X_test_a_text = torch.tensor(vec.transform(test_a_texts).toarray(), dtype=torch.float32)
    X_test_b_text = torch.tensor(vec.transform(test_b_texts).toarray(), dtype=torch.float32)
    
    y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
    y_val = torch.tensor(val_df['label'].values, dtype=torch.long)
    y_test_a = torch.tensor(test_a_df['label'].values, dtype=torch.long)
    y_test_b = torch.tensor(test_b_df['label'].values, dtype=torch.long)
    
    # Train Stage 2 Baseline for empirical comparison
    res_stage2 = train_stage2_baseline(
        X_train_text, X_val_text, X_test_a_text, X_test_b_text,
        y_train, y_val, y_test_a, y_test_b,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )
    
    # 3. Initialize Stage 3 Hybrid Model
    print("\nInitializing CYBERGUARD Hybrid Architecture (Fusion = 768 + 64 = 832-d)...")
    model = CyberGuardHybridModel(
        text_dim=768, 
        cyber_in_dim=21, 
        cyber_out_dim=64, 
        hidden_dim=256, 
        num_classes=2, 
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Cosine annealing scheduler
    total_steps = epochs * (len(train_df) // batch_size + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    ds_train = TensorDataset(X_train_text, torch.tensor(X_train_cyber, dtype=torch.float32), y_train)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b_text, b_cyber, b_y in loader_train:
            b_text, b_cyber, b_y = b_text.to(device), b_cyber.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(text_embeddings=b_text, cyber_features=b_cyber)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(b_y)
            
        train_loss = total_loss / len(train_df)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_cyber_t = torch.tensor(X_val_cyber, dtype=torch.float32).to(device)
            val_logits = model(text_embeddings=X_val_text.to(device), cyber_features=val_cyber_t)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            v_m = evaluate_predictions(y_val.numpy(), val_preds)
            
        print(f" Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {v_m['acc']}% | Val F1: {v_m['f1']}% | Prec: {v_m['prec']}% | Rec: {v_m['rec']}%")
        
    train_time = time.time() - t_start
    print(f"Training completed in {train_time:.1f}s.")
    
    # 4. Evaluation on Test A & Test B
    model.eval()
    with torch.no_grad():
        a_cyber_t = torch.tensor(X_test_a_cyber, dtype=torch.float32).to(device)
        test_a_logits = model(text_embeddings=X_test_a_text.to(device), cyber_features=a_cyber_t)
        test_a_preds = torch.argmax(test_a_logits, dim=1).cpu().numpy()
        test_a_probs = torch.softmax(test_a_logits, dim=1)[:, 1].cpu().numpy()
        m_a = evaluate_predictions(y_test_a.numpy(), test_a_preds)
        
        b_cyber_t = torch.tensor(X_test_b_cyber, dtype=torch.float32).to(device)
        test_b_logits = model(text_embeddings=X_test_b_text.to(device), cyber_features=b_cyber_t)
        test_b_preds = torch.argmax(test_b_logits, dim=1).cpu().numpy()
        test_b_probs = torch.softmax(test_b_logits, dim=1)[:, 1].cpu().numpy()
        m_b = evaluate_predictions(y_test_b.numpy(), test_b_preds)
        
    print("\n" + "="*70)
    print("      STAGE 3 CYBERGUARD HYBRID BENCHMARK RESULTS")
    print("="*70)
    print(f" Test A (Standard Stratified): Accuracy = {m_a['acc']}% | Macro F1 = {m_a['f1']}% | Prec = {m_a['prec']}% | Rec = {m_a['rec']}%")
    print(f" Test B (Unseen Templates):   Accuracy = {m_b['acc']}% | Macro F1 = {m_b['f1']}% | Prec = {m_b['prec']}% | Rec = {m_b['rec']}%")
    print("="*70)
    
    # 5. Fit Calibrated Risk Engine on Validation Set
    val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
    risk_eng = RiskEngine()
    calib_metrics = risk_eng.fit_calibration(val_probs, y_val.numpy())
    
    # 6. Save Complete Production Bundle
    bundle = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'text_dim': 768,
            'cyber_in_dim': 21,
            'cyber_out_dim': 64,
            'hidden_dim': 256,
            'num_classes': 2,
            'dropout': 0.2
        },
        'scaler': scaler,
        'vectorizer': vec,
        'feature_names': extractor.FEATURE_NAMES,
        'risk_engine': risk_eng,
        'stage2_metrics': res_stage2,
        'test_a_metrics': m_a,
        'test_b_metrics': m_b,
        'calibration_metrics': calib_metrics
    }
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(BUNDLE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\nSaved CYBERGUARD Production Bundle -> {BUNDLE_PATH}")
    
    return {
        'model': model,
        'scaler': scaler,
        'vectorizer': vec,
        'risk_engine': risk_eng,
        'test_a_metrics': m_a,
        'test_b_metrics': m_b,
        'stage2_metrics': res_stage2
    }

def run_full_training_pipeline():
    print("\n" + "#"*70)
    print("      CYBERGUARD END-TO-END PROGRESSIVE TRAINING PIPELINE")
    print("#"*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Hardware Engine: {device}")
    
    # Load Master Dataset
    master_csv = r'e:\project 2\unified_messages_clean.csv'
    print(f"Loading master clean dataset: {master_csv}")
    df = pd.read_csv(master_csv)
    
    # Generate Anti-Leakage Splits
    splitter = TemplateClusterSplitter(df)
    train_df, val_df, test_a_df, test_b_df = splitter.create_anti_leakage_splits()
    
    # Stage 1: TF-IDF + Logistic Regression
    res1 = train_stage1_baseline(train_df, val_df, test_a_df, test_b_df)
    
    # Stage 2 & 3: Semantic Baseline and CYBERGUARD Hybrid
    res3 = train_stage3_hybrid(train_df, val_df, test_a_df, test_b_df, epochs=4, device=device)
    
    # Progressive Comparison Summary Table
    print("\n" + "="*75)
    print("               PROGRESSIVE MULTI-STAGE BENCHMARK COMPARISON")
    print("="*75)
    print(f"{'Stage':<25} | {'Test A Acc':<11} | {'Test A F1':<10} | {'Test B Acc':<11} | {'Test B F1':<10}")
    print("-" * 75)
    print(f"{'Stage 1: TF-IDF + LR':<25} | {res1['test_a_metrics']['acc']:<10}% | {res1['test_a_metrics']['f1']:<9}% | {res1['test_b_metrics']['acc']:<10}% | {res1['test_b_metrics']['f1']:<9}%")
    print(f"{'Stage 2: Semantic NN Only':<25} | {res3['stage2_metrics']['test_a_metrics']['acc']:<10}% | {res3['stage2_metrics']['test_a_metrics']['f1']:<9}% | {res3['stage2_metrics']['test_b_metrics']['acc']:<10}% | {res3['stage2_metrics']['test_b_metrics']['f1']:<9}%")
    print(f"{'Stage 3: CYBERGUARD Hybrid':<25} | {res3['test_a_metrics']['acc']:<10}% | {res3['test_a_metrics']['f1']:<9}% | {res3['test_b_metrics']['acc']:<10}% | {res3['test_b_metrics']['f1']:<9}%")
    print("="*75)
    
    return res3

if __name__ == '__main__':
    run_full_training_pipeline()
