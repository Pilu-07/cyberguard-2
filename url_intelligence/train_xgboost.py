import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from url_intelligence.url_features import URLFeatureExtractor
from url_intelligence.preprocessing import URLCanonicalizer
from url_intelligence.risk.calibration import URLRiskCalibrator

DATA_DIR = r'e:\project 2\data'
SAVED_MODELS_DIR = r'e:\project 2\saved_models'

def extract_features_df(df: pd.DataFrame) -> np.ndarray:
    print(f"Extracting 42 features across {len(df)} samples...", flush=True)
    extractor = URLFeatureExtractor()
    X = []
    for row in df.itertuples(index=False):
        f = extractor.extract_features(row.normalized_url)
        v = [f[name] for name in extractor.FEATURE_NAMES]
        X.append(v)
    return np.array(X, dtype=np.float32)

def train_baseline_gbm():
    print("="*70, flush=True)
    print("PHASE 10: TRAINING MODEL A (ENGINEERED FEATURES + HIGH-SPEED GRADIENT BOOSTING)", flush=True)
    print("="*70, flush=True)
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values

    # Check class balance
    benign_pct = (y_train == 0).mean() * 100
    mal_pct = (y_train == 1).mean() * 100
    print(f"\n[Phase 11: Class Distribution]: Benign={benign_pct:.1f}%, Malicious={mal_pct:.1f}%", flush=True)

    X_train = extract_features_df(train_df)
    X_val = extract_features_df(val_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\nTraining HistGradientBoostingClassifier (Fast histogram-based tree booster)...", flush=True)
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.01,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)

    # Validation predictions
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds)
    val_rec = recall_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"\n[Validation Results]: Acc={val_acc*100:.2f}%, F1={val_f1*100:.2f}%, Prec={val_prec*100:.2f}%, Rec={val_rec*100:.2f}%, AUC={val_auc:.4f}", flush=True)

    # Phase 14: Probability Calibration
    print("\n[Phase 14]: Fitting Isotonic Probability Calibration...", flush=True)
    calibrator = URLRiskCalibrator()
    calib_metrics = calibrator.fit(val_probs, y_val)
    print(f"Calibration Metrics: Raw ECE={calib_metrics['raw_ece']*100:.2f}%, Calibrated ECE={calib_metrics['calibrated_ece']*100:.2f}%", flush=True)

    # Save Bundle
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    bundle_path = os.path.join(SAVED_MODELS_DIR, 'url_intelligence_bundle.pkl')
    with open(bundle_path, 'wb') as f:
        pickle.dump({
            'model_type': 'tree',
            'model': model,
            'scaler': scaler,
            'calibrator': calibrator,
            'val_metrics': {
                'accuracy': val_acc,
                'f1': val_f1,
                'precision': val_prec,
                'recall': val_rec,
                'roc_auc': val_auc
            }
        }, f)
    print(f"\nModel bundle saved to: {bundle_path}", flush=True)

    return model, scaler, calibrator

if __name__ == '__main__':
    train_baseline_gbm()
