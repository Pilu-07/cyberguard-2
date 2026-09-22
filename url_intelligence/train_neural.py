import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from url_intelligence.url_features import URLFeatureExtractor
from url_intelligence.tokenizer import CharacterURLTokenizer
from url_intelligence.model import HybridURLIntelligenceModel
from url_intelligence.risk.calibration import URLRiskCalibrator

DATA_DIR = r'e:\project 2\data'
SAVED_MODELS_DIR = r'e:\project 2\saved_models'

class URLDataset(Dataset):
    def __init__(self, char_tokens, feat_vectors, labels):
        self.char_tokens = torch.tensor(char_tokens, dtype=torch.long)
        self.feat_vectors = torch.tensor(feat_vectors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.char_tokens[idx], self.feat_vectors[idx], self.labels[idx]

def extract_features_df(df: pd.DataFrame) -> np.ndarray:
    extractor = URLFeatureExtractor()
    X = []
    for row in df.itertuples(index=False):
        f = extractor.extract_features(row.normalized_url)
        v = [f[name] for name in extractor.FEATURE_NAMES]
        X.append(v)
    return np.array(X, dtype=np.float32)

def train_hybrid_model():
    print("="*70)
    print("PHASE 9 & 12: TRAINING MODEL C (HYBRID CHAR-CNN + FEATURE MLP)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))

    tokenizer = CharacterURLTokenizer(max_length=150)
    print("Encoding character sequences...")
    char_train = tokenizer.encode_batch(train_df['normalized_url'].tolist())
    char_val = tokenizer.encode_batch(val_df['normalized_url'].tolist())

    print("Extracting engineered features...")
    feat_train = extract_features_df(train_df)
    feat_val = extract_features_df(val_df)

    scaler = StandardScaler()
    feat_train_scaled = scaler.fit_transform(feat_train)
    feat_val_scaled = scaler.transform(feat_val)

    y_train = train_df['label'].values
    y_val = val_df['label'].values

    train_dataset = URLDataset(char_train, feat_train_scaled, y_train)
    val_dataset = URLDataset(char_val, feat_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 48,
        'char_out_dim': 128,
        'num_engineered_feats': 42,
        'mlp_out_dim': 64,
        'hidden_dim': 128,
        'num_classes': 2,
        'dropout': 0.2
    }

    model = HybridURLIntelligenceModel(**model_config).to(device)

    # Class weights for focal balance
    class_counts = np.bincount(y_train)
    weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32).to(device)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    best_val_f1 = 0.0
    best_state = None
    epochs = 8

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b_chars, b_feats, b_labels in train_loader:
            b_chars = b_chars.to(device)
            b_feats = b_feats.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_chars, b_feats)
            loss = criterion(logits, b_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for b_chars, b_feats, _ in val_loader:
                b_chars = b_chars.to(device)
                b_feats = b_feats.to(device)
                logits = model(b_chars, b_feats)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs_list.extend(probs)

        val_probs = np.array(val_probs_list)
        val_preds = (val_probs >= 0.5).astype(int)

        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds)
        val_prec = precision_score(y_val, val_preds)
        val_rec = recall_score(y_val, val_preds)

        print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Val Rec: {val_rec*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_state)

    # Compute best validation probabilities for calibration
    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for b_chars, b_feats, _ in val_loader:
            logits = model(b_chars.to(device), b_feats.to(device))
            val_probs_list.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    val_probs = np.array(val_probs_list)

    # Probability Calibration
    calibrator = URLRiskCalibrator()
    calib_metrics = calibrator.fit(val_probs, y_val)
    print(f"\n[Calibration]: Raw ECE={calib_metrics['raw_ece']*100:.2f}%, Calibrated ECE={calib_metrics['calibrated_ece']*100:.2f}%")

    # Save Bundle
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    bundle_path = os.path.join(SAVED_MODELS_DIR, 'url_hybrid_model_bundle.pkl')
    with open(bundle_path, 'wb') as f:
        pickle.dump({
            'model_type': 'neural',
            'model_config': model_config,
            'model_state_dict': best_state,
            'scaler': scaler,
            'calibrator': calibrator,
            'best_val_f1': best_val_f1
        }, f)
    print(f"Hybrid model saved successfully to: {bundle_path}")

if __name__ == '__main__':
    train_hybrid_model()
