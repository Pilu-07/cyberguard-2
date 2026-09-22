# 🛡️ CYBERGUARD — Benchmark Evaluation Matrix

Comprehensive multi-level empirical verification report for **CYBERGUARD** (Hybrid DeBERTa-v3 Semantic Representation + 21 Cyber Signal MLP Architecture).

## 1. Multi-Level Evaluation Matrix

| Evaluation Level | Samples | Accuracy | Macro F1 | Precision | Recall | Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Master Test Set** | 5079 | **97.32%** | 97.32% | 97.32% | 97.32% | >= 95.0% | PASS |
| **Channel: EMAIL** | 2256 | **97.52%** | 97.44% | 97.42% | 97.45% | >= 96.0% | PASS |
| **Channel: SMS** | 2082 | **97.07%** | 96.49% | 96.56% | 96.43% | >= 95.5% | PASS |
| **Channel: SOCIAL_CHAT** | 741 | **97.44%** | 96.61% | 97.56% | 95.75% | >= 94.0% | PASS |
| **Test A (Standard Stratified)** | 2538 | **97.36%** | 97.36% | 97.35% | 97.37% | >= 95.0% | PASS |
| **Test B (Unseen Templates)** | 2541 | **97.28%** | 97.28% | 97.29% | 97.28% | >= 93.5% | PASS |
| **Test C (External Unseen Dataset)** | 6309 | **94.67%** | 94.43% | 94.06% | 94.90% | >= 90.0% | PASS |

## 2. Confusion Matrix Breakdown

| Evaluation Level | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: |
| Overall Master Test Set | 2512 | 65 | 2431 | 71 |
| Channel: EMAIL | 901 | 29 | 1299 | 27 |
| Channel: SMS | 1433 | 32 | 588 | 29 |
| Channel: SOCIAL_CHAT | 178 | 4 | 544 | 15 |
| Test A (Standard Stratified) | 1263 | 29 | 1208 | 38 |
| Test B (Unseen Templates) | 1249 | 36 | 1223 | 33 |
| Test C (External Unseen Dataset) | 3646 | 100 | 2327 | 236 |

## 3. Generalization & Anti-Leakage Verification Summary
- **Test A (Standard Stratified Holdout)**: Validates baseline classification efficacy on known distribution.
- **Test B (Template-Separated Holdout)**: Validates zero-memorization generalization on unseen structural message templates.
- **Test C (External Unseen Holdout)**: Validates real-world deployment robustness on completely independent external phishing and smishing streams.
