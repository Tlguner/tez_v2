# -*- coding: utf-8 -*-
"""
McNemar Statistical Significance Test Script
Compares test predictions of top DL & ML models to verify if accuracy differences are statistically significant (p < 0.05).
"""

import os
import sys
import json
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import scipy.stats as stats
from scipy.stats import chi2

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run Directory (D1 82.5k)
RUN_DIR = "thesis_pipeline/DL/runs/run_2026-08-05_18-37-42"
ML_RUN_DIR = "thesis_pipeline/ML/runs/run_2026-08-04_12-17-11"
OUTPUT_JSON = "thesis_pipeline/plots/mcnemar_test_results.json"
os.makedirs("thesis_pipeline/plots", exist_ok=True)

models_dir = os.path.join(RUN_DIR, "models")
if models_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(models_dir))

try:
    from auto_DL_models import *
except Exception:
    pass

def load_data():
    X_test = np.load(os.path.join(RUN_DIR, "X_test.npy"))
    y_test = np.ravel(np.load(os.path.join(RUN_DIR, "y_test.npy"))).astype(int)
    return X_test, y_test

def get_dl_predictions(model_name, X_test):
    mpath = os.path.join(models_dir, f"{model_name}.pt")
    if not os.path.exists(mpath):
        mpath = os.path.join(models_dir, f"{model_name}.h5")
    
    model = torch.load(mpath, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), 512):
            batch_x = X_test_tensor[i:i+512].to(device)
            outputs = model(batch_x)
            if outputs.shape[-1] == 1:
                probs = outputs.squeeze(-1)
            else:
                probs = torch.softmax(outputs, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
    proba = np.concatenate(all_probs).ravel()
    return (proba >= 0.5).astype(int)

def get_ml_predictions():
    # Load ML test features (X_test.npz) and best ML model (VotingClassifier) from run_2026-08-04_12-17-11
    ml_x_path = os.path.join(ML_RUN_DIR, "X_test.npz")
    ml_models_dir = os.path.join(ML_RUN_DIR, "models")
    ml_model_path = os.path.join(ml_models_dir, "VotingClassifier.pkl")
    if not os.path.exists(ml_model_path):
        ml_model_path = os.path.join(ml_models_dir, "votingclassifier.pkl")
    
    if os.path.exists(ml_model_path) and os.path.exists(ml_x_path):
        import scipy.sparse
        X_ml = scipy.sparse.load_npz(ml_x_path)
        model = joblib.load(ml_model_path)
        return model.predict(X_ml)
    return None

def run_mcnemar(preds_A, preds_B, y_true, name_A, name_B):
    correct_A = (preds_A == y_true)
    correct_B = (preds_B == y_true)
    
    # Contingency table
    #           B Correct | B Wrong
    # A Correct     a     |    b
    # A Wrong       c     |    d
    a = int(np.sum(correct_A & correct_B))
    b = int(np.sum(correct_A & (~correct_B)))
    c = int(np.sum((~correct_A) & correct_B))
    d = int(np.sum((~correct_A) & (~correct_B)))
    
    table = [[a, b], [c, d]]
    
    # Run McNemar test using continuity-corrected Chi-Square
    if (b + c) > 0:
        stat = float(((abs(b - c) - 1) ** 2) / (b + c))
        p_val = float(chi2.sf(stat, df=1))
    else:
        stat = 0.0
        p_val = 1.0
    significant = bool(p_val < 0.05)
    
    return {
        "pair": f"{name_A} vs {name_B}",
        "contingency_matrix": {"a": a, "b": b, "c": c, "d": d},
        "statistic": round(stat, 4),
        "p_value": p_val,
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "significant_at_005": significant,
        "conclusion": f"Fark istatistiksel olarak {'ANLAMLI (p < 0.05)' if significant else 'ANLAMSIZ (p >= 0.05)'}"
    }

def main():
    print("\n--- Starting McNemar Statistical Significance Test ---")
    X_test, y_test = load_data()
    print(f"Loaded Test Set: {len(y_test)} samples.")
    
    print("\nGetting predictions for models...")
    preds_textcnn = get_dl_predictions("text_cnn", X_test)
    preds_bilstm = get_dl_predictions("bilstm", X_test)
    preds_auto = get_dl_predictions("auto_model", X_test)
    preds_ml = get_ml_predictions()
    
    tests_summary = []
    
    # Test 1: TextCNN vs BiLSTM
    print("\n[TEST 1] TextCNN vs BiLSTM...")
    t1 = run_mcnemar(preds_textcnn, preds_bilstm, y_test, "TextCNN", "BiLSTM")
    tests_summary.append(t1)
    print(f"  -> b (TextCNN correct, BiLSTM wrong): {t1['contingency_matrix']['b']}")
    print(f"  -> c (BiLSTM correct, TextCNN wrong): {t1['contingency_matrix']['c']}")
    print(f"  -> p-value: {t1['p_value_formatted']} | {t1['conclusion']}")
    
    # Test 2: TextCNN vs auto_model
    print("\n[TEST 2] TextCNN vs auto_model...")
    t2 = run_mcnemar(preds_textcnn, preds_auto, y_test, "TextCNN", "auto_model")
    tests_summary.append(t2)
    print(f"  -> b (TextCNN correct, auto_model wrong): {t2['contingency_matrix']['b']}")
    print(f"  -> c (auto_model correct, TextCNN wrong): {t2['contingency_matrix']['c']}")
    print(f"  -> p-value: {t2['p_value_formatted']} | {t2['conclusion']}")

    # Test 3: TextCNN vs VotingClassifier (ML)
    if preds_ml is not None:
        print("\n[TEST 3] TextCNN (DL) vs VotingClassifier (ML)...")
        t3 = run_mcnemar(preds_textcnn, preds_ml, y_test, "TextCNN (DL)", "VotingClassifier (ML)")
        tests_summary.append(t3)
        print(f"  -> b (TextCNN correct, Voting ML wrong): {t3['contingency_matrix']['b']}")
        print(f"  -> c (Voting ML correct, TextCNN wrong): {t3['contingency_matrix']['c']}")
        print(f"  -> p-value: {t3['p_value_formatted']} | {t3['conclusion']}")
        
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tests_summary, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] McNemar test completed! Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
