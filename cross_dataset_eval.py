# -*- coding: utf-8 -*-
"""
Cross-Dataset Generalization Evaluation Script
Evaluates models trained on Dataset 1 (82.5k) against Dataset 2 (18.6k) test split,
and models trained on Dataset 2 (18.6k) against Dataset 1 (82.5k) test split.
"""

import os
import sys
import json
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Cross-Dataset Evaluation PyTorch Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Run Directories
D1_RUN_DIR = "thesis_pipeline/DL/runs/run_2026-08-05_18-37-42"  # 82.5k Dataset (Seq len 400)
D2_RUN_DIR = "thesis_pipeline/DL/runs/run_2026-07-07_12-54-31"  # 18.6k Dataset (Seq len 200)

OUTPUT_JSON = "thesis_pipeline/plots/cross_dataset_results.json"
os.makedirs("thesis_pipeline/plots", exist_ok=True)

# Add run models directories to sys.path
for rdir in [D1_RUN_DIR, D2_RUN_DIR]:
    m_abs = os.path.abspath(os.path.join(rdir, "models"))
    if m_abs not in sys.path:
        sys.path.insert(0, m_abs)

try:
    from auto_DL_models import *
except Exception:
    pass

def load_data(run_dir):
    x_path = os.path.join(run_dir, "X_test.npy")
    y_path = os.path.join(run_dir, "y_test.npy")
    X_test = np.load(x_path)
    y_test = np.ravel(np.load(y_path)).astype(int)
    return X_test, y_test

def adjust_sequence_and_vocab(X, target_seq_len=400, max_vocab_size=15000):
    curr_len = X.shape[1]
    if curr_len < target_seq_len:
        pad_width = target_seq_len - curr_len
        X_out = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
    else:
        X_out = X[:, :target_seq_len]
    # Clip token IDs to valid vocabulary range for the target model
    return np.clip(X_out, 0, max_vocab_size - 1)

def evaluate_model(model_path, X_test, y_test, target_seq_len=400, max_vocab_size=15000):
    X_test_adj = adjust_sequence_and_vocab(X_test, target_seq_len, max_vocab_size)
    
    # Try PyTorch loading first
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(model, nn.Module) or hasattr(model, 'eval'):
            model.to(device)
            model.eval()
            
            X_test_tensor = torch.tensor(X_test_adj, dtype=torch.long)
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
            preds = (proba >= 0.5).astype(int)

            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, zero_division=0))
            rec = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, proba))
            except Exception:
                auc = 0.5

            return {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "auc": round(auc, 4)
            }
    except Exception:
        pass

    # Try Keras loading fallback
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        proba = model.predict(X_test_adj, batch_size=512, verbose=0).ravel()
        preds = (proba >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = 0.5

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4)
        }
    except Exception as e:
        print(f"[WARN] Failed to evaluate model {os.path.basename(model_path)}: {e}")
        return None

def main():
    print("\n--- Starting Bidirectional Cross-Dataset Evaluation (D1 <-> D2) ---")
    
    print("\nLoading test sets...")
    X_test_D1, y_test_D1 = load_data(D1_RUN_DIR)
    X_test_D2, y_test_D2 = load_data(D2_RUN_DIR)
    print(f"D1 Test Set Shape: {X_test_D1.shape}, D2 Test Set Shape: {X_test_D2.shape}")

    results = {
        "D1_trained_on_D2_test": {},
        "D2_trained_on_D1_test": {}
    }

    model_names = ["text_cnn", "hybrid", "bilstm", "gru", "cnn", "auto_model", "lstm"]

    # 1. Models trained on D1 (82.5k) tested on D2 (18.6k) - Target seq len 400
    print("\n[EVAL 1/2] D1 Models (82.5k) evaluated on D2 Test Set (18.6k padded to 400)...")
    models_D1_dir = os.path.join(D1_RUN_DIR, "models")
    for mname in model_names:
        mpath = os.path.join(models_D1_dir, f"{mname}.pt")
        if not os.path.exists(mpath):
            mpath = os.path.join(models_D1_dir, f"{mname}.h5")
        if os.path.exists(mpath):
            metrics = evaluate_model(mpath, X_test_D2, y_test_D2, target_seq_len=400)
            if metrics:
                results["D1_trained_on_D2_test"][mname] = metrics
                print(f"  -> {mname:12s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")

    # 2. Models trained on D2 (18.6k) tested on D1 (82.5k) - Target seq len 200, vocab 5000
    print("\n[EVAL 2/2] D2 Models (18.6k) evaluated on D1 Test Set (82.5k truncated to 200, vocab 5000)...")
    models_D2_dir = os.path.join(D2_RUN_DIR, "models")
    for mname in model_names:
        mpath = os.path.join(models_D2_dir, f"{mname}.h5")
        if not os.path.exists(mpath):
            mpath = os.path.join(models_D2_dir, f"{mname}.pt")
        if os.path.exists(mpath):
            metrics = evaluate_model(mpath, X_test_D1, y_test_D1, target_seq_len=200, max_vocab_size=5000)
            if metrics:
                results["D2_trained_on_D1_test"][mname] = metrics
                print(f"  -> {mname:12s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")

    # Save results to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Cross-Dataset Generalization evaluation complete! Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
