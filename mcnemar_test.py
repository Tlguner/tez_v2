# -*- coding: utf-8 -*-
"""
Universal McNemar Statistical Significance Test Script
Dynamically compares test predictions of top models (DL vs DL, ML vs ML, DL vs ML)
across any dataset (18k, 82k, etc.) to evaluate if performance differences are statistically significant (p < 0.05).
"""

import os
import sys
import glob
import json
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add possible model paths to sys.path
for root, dirs, _ in os.walk("thesis_pipeline"):
    if "models" in dirs:
        m_path = os.path.abspath(os.path.join(root, "models"))
        if m_path not in sys.path:
            sys.path.insert(0, m_path)

try:
    from auto_DL_models import *
except Exception:
    pass

def find_latest_run(mode="DL"):
    runs_dir = f"thesis_pipeline/{mode}/runs"
    if os.path.exists(runs_dir):
        valid = [r for r in glob.glob(os.path.join(runs_dir, "run_*")) if os.path.exists(os.path.join(r, "models")) and os.path.exists(os.path.join(r, "auto_ML_results.json"))]
        valid.sort(key=lambda x: os.path.basename(x).replace("-", "").replace("_", ""))
        if valid:
            return valid[-1]
    latest = f"thesis_pipeline/{mode}/latest"
    if os.path.exists(latest):
        return latest
    return None

def load_dataset_from_run(run_dir):
    """Loads X_test and y_test from a run directory (supports both .npy and .npz)."""
    # Try .npy first
    x_npy = os.path.join(run_dir, "X_test.npy")
    y_npy = os.path.join(run_dir, "y_test.npy")
    
    if os.path.exists(x_npy) and os.path.exists(y_npy):
        X_test = np.load(x_npy)
        y_test = np.ravel(np.load(y_npy)).astype(int)
        return X_test, y_test
    
    # Try .npz for sparse ML features
    x_npz = os.path.join(run_dir, "X_test.npz")
    if os.path.exists(x_npz) and os.path.exists(y_npy):
        import scipy.sparse
        X_test = scipy.sparse.load_npz(x_npz)
        y_test = np.ravel(np.load(y_npy)).astype(int)
        return X_test, y_test
        
    return None, None

def get_predictions_for_model(model_path, X_test):
    """Universal predictor supporting PyTorch, Scikit-learn, and Keras models."""
    if not os.path.exists(model_path):
        return None
        
    # 1. Scikit-learn (.pkl / .joblib)
    if model_path.endswith(".pkl") or model_path.endswith(".joblib"):
        try:
            model = joblib.load(model_path)
            # If X_test is sparse and model expects dense (e.g. GaussianNB)
            if hasattr(X_test, "toarray") and type(model).__name__ in ["GaussianNB", "HistGradientBoostingClassifier"]:
                return model.predict(X_test.toarray())
            return model.predict(X_test)
        except Exception as e:
            return None

    # 2. PyTorch (.pt / .h5)
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(model, nn.Module) or hasattr(model, "eval"):
            model.to(device)
            model.eval()
            X_tensor = torch.tensor(X_test, dtype=torch.long) if isinstance(X_test, np.ndarray) else torch.tensor(X_test.toarray(), dtype=torch.long)
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), 512):
                    batch_x = X_tensor[i:i+512].to(device)
                    outputs = model(batch_x)
                    if outputs.shape[-1] == 1:
                        probs = outputs.squeeze(-1)
                    else:
                        probs = torch.softmax(outputs, dim=-1)[:, 1]
                    all_probs.append(probs.cpu().numpy())
            proba = np.concatenate(all_probs).ravel()
            return (proba >= 0.5).astype(int)
    except Exception:
        pass

    # 3. Keras fallback
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        proba = model.predict(X_test, batch_size=512, verbose=0).ravel()
        return (proba >= 0.5).astype(int)
    except Exception:
        pass

    return None

def run_mcnemar(preds_A, preds_B, y_true, name_A, name_B):
    """Calculates contingency matrix and McNemar Chi-Square p-value with continuity correction."""
    if len(preds_A) != len(preds_B) or len(preds_A) != len(y_true):
        return None
        
    correct_A = (preds_A == y_true)
    correct_B = (preds_B == y_true)
    
    a = int(np.sum(correct_A & correct_B))
    b = int(np.sum(correct_A & (~correct_B)))
    c = int(np.sum((~correct_A) & correct_B))
    d = int(np.sum((~correct_A) & (~correct_B)))
    
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
    target_run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not target_run_dir or not os.path.exists(target_run_dir):
        # Default to latest DL run or latest ML run
        target_run_dir = find_latest_run("DL") or find_latest_run("ML")
        
    if not target_run_dir or not os.path.exists(target_run_dir):
        print("[ERROR] No valid run directory found.")
        return

    print(f"\n--- Universal McNemar Significance Testing ---")
    print(f"Target Run Directory: {target_run_dir}")
    
    X_test, y_test = load_dataset_from_run(target_run_dir)
    if X_test is None or y_test is None:
        print("[ERROR] Could not load test dataset from run directory.")
        return
        
    print(f"Loaded Test Samples: {len(y_test)}")
    
    models_dir = os.path.join(target_run_dir, "models")
    if not os.path.exists(models_dir):
        print(f"[ERROR] Models directory not found: {models_dir}")
        return

    # Find all available models in this run
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pt', '.h5', '.pkl', '.joblib'))]
    model_predictions = {}
    
    for mf in model_files:
        mname = os.path.splitext(mf)[0]
        mpath = os.path.join(models_dir, mf)
        preds = get_predictions_for_model(mpath, X_test)
        if preds is not None:
            model_predictions[mname] = preds
            acc = float(np.mean(preds == y_test))
            print(f"  -> Model '{mname:15s}' loaded | Test Accuracy: {acc*100:.2f}%")

    if len(model_predictions) < 2:
        print("[WARN] Need at least 2 models to perform McNemar comparisons.")
        return

    # Sort models by accuracy descending
    sorted_models = sorted(model_predictions.keys(), key=lambda k: np.mean(model_predictions[k] == y_test), reverse=True)
    best_model_name = sorted_models[0]
    print(f"\nChampion Model: {best_model_name}")

    tests_summary = []
    
    # 1. Compare Champion vs other top models in the same run
    for other_name in sorted_models[1:]:
        t = run_mcnemar(model_predictions[best_model_name], model_predictions[other_name], y_test, best_model_name, other_name)
        if t:
            tests_summary.append(t)
            print(f"\n[TEST] {t['pair']}")
            print(f"  -> b ({best_model_name} correct, {other_name} wrong): {t['contingency_matrix']['b']}")
            print(f"  -> c ({other_name} correct, {best_model_name} wrong): {t['contingency_matrix']['c']}")
            print(f"  -> p-value: {t['p_value_formatted']} | {t['conclusion']}")

    # 2. Check for Counterpart Run (ML vs DL) on matching sample size
    is_dl = "DL" in target_run_dir
    counterpart_mode = "ML" if is_dl else "DL"
    
    for c_run in sorted(glob.glob(f"thesis_pipeline/{counterpart_mode}/runs/run_*"), reverse=True):
        cX_test, cy_test = load_dataset_from_run(c_run)
        if cy_test is not None and len(cy_test) == len(y_test):
            # Matching dataset size found!
            c_models_dir = os.path.join(c_run, "models")
            if not os.path.exists(c_models_dir):
                continue
            c_model_files = [f for f in os.listdir(c_models_dir) if f.endswith(('.pt', '.h5', '.pkl', '.joblib'))]
            for cmf in c_model_files:
                cmname = os.path.splitext(cmf)[0]
                cmpath = os.path.join(c_models_dir, cmf)
                cpreds = get_predictions_for_model(cmpath, cX_test)
                if cpreds is not None and len(cpreds) == len(y_test):
                    t_cross = run_mcnemar(model_predictions[best_model_name], cpreds, y_test, f"{best_model_name} ({'DL' if is_dl else 'ML'})", f"{cmname} ({counterpart_mode})")
                    if t_cross:
                        tests_summary.append(t_cross)
                        print(f"\n[CROSS-MODE TEST] {t_cross['pair']}")
                        print(f"  -> p-value: {t_cross['p_value_formatted']} | {t_cross['conclusion']}")
                    break
            break

    # Save results to target run directory and plots directory
    os.makedirs("thesis_pipeline/plots", exist_ok=True)
    plots_json = "thesis_pipeline/plots/mcnemar_test_results.json"
    run_json = os.path.join(target_run_dir, "mcnemar_test_results.json")
    
    with open(plots_json, "w", encoding="utf-8") as f:
        json.dump(tests_summary, f, indent=2, ensure_ascii=False)
        
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(tests_summary, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] McNemar tests saved to {run_json} and {plots_json}")

if __name__ == "__main__":
    main()
