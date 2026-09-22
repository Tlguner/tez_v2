# -*- coding: utf-8 -*-
"""
Multiple Random Seeds Stability Validation Script
Trains top models across 5 different random seeds (0, 42, 100, 555, 2026) to compute mean +/- std accuracy and F1 score,
proving result stability and non-randomness for thesis defense.
"""

import os
import sys
import json
import io
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Multiple Seed Evaluation PyTorch Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

RUN_DIR = "thesis_pipeline/DL/runs/run_2026-08-05_18-37-42"
OUTPUT_JSON = "thesis_pipeline/plots/seed_stability_results.json"
OUTPUT_PLOT = "thesis_pipeline/plots/seed_stability_comparison.png"
os.makedirs("thesis_pipeline/plots", exist_ok=True)

models_dir = os.path.join(RUN_DIR, "models")
if models_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(models_dir))

try:
    from auto_DL_models import *
except Exception:
    pass

def load_data():
    X_train = np.load(os.path.join(RUN_DIR, "X_train.npy"))
    y_train = np.ravel(np.load(os.path.join(RUN_DIR, "y_train.npy"))).astype(int)
    X_test = np.load(os.path.join(RUN_DIR, "X_test.npy"))
    y_test = np.ravel(np.load(os.path.join(RUN_DIR, "y_test.npy"))).astype(int)
    return X_train, y_train, X_test, y_test

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def build_model(model_type, vocab_size=15000, embed_dim=128, units=128, dropout_rate=0.2):
    if model_type == "text_cnn":
        return TextCNNModel(vocab_size=vocab_size, embed_dim=embed_dim, filters=64, dropout_rate=dropout_rate)
    elif model_type == "hybrid":
        return HybridModel(vocab_size=vocab_size, embed_dim=embed_dim, filters=64, units=units, dropout_rate=dropout_rate)
    elif model_type == "bilstm":
        return BiLSTMModel(vocab_size=vocab_size, embed_dim=embed_dim, units=units, dropout_rate=dropout_rate)
    elif model_type == "auto_model":
        return AutoModel(vocab_size=vocab_size, embed_dim=embed_dim, units=units, dropout_rate=dropout_rate)
    else:
        return LSTMModel(vocab_size=vocab_size, embed_dim=embed_dim, units=units, dropout_rate=dropout_rate)

def train_and_eval_seed(model_type, X_train, y_train, X_test, y_test, seed, epochs=4, batch_size=256):
    set_seed(seed)
    
    model = build_model(model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    # Eval in batches
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
    preds = (proba >= 0.5).astype(int)
    
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

def main():
    print("\n--- Starting Multiple Random Seeds Validation ---")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Dataset shapes: Train={X_train.shape}, Test={X_test.shape}")
    
    seeds = [0, 42, 100, 555, 2026]
    
    # Dynamically select top-4 models based on test F1-score if results file exists
    target_models = ["text_cnn", "hybrid", "bilstm", "auto_model"]
    results_path = os.path.join(RUN_DIR, "auto_ML_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                res_dict = json.load(f)
            # Map canonical names to model_type strings
            name_map = {
                "text_cnn": "text_cnn", "textcnn": "text_cnn",
                "hybrid": "hybrid", "hybrid_cnn_dense": "hybrid",
                "bilstm": "bilstm", "bidirectional_lstm": "bilstm",
                "auto_model": "auto_model", "lstm": "lstm", "gru": "gru", "cnn": "cnn"
            }
            sorted_models = sorted(
                res_dict.items(),
                key=lambda item: item[1].get("f1_score", item[1].get("accuracy", 0)),
                reverse=True
            )
            dyn_targets = []
            for name, _ in sorted_models:
                clean_name = name.lower().replace("-", "_").replace(" ", "_")
                for k, v in name_map.items():
                    if k in clean_name and v not in dyn_targets:
                        dyn_targets.append(v)
                        break
                if len(dyn_targets) == 4:
                    break
            if len(dyn_targets) >= 3:
                target_models = dyn_targets
                print(f"[DYNAMIC SELECTION] Top 4 models selected for seed evaluation: {target_models}")
        except Exception as e:
            print(f"[WARN] Could not dynamically parse top models ({e}), using default top-4.")
    
    summary_results = {}
    
    for mtype in target_models:
        print(f"\n[EVAL] Model '{mtype}' across 5 random seeds...")
        seed_accs = []
        seed_f1s = []
        seed_details = []
        
        t0 = time.time()
        for seed in seeds:
            res = train_and_eval_seed(mtype, X_train, y_train, X_test, y_test, seed=seed)
            seed_accs.append(res["accuracy"])
            seed_f1s.append(res["f1"])
            seed_details.append({"seed": seed, "accuracy": round(res["accuracy"], 4), "f1": round(res["f1"], 4)})
            print(f"  -> Seed {seed:4d} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f}")
            
        mean_acc = float(np.mean(seed_accs))
        std_acc = float(np.std(seed_accs))
        mean_f1 = float(np.mean(seed_f1s))
        std_f1 = float(np.std(seed_f1s))
        elapsed = round(time.time() - t0, 2)
        
        summary_results[mtype] = {
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "formatted_accuracy": f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
            "mean_f1": round(mean_f1, 4),
            "std_f1": round(std_f1, 4),
            "formatted_f1": f"{mean_f1*100:.2f}% ± {std_f1*100:.2f}%",
            "execution_time_seconds": elapsed,
            "seed_runs": seed_details
        }
        print(f" ==> {mtype:10s} Mean Acc: {mean_acc*100:.2f}% ± {std_acc*100:.2f}% | Mean F1: {mean_f1*100:.2f}% ± {std_f1*100:.2f}%")
        
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
    # Plotting Stability Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(summary_results.keys())
    means = [summary_results[m]["mean_accuracy"] for m in model_names]
    stds = [summary_results[m]["std_accuracy"] for m in model_names]
    
    bars = ax.bar(model_names, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.85)
    ax.set_ylim(0.90, 1.0)
    ax.set_title("Model Accuracy Stability Across 5 Random Seeds (Mean ± Std)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{mean*100:.2f}%\n(±{std*100:.2f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] Multiple seed evaluation complete! Saved to {OUTPUT_JSON} and {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
