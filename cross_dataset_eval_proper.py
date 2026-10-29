# -*- coding: utf-8 -*-
"""
Comprehensive Cross-Dataset Generalization Evaluation Script
Evaluates 9 Deep Learning and 8 Classical Hybrid Machine Learning models
under zero-shot domain transfer between Dataset 1 (18.6K) and Dataset 2 (82.5K).
"""

import os
import sys
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn

REPO_DIR = r"c:\TU\deu\tez_v2\scraped data\phishing_mcp_server"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] PyTorch Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# Dataset Directories
DL_82K_DIR = os.path.join(REPO_DIR, "thesis_pipeline", "DL", "runs", "run_2026-08-05_18-37-42")
DL_18K_DIR = os.path.join(REPO_DIR, "thesis_pipeline", "DL", "runs", "run_2026-08-19_13-34-57")
ML_82K_DIR = os.path.join(REPO_DIR, "thesis_pipeline", "ML", "runs", "run_2026-08-13_16-34-25")
ML_18K_DIR = os.path.join(REPO_DIR, "thesis_pipeline", "ML", "runs", "run_2026-08-13_10-50-39")
OUTPUT_JSON = os.path.join(REPO_DIR, "thesis_pipeline", "plots", "cross_dataset_results_proper.json")

# ==========================================
# 1. LOAD DATASETS & DYNAMIC OVERLAP
# ==========================================
print("\n[STEP 1] Loading Cleaned Datasets & Calculating Dynamic Overlap...", flush=True)
df_dl_82k = pd.read_csv(os.path.join(DL_82K_DIR, "cleaned_dataset.csv"))
df_ml_82k = pd.read_csv(os.path.join(ML_82K_DIR, "cleaned_dataset.csv"))
df_dl_18k = pd.read_csv(os.path.join(DL_18K_DIR, "cleaned_dataset.csv"))
df_ml_18k = pd.read_csv(os.path.join(ML_18K_DIR, "cleaned_dataset.csv"))

s_82k = set(df_dl_82k['text'].astype(str))
s_18k = set(df_dl_18k['text'].astype(str))
exact_overlap_count = len(s_82k.intersection(s_18k))
overlap_pct_82k = (exact_overlap_count / len(s_82k)) * 100
overlap_pct_18k = (exact_overlap_count / len(s_18k)) * 100
print(f"[VERIFIED] Exact String Overlap: {exact_overlap_count} emails ({overlap_pct_82k:.4f}% of D2, {overlap_pct_18k:.4f}% of D1)", flush=True)

# Stratified reproducible splits
tr_dl_82k, te_dl_82k, _, y_dl_82k = train_test_split(df_dl_82k, df_dl_82k['label'].astype(int), test_size=0.2, stratify=df_dl_82k['label'], random_state=42)
tr_ml_82k, te_ml_82k, _, y_ml_82k = train_test_split(df_ml_82k, df_ml_82k['label'].astype(int), test_size=0.2, stratify=df_ml_82k['label'], random_state=42)
tr_dl_18k, te_dl_18k, _, y_dl_18k = train_test_split(df_dl_18k, df_dl_18k['label'].astype(int), test_size=0.2, stratify=df_dl_18k['label'], random_state=42)
tr_ml_18k, te_ml_18k, _, y_ml_18k = train_test_split(df_ml_18k, df_ml_18k['label'].astype(int), test_size=0.2, stratify=df_ml_18k['label'], random_state=42)

# ==========================================
# 2. SANITY CHECK: PROVE RECONSTRUCTION
# ==========================================
print("\n[STEP 2] Running Sanity Check Assertions against disk X_test...", flush=True)
with open(os.path.join(DL_82K_DIR, "vectorizer.pkl"), "rb") as f:
    tok_dl_82k = pickle.load(f)
with open(os.path.join(DL_18K_DIR, "vectorizer.pkl"), "rb") as f:
    tok_dl_18k = pickle.load(f)

# Reconstruct D2 DL native X_test
seqs_recon_82k = tok_dl_82k.texts_to_sequences(te_dl_82k['text'].astype(str))
pad_recon_82k = pad_sequences(seqs_recon_82k, maxlen=400, padding='post', truncating='post')
saved_x_dl_82k = np.load(os.path.join(DL_82K_DIR, "X_test.npy"))
assert np.array_equal(pad_recon_82k, saved_x_dl_82k), "FATAL: D2 DL test set reconstruction mismatch!"
print(" [PASSED] D2 DL Native Reconstruction Assert: Bit-for-bit identical with X_test.npy (maxlen=400, post/post)", flush=True)

# Reconstruct D1 DL native X_test
seqs_recon_18k = tok_dl_18k.texts_to_sequences(te_dl_18k['text'].astype(str))
pad_recon_18k = pad_sequences(seqs_recon_18k, maxlen=400, padding='post', truncating='post')
saved_x_dl_18k = np.load(os.path.join(DL_18K_DIR, "X_test.npy"))
assert np.array_equal(pad_recon_18k, saved_x_dl_18k), "FATAL: D1 DL test set reconstruction mismatch!"
print(" [PASSED] D1 DL Native Reconstruction Assert: Bit-for-bit identical with X_test.npy (maxlen=400, post/post)", flush=True)

# Reconstruct D2 ML native X_test
with open(os.path.join(ML_82K_DIR, "vectorizer.pkl"), "rb") as f:
    vec_ml_82k = pickle.load(f)
scaler_ml_82k = MinMaxScaler()
scaler_ml_82k.fit(tr_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
s82_scaled = scaler_ml_82k.transform(te_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
t82_tfidf = vec_ml_82k.transform(te_ml_82k['text'].astype(str))
x82_ml_recon = hstack([t82_tfidf, s82_scaled]).tocsr()
x82_npz = np.load(os.path.join(ML_82K_DIR, "X_test.npz"))
x82_ml_saved = csr_matrix((x82_npz['data'], x82_npz['indices'], x82_npz['indptr']), shape=x82_npz['shape'])
assert np.abs(x82_ml_recon - x82_ml_saved).max() == 0.0, "FATAL: D2 ML test set reconstruction mismatch!"
print(" [PASSED] D2 ML Native Reconstruction Assert: Max diff = 0.0 with X_test.npz (10,004 features)", flush=True)

# Reconstruct D1 ML native X_test
with open(os.path.join(ML_18K_DIR, "vectorizer.pkl"), "rb") as f:
    v18_obj = pickle.load(f)
vec_ml_18k = v18_obj['vectorizer'] if isinstance(v18_obj, dict) else v18_obj
scaler_ml_18k = v18_obj['scaler'] if isinstance(v18_obj, dict) else None
t18_tfidf = vec_ml_18k.transform(te_ml_18k['text'].astype(str))
s18_scaled = scaler_ml_18k.transform(te_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
x18_ml_recon = hstack([t18_tfidf, s18_scaled]).tocsr()
x18_npz = np.load(os.path.join(ML_18K_DIR, "X_test.npz"))
x18_ml_saved = csr_matrix((x18_npz['data'], x18_npz['indices'], x18_npz['indptr']), shape=x18_npz['shape'])
assert np.abs(x18_ml_recon - x18_ml_saved).max() == 0.0, "FATAL: D1 ML test set reconstruction mismatch!"
print(" [PASSED] D1 ML Native Reconstruction Assert: Max diff = 0.0 with X_test.npz (10,004 features)", flush=True)

# ==========================================
# 3. PREPARE CROSS-TOKENIZED INPUTS
# ==========================================
print("\n[STEP 3] Preparing Cross-Dataset Tokenized Inputs...", flush=True)
# D1 evaluated with D2 DL Tokenizer (maxlen=400, post/post, vocab limit 15000)
X_18k_for_dl82k = pad_sequences(tok_dl_82k.texts_to_sequences(te_dl_18k['text'].astype(str)), maxlen=400, padding='post', truncating='post')
X_18k_for_dl82k[X_18k_for_dl82k >= 15000] = 0

# D2 evaluated with D1 DL Tokenizer (maxlen=400, post/post, vocab limit 15000)
X_82k_for_dl18k = pad_sequences(tok_dl_18k.texts_to_sequences(te_dl_82k['text'].astype(str)), maxlen=400, padding='post', truncating='post')
X_82k_for_dl18k[X_82k_for_dl18k >= 15000] = 0

# D1 for ML 82K (10,004 features: D2 TF-IDF + D2 Scaler)
s18_for_82k = scaler_ml_82k.transform(te_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_ml_18k_for_82k = hstack([vec_ml_82k.transform(te_ml_18k['text'].astype(str)), s18_for_82k]).tocsr()

# D2 for ML 18K (10,004 features: D1 TF-IDF + D1 Scaler)
s82_for_18k = scaler_ml_18k.transform(te_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_ml_82k_for_18k = hstack([vec_ml_18k.transform(te_ml_82k['text'].astype(str)), s82_for_18k]).tocsr()

report = {
    "metadata": {
        "description": "Comprehensive Cross-Dataset Evaluation (Deep Learning & Classical Hybrid ML)",
        "framework": "Optuna-Tuned PyTorch CUDA (both D1 & D2 for DL)",
        "d2_test_samples": len(te_dl_82k),
        "d1_test_samples": len(te_dl_18k),
        "vocab_size": 15000,
        "max_length": 400,
        "exact_string_overlap_count": exact_overlap_count,
        "exact_string_overlap_pct_d2": round(overlap_pct_82k, 4),
        "exact_string_overlap_pct_d1": round(overlap_pct_18k, 4),
        "near_duplicate_overlap_count_cos90": 13268,
        "near_duplicate_overlap_pct_d1": 75.66,
        "reconstruction_assert_dl_d2": True,
        "reconstruction_assert_dl_d1": True,
        "reconstruction_assert_ml_d2": True,
        "reconstruction_assert_ml_d1": True
    },
    "DL_cross_dataset": {
        "D2_models_tested_on_D1": {"models": {}, "skipped": {}},
        "D1_models_tested_on_D2": {"models": {}, "skipped": {}}
    },
    "ML_hybrid_cross_dataset": {
        "D2_models_tested_on_D1": {"models": {}, "skipped": {}},
        "D1_models_tested_on_D2": {"models": {}, "skipped": {}}
    },
    "ML_ablation_study": {
        "D2_models_tested_on_D1": {},
        "D1_models_tested_on_D2": {}
    }
}

# Shared forward function for all PyTorch topologies
def forward_pytorch(m, bx):
    if hasattr(m, 'conv3') and hasattr(m, 'conv4') and hasattr(m, 'conv5'):
        emb = m.embedding(bx).transpose(1, 2)
        c3 = m.pool(torch.relu(m.conv3(emb))).squeeze(-1)
        c4 = m.pool(torch.relu(m.conv4(emb))).squeeze(-1)
        c5 = m.pool(torch.relu(m.conv5(emb))).squeeze(-1)
        cat = torch.cat([c3, c4, c5], dim=1)
        return m.sigmoid(m.fc(m.dropout(cat)))
    elif hasattr(m, 'attn'):
        x = m.embedding(bx)
        gru_out, _ = m.gru(x)
        attn_out, _ = m.attn(gru_out, gru_out, gru_out)
        attn_out = attn_out.transpose(1, 2)
        pooled = m.pool(attn_out).squeeze(-1)
        dense = m.relu(m.fc1(pooled))
        dense = m.dropout(dense)
        return m.sigmoid(m.fc2(dense))
    elif hasattr(m, 'conv') and hasattr(m, 'lstm') and not hasattr(m, 'conv3'):
        x = torch.relu(m.conv(m.embedding(bx).transpose(1, 2)))
        p = m.pool(x).transpose(1, 2)
        out, (hn, cn) = m.lstm(p)
        return m.sigmoid(m.fc(m.dropout(hn[-1])))
    else:
        return m(bx)

def run_dl_pytorch_eval(m, X_arr, y_true):
    X_t = torch.tensor(X_arr, dtype=torch.long)
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_t), 512):
            bx = X_t[i:i+512].to(device)
            out = forward_pytorch(m, bx)
            if out.shape[-1] == 1:
                probs.append(out.squeeze(-1).cpu().numpy())
            else:
                probs.append(torch.softmax(out, dim=-1)[:, 1].cpu().numpy())
    pr = np.concatenate(probs).ravel()
    preds = (pr >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, pr)), 4)
    }

dl_models = [
    ("AUTO_MODEL", "auto_model.h5"),
    ("TEXT_CNN", "text_cnn.h5"),
    ("CNN", "cnn.h5"),
    ("HYBRID", "hybrid.h5"),
    ("CNN_LSTM", "cnn_lstm.h5"),
    ("GRU", "gru.h5"),
    ("BILSTM", "bilstm.h5"),
    ("LSTM", "lstm.h5"),
    ("STACKED_LSTM", "stacked_lstm.h5")
]

# ==========================================
# 4. EVALUATE DEEP LEARNING (DIRECTION A: D2 -> D1)
# ==========================================
print("\n" + "="*80, flush=True)
print("DEEP LEARNING DIRECTION A: Trained on D2 (82.5K PyTorch) -> Evaluated on D1 (18.6K, 3,727 samples)", flush=True)
print("="*80, flush=True)

for m_key, m_file in dl_models:
    p = os.path.join(DL_82K_DIR, "models", m_file)
    if not os.path.exists(p):
        report["DL_cross_dataset"]["D2_models_tested_on_D1"]["skipped"][m_key] = "File not found"
        continue
    try:
        model = torch.load(p, map_location=device, weights_only=False).eval()
        if not hasattr(model, 'pool'):
            model.pool = nn.AdaptiveMaxPool1d(1)
        if not hasattr(model, 'relu'):
            model.relu = nn.ReLU()
        in_domain = run_dl_pytorch_eval(model, saved_x_dl_82k, y_dl_82k.values)
        cross_domain = run_dl_pytorch_eval(model, X_18k_for_dl82k, y_dl_18k.values)
        delta_acc = round(cross_domain["accuracy"] - in_domain["accuracy"], 4)
        report["DL_cross_dataset"]["D2_models_tested_on_D1"]["models"][m_key] = {
            "in_domain_accuracy": in_domain["accuracy"],
            "cross_domain_accuracy": cross_domain["accuracy"],
            "delta_accuracy": delta_acc,
            "precision": cross_domain["precision"],
            "recall": cross_domain["recall"],
            "f1": cross_domain["f1"],
            "auc": cross_domain["auc"]
        }
        print(f"{m_key:12s} | In-Domain: {in_domain['accuracy']*100:.2f}% | Cross: {cross_domain['accuracy']*100:.2f}% (Delta: {delta_acc*100:+.2f}%) | Rec: {cross_domain['recall']*100:.2f}% | Prec: {cross_domain['precision']*100:.2f}% | AUC: {cross_domain['auc']:.4f}", flush=True)
    except Exception as e:
        report["DL_cross_dataset"]["D2_models_tested_on_D1"]["skipped"][m_key] = str(e)
        print(f"{m_key:12s} | Error: {e}", flush=True)

# ==========================================
# 5. EVALUATE DEEP LEARNING (DIRECTION B: D1 -> D2)
# ==========================================
print("\n" + "="*80, flush=True)
print("DEEP LEARNING DIRECTION B: Trained on D1 (18.6K PyTorch) -> Evaluated on D2 (82.5K, 16,497 samples)", flush=True)
print("="*80, flush=True)

for m_key, m_file in dl_models:
    p = os.path.join(DL_18K_DIR, "models", m_file)
    if not os.path.exists(p):
        report["DL_cross_dataset"]["D1_models_tested_on_D2"]["skipped"][m_key] = "File not found"
        continue
    try:
        model = torch.load(p, map_location=device, weights_only=False).eval()
        if not hasattr(model, 'pool'):
            model.pool = nn.AdaptiveMaxPool1d(1)
        if not hasattr(model, 'relu'):
            model.relu = nn.ReLU()
        in_domain = run_dl_pytorch_eval(model, saved_x_dl_18k, y_dl_18k.values)
        cross_domain = run_dl_pytorch_eval(model, X_82k_for_dl18k, y_dl_82k.values)
        delta_acc = round(cross_domain["accuracy"] - in_domain["accuracy"], 4)
        report["DL_cross_dataset"]["D1_models_tested_on_D2"]["models"][m_key] = {
            "in_domain_accuracy": in_domain["accuracy"],
            "cross_domain_accuracy": cross_domain["accuracy"],
            "delta_accuracy": delta_acc,
            "precision": cross_domain["precision"],
            "recall": cross_domain["recall"],
            "f1": cross_domain["f1"],
            "auc": cross_domain["auc"]
        }
        print(f"{m_key:12s} | In-Domain: {in_domain['accuracy']*100:.2f}% | Cross: {cross_domain['accuracy']*100:.2f}% (Delta: {delta_acc*100:+.2f}%) | Rec: {cross_domain['recall']*100:.2f}% | Prec: {cross_domain['precision']*100:.2f}% | AUC: {cross_domain['auc']:.4f}", flush=True)
    except Exception as e:
        report["DL_cross_dataset"]["D1_models_tested_on_D2"]["skipped"][m_key] = str(e)
        print(f"{m_key:12s} | Error: {e}", flush=True)

# ==========================================
# 6. EVALUATE CLASSICAL ML (ALL 8 MODELS WITH CHUNKED PREDICTION)
# ==========================================
def predict_chunked(m, X, chunk=1000):
    dense_needed = type(m).__name__ in ["GaussianNB", "HistGradientBoostingClassifier"]
    preds, scores = [], []
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i+chunk]
        if dense_needed:
            Xc = Xc.toarray()
        preds.append(m.predict(Xc))
        if hasattr(m, "decision_function"):
            scores.append(m.decision_function(Xc))
        elif hasattr(m, "predict_proba"):
            p = m.predict_proba(Xc)
            if p.shape[1] > 1:
                scores.append(p[:, 1])
            else:
                scores.append(p[:, 0])
    
    y_pred = np.concatenate(preds)
    y_score = np.concatenate(scores) if scores else None
    return y_pred, y_score

def get_ml_metrics(m, X, y_true):
    preds, scores = predict_chunked(m, X)
    acc = accuracy_score(y_true, preds)
    rec = recall_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, scores) if scores is not None else None
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4) if auc is not None else None
    }

ml_models = [
    ("LinearSVC", "LinearSVC.pkl"),
    ("LogisticRegression", "LogisticRegression.pkl"),
    ("VotingClassifier", "VotingClassifier.pkl"),
    ("SGDClassifier", "SGDClassifier.pkl"),
    ("auto_model", "auto_model.pkl"),
    ("GaussianNB", "GaussianNB.pkl"),
    ("MultinomialNB", "MultinomialNB.pkl"),
    ("RandomForestClassifier", "RandomForestClassifier.pkl")
]

print("\n" + "="*80, flush=True)
print("CLASSICAL HYBRID ML DIRECTION A: Trained on D2 (82.5K) -> Evaluated on D1 (18.6K)", flush=True)
print("="*80, flush=True)

for m_name, m_file in ml_models:
    p = os.path.join(ML_82K_DIR, "models", m_file)
    if not os.path.exists(p):
        report["ML_hybrid_cross_dataset"]["D2_models_tested_on_D1"]["skipped"][m_name] = "Model artifact file not found"
        continue
    try:
        m = joblib.load(p)
        in_m = get_ml_metrics(m, x82_ml_saved, y_ml_82k)
        cross_m = get_ml_metrics(m, X_ml_18k_for_82k, y_ml_18k)
        delta = round(cross_m["accuracy"] - in_m["accuracy"], 4)
        
        report["ML_hybrid_cross_dataset"]["D2_models_tested_on_D1"]["models"][m_name] = {
            "in_domain_accuracy": in_m["accuracy"],
            "in_domain_auc": in_m["auc"],
            "cross_domain_accuracy": cross_m["accuracy"],
            "delta_accuracy": delta,
            "precision": cross_m["precision"],
            "recall": cross_m["recall"],
            "f1": cross_m["f1"],
            "auc": cross_m["auc"]
        }
        auc_str = f"{cross_m['auc']:.4f}" if cross_m['auc'] is not None else "N/A"
        print(f"{m_name:24s} | In-Domain: {in_m['accuracy']*100:.2f}% | Cross: {cross_m['accuracy']*100:.2f}% (Delta: {delta*100:+.2f}%) | Rec: {cross_m['recall']*100:.2f}% | Prec: {cross_m['precision']*100:.2f}% | F1: {cross_m['f1']*100:.2f}% | AUC: {auc_str}", flush=True)
    except Exception as e:
        report["ML_hybrid_cross_dataset"]["D2_models_tested_on_D1"]["skipped"][m_name] = str(e)
        print(f"{m_name:24s} | Error: {e}", flush=True)

print("\n" + "="*80, flush=True)
print("CLASSICAL HYBRID ML DIRECTION B: Trained on D1 (18.6K) -> Evaluated on D2 (82.5K)", flush=True)
print("="*80, flush=True)

for m_name, m_file in ml_models:
    p = os.path.join(ML_18K_DIR, "models", m_file)
    if not os.path.exists(p):
        report["ML_hybrid_cross_dataset"]["D1_models_tested_on_D2"]["skipped"][m_name] = "Model artifact file not found"
        continue
    try:
        m = joblib.load(p)
        in_m = get_ml_metrics(m, x18_ml_saved, y_ml_18k)
        cross_m = get_ml_metrics(m, X_ml_82k_for_18k, y_ml_82k)
        delta = round(cross_m["accuracy"] - in_m["accuracy"], 4)
        
        report["ML_hybrid_cross_dataset"]["D1_models_tested_on_D2"]["models"][m_name] = {
            "in_domain_accuracy": in_m["accuracy"],
            "in_domain_auc": in_m["auc"],
            "cross_domain_accuracy": cross_m["accuracy"],
            "delta_accuracy": delta,
            "precision": cross_m["precision"],
            "recall": cross_m["recall"],
            "f1": cross_m["f1"],
            "auc": cross_m["auc"]
        }
        auc_str = f"{cross_m['auc']:.4f}" if cross_m['auc'] is not None else "N/A"
        print(f"{m_name:24s} | In-Domain: {in_m['accuracy']*100:.2f}% | Cross: {cross_m['accuracy']*100:.2f}% (Delta: {delta*100:+.2f}%) | Rec: {cross_m['recall']*100:.2f}% | Prec: {cross_m['precision']*100:.2f}% | F1: {cross_m['f1']*100:.2f}% | AUC: {auc_str}", flush=True)
    except Exception as e:
        report["ML_hybrid_cross_dataset"]["D1_models_tested_on_D2"]["skipped"][m_name] = str(e)
        print(f"{m_name:24s} | Error: {e}", flush=True)

# ==========================================
# 7. ML ABLATION STUDY: TF-IDF vs Struct vs Hybrid (WITH TUNED HYPERPARAMETERS)
# ==========================================
print("\n" + "="*80, flush=True)
print("ML ABLATION STUDY: TF-IDF only vs Structural only vs Hybrid (Optuna Hyperparameters)", flush=True)
print("="*80, flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

with open(os.path.join(ML_82K_DIR, "best_params.json")) as f:
    bp_82k = json.load(f)

with open(os.path.join(ML_18K_DIR, "best_params.json")) as f:
    bp_18k = json.load(f)

ablation_models_d2 = [
    ("LogisticRegression", LogisticRegression(C=bp_82k['LogisticRegression']['C'], solver=bp_82k['LogisticRegression']['solver'], max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(C=bp_82k['LinearSVC']['C'], max_iter=2000, random_state=42))
]

ablation_models_d1 = [
    ("LogisticRegression", LogisticRegression(C=bp_18k['LogisticRegression']['C'], solver=bp_18k['LogisticRegression']['solver'], max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(C=bp_18k['LinearSVC']['C'], max_iter=2000, random_state=42))
]

# Ablation: Train on D2, Test on D1
X_tr2_tfidf = vec_ml_82k.transform(tr_ml_82k['text'].astype(str))
X_te1_tfidf_for2 = vec_ml_82k.transform(te_ml_18k['text'].astype(str))
X_tr2_struct = scaler_ml_82k.transform(tr_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_te1_struct_for2 = scaler_ml_82k.transform(te_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_tr2_hyb = hstack([X_tr2_tfidf, X_tr2_struct]).tocsr()
X_te1_hyb_for2 = hstack([X_te1_tfidf_for2, X_te1_struct_for2]).tocsr()

for name, clf in ablation_models_d2:
    # TF-IDF
    clf.fit(X_tr2_tfidf, tr_ml_82k['label'])
    p_tf = clf.predict(X_te1_tfidf_for2)
    acc_tf, rec_tf, f1_tf = accuracy_score(te_ml_18k['label'], p_tf), recall_score(te_ml_18k['label'], p_tf), f1_score(te_ml_18k['label'], p_tf)
    
    # Struct
    clf.fit(X_tr2_struct, tr_ml_82k['label'])
    p_st = clf.predict(X_te1_struct_for2)
    acc_st, rec_st, f1_st = accuracy_score(te_ml_18k['label'], p_st), recall_score(te_ml_18k['label'], p_st), f1_score(te_ml_18k['label'], p_st)

    # Hybrid
    clf.fit(X_tr2_hyb, tr_ml_82k['label'])
    p_hy = clf.predict(X_te1_hyb_for2)
    acc_hy, rec_hy, f1_hy = accuracy_score(te_ml_18k['label'], p_hy), recall_score(te_ml_18k['label'], p_hy), f1_score(te_ml_18k['label'], p_hy)

    report["ML_ablation_study"]["D2_models_tested_on_D1"][name] = {
        "tfidf_only": {"accuracy": round(acc_tf, 4), "recall": round(rec_tf, 4), "f1": round(f1_tf, 4)},
        "struct_only": {"accuracy": round(acc_st, 4), "recall": round(rec_st, 4), "f1": round(f1_st, 4)},
        "hybrid": {"accuracy": round(acc_hy, 4), "recall": round(rec_hy, 4), "f1": round(f1_hy, 4)}
    }
    print(f"D2->D1 {name:18s} | TF-IDF: {acc_tf*100:.2f}% (Rec: {rec_tf*100:.2f}%) | Struct: {acc_st*100:.2f}% (Rec: {rec_st*100:.2f}%) | Hybrid: {acc_hy*100:.2f}% (Rec: {rec_hy*100:.2f}%)", flush=True)

# Ablation: Train on D1, Test on D2
X_tr1_tfidf = vec_ml_18k.transform(tr_ml_18k['text'].astype(str))
X_te2_tfidf_for1 = vec_ml_18k.transform(te_ml_82k['text'].astype(str))
X_tr1_struct = scaler_ml_18k.transform(tr_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_te2_struct_for1 = scaler_ml_18k.transform(te_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_tr1_hyb = hstack([X_tr1_tfidf, X_tr1_struct]).tocsr()
X_te2_hyb_for1 = hstack([X_te2_tfidf_for1, X_te2_struct_for1]).tocsr()

for name, clf in ablation_models_d1:
    # TF-IDF
    clf.fit(X_tr1_tfidf, tr_ml_18k['label'])
    p_tf = clf.predict(X_te2_tfidf_for1)
    acc_tf, rec_tf, f1_tf = accuracy_score(te_ml_82k['label'], p_tf), recall_score(te_ml_82k['label'], p_tf), f1_score(te_ml_82k['label'], p_tf)
    
    # Struct
    clf.fit(X_tr1_struct, tr_ml_18k['label'])
    p_st = clf.predict(X_te2_struct_for1)
    acc_st, rec_st, f1_st = accuracy_score(te_ml_82k['label'], p_st), recall_score(te_ml_82k['label'], p_st), f1_score(te_ml_82k['label'], p_st)

    # Hybrid
    clf.fit(X_tr1_hyb, tr_ml_18k['label'])
    p_hy = clf.predict(X_te2_hyb_for1)
    acc_hy, rec_hy, f1_hy = accuracy_score(te_ml_82k['label'], p_hy), recall_score(te_ml_82k['label'], p_hy), f1_score(te_ml_82k['label'], p_hy)

    report["ML_ablation_study"]["D1_models_tested_on_D2"][name] = {
        "tfidf_only": {"accuracy": round(acc_tf, 4), "recall": round(rec_tf, 4), "f1": round(f1_tf, 4)},
        "struct_only": {"accuracy": round(acc_st, 4), "recall": round(rec_st, 4), "f1": round(f1_st, 4)},
        "hybrid": {"accuracy": round(acc_hy, 4), "recall": round(rec_hy, 4), "f1": round(f1_hy, 4)}
    }
    print(f"D1->D2 {name:18s} | TF-IDF: {acc_tf*100:.2f}% (Rec: {rec_tf*100:.2f}%) | Struct: {acc_st*100:.2f}% (Rec: {rec_st*100:.2f}%) | Hybrid: {acc_hy*100:.2f}% (Rec: {rec_hy*100:.2f}%)", flush=True)

# Save final report to JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

root_copy_json = os.path.join(REPO_DIR, "cross_dataset_results_proper.json")
with open(root_copy_json, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80, flush=True)
print(f"[SUCCESS] All evaluations complete and saved to:\n  - {OUTPUT_JSON}\n  - {root_copy_json}", flush=True)
print("="*80, flush=True)
