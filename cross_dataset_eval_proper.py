# -*- coding: utf-8 -*-
"""
Comprehensive Cross-Dataset Generalization & Domain Shift Evaluation Pipeline
Evaluates 9 Deep Learning and 8 Classical Hybrid Machine Learning models
under zero-shot domain transfer between Dataset 1 (18.6K) and Dataset 2 (82.5K).

Includes:
- Dynamic lexical similarity & near-duplicate audit (Source Train vs. Target Test)
- Deduplicated intra-dataset leakage analysis
- Native in-domain structural-only baselines
- Full feature ablation study (TF-IDF vs. Struct vs. Hybrid) with ROC-AUC
- Stratified subset evaluation (Near-Duplicates vs. Truly Novel samples)
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
ROOT_COPY_JSON = os.path.join(REPO_DIR, "cross_dataset_results_proper.json")

# ==========================================
# 1. LOAD DATASETS & REPRODUCIBLE SPLITS
# ==========================================
print("\n[STEP 1] Loading Cleaned Datasets & Generating Stratified Splits...", flush=True)
df_dl_82k = pd.read_csv(os.path.join(DL_82K_DIR, "cleaned_dataset.csv"))
df_ml_82k = pd.read_csv(os.path.join(ML_82K_DIR, "cleaned_dataset.csv"))
df_dl_18k = pd.read_csv(os.path.join(DL_18K_DIR, "cleaned_dataset.csv"))
df_ml_18k = pd.read_csv(os.path.join(ML_18K_DIR, "cleaned_dataset.csv"))

# DL Stratified splits
tr_dl_82k, te_dl_82k, _, y_dl_82k = train_test_split(df_dl_82k, df_dl_82k['label'].astype(int), test_size=0.2, stratify=df_dl_82k['label'], random_state=42)
tr_dl_18k, te_dl_18k, _, y_dl_18k = train_test_split(df_dl_18k, df_dl_18k['label'].astype(int), test_size=0.2, stratify=df_dl_18k['label'], random_state=42)

# ML Stratified splits
tr_ml_82k, te_ml_82k, _, y_ml_82k = train_test_split(df_ml_82k, df_ml_82k['label'].astype(int), test_size=0.2, stratify=df_ml_82k['label'], random_state=42)
tr_ml_18k, te_ml_18k, _, y_ml_18k = train_test_split(df_ml_18k, df_ml_18k['label'].astype(int), test_size=0.2, stratify=df_ml_18k['label'], random_state=42)

# Global exact overlap
s_82k_all = set(df_dl_82k['text'].astype(str))
s_18k_all = set(df_dl_18k['text'].astype(str))
exact_overlap_count = len(s_82k_all.intersection(s_18k_all))
overlap_pct_82k = (exact_overlap_count / len(s_82k_all)) * 100
overlap_pct_18k = (exact_overlap_count / len(s_18k_all)) * 100
print(f"[VERIFIED] Global Exact-String Overlap: {exact_overlap_count} emails ({overlap_pct_82k:.4f}% of D2, {overlap_pct_18k:.4f}% of D1)", flush=True)

# Intra-dataset train-test leakage check
s_tr_18k = set(tr_dl_18k['text'].astype(str))
clean_te1_mask_dl = (~te_dl_18k['text'].astype(str).isin(s_tr_18k)).values
clean_te1_mask_ml = (~te_ml_18k['text'].astype(str).isin(set(tr_ml_18k['text'].astype(str)))).values
d1_intra_leak_count = int(np.sum(~clean_te1_mask_dl))
d1_intra_leak_pct = round(float(d1_intra_leak_count / len(te_dl_18k) * 100), 2)
print(f"[AUDIT] D1 Intra-Dataset Train-Test Leakage: {d1_intra_leak_count} / {len(te_dl_18k)} ({d1_intra_leak_pct}%)", flush=True)

s_tr_82k = set(tr_dl_82k['text'].astype(str))
d2_intra_leak_count = int(np.sum(te_dl_82k['text'].astype(str).isin(s_tr_82k)))
d2_intra_leak_pct = round(float(d2_intra_leak_count / len(te_dl_82k) * 100), 2)
print(f"[AUDIT] D2 Intra-Dataset Train-Test Leakage: {d2_intra_leak_count} / {len(te_dl_82k)} ({d2_intra_leak_pct}%)", flush=True)

# Conflicting labels audit
d1_grouped = df_dl_18k.groupby('text')['label'].nunique()
d1_conflicts = d1_grouped[d1_grouped > 1]
d1_conflict_rows = int(df_dl_18k['text'].isin(d1_conflicts.index).sum())

d2_grouped = df_dl_82k.groupby('text')['label'].nunique()
d2_conflicts = d2_grouped[d2_grouped > 1]
d2_conflict_rows = int(df_dl_82k['text'].isin(d2_conflicts.index).sum())

leaked_te_d1 = te_dl_18k[te_dl_18k['text'].astype(str).isin(s_tr_18k)]
conflicting_leaked_count = 0
for _, row in leaked_te_d1.iterrows():
    txt = str(row['text'])
    te_label = row['label']
    tr_labels = tr_dl_18k[tr_dl_18k['text'].astype(str) == txt]['label'].unique()
    if te_label not in tr_labels or len(tr_labels) > 1:
        conflicting_leaked_count += 1
conflicting_leaked_pct = round(float(conflicting_leaked_count / len(leaked_te_d1) * 100), 2) if len(leaked_te_d1) > 0 else 0.0

print(f"[AUDIT] D1 Conflicting Labels: {len(d1_conflicts)} unique texts ({d1_conflict_rows} total rows). Leaked test conflicting: {conflicting_leaked_count}/{len(leaked_te_d1)} ({conflicting_leaked_pct}%)", flush=True)
print(f"[AUDIT] D2 Conflicting Labels: {len(d2_conflicts)} unique texts ({d2_conflict_rows} total rows)", flush=True)

# ==========================================
# 2. DYNAMIC LEXICAL SIMILARITY (SOURCE TRAIN VS TARGET TEST)
# ==========================================
print("\n[STEP 2] Computing Dynamic Lexical Similarity (Source Train vs. Target Test)...", flush=True)
sim_vec = TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1, 2))
chunk_size = 500

# Direction A: Target Test = D1 Test (te_dl_18k) vs Source Train = D2 Train (tr_dl_82k)
texts_a = list(te_dl_18k['text'].astype(str)) + list(tr_dl_82k['text'].astype(str))
X_a = sim_vec.fit_transform(texts_a)
X_te1 = X_a[:len(te_dl_18k)]
X_tr2 = X_a[len(te_dl_18k):]

max_sim_te1 = np.zeros(len(te_dl_18k))
for i in range(0, X_te1.shape[0], chunk_size):
    sim = linear_kernel(X_te1[i:i+chunk_size], X_tr2)
    max_sim_te1[i:i+chunk_size] = np.max(sim, axis=1)

mask_te1_near80 = (max_sim_te1 >= 0.80)
mask_te1_near90 = (max_sim_te1 >= 0.90)
mask_te1_novel50 = (max_sim_te1 < 0.50)

dir_a_sim_stats = {
    "target_test_samples": len(te_dl_18k),
    "source_train_samples": len(tr_dl_82k),
    "exact_matches": int(te_dl_18k['text'].astype(str).isin(s_tr_82k).sum()),
    "near_duplicates_cos90_count": int(np.sum(mask_te1_near90)),
    "near_duplicates_cos90_pct": round(float(np.mean(mask_te1_near90) * 100), 2),
    "near_duplicates_cos80_count": int(np.sum(mask_te1_near80)),
    "near_duplicates_cos80_pct": round(float(np.mean(mask_te1_near80) * 100), 2),
    "moderate_similarity_cos50_count": int(np.sum(max_sim_te1 >= 0.50)),
    "moderate_similarity_cos50_pct": round(float(np.mean(max_sim_te1 >= 0.50) * 100), 2),
    "truly_novel_cos_lt50_count": int(np.sum(mask_te1_novel50)),
    "truly_novel_cos_lt50_pct": round(float(np.mean(mask_te1_novel50) * 100), 2)
}
print(f" [Direction A: D2 Train -> D1 Test] Near-dup (>=0.80): {dir_a_sim_stats['near_duplicates_cos80_count']} ({dir_a_sim_stats['near_duplicates_cos80_pct']}%) | Truly novel (<0.50): {dir_a_sim_stats['truly_novel_cos_lt50_count']} ({dir_a_sim_stats['truly_novel_cos_lt50_pct']}%)", flush=True)

# Direction B (DL): Target Test = D2 Test (te_dl_82k, N=16497) vs Source Train = D1 Train (tr_dl_18k, N=14907)
texts_b_dl = list(te_dl_82k['text'].astype(str)) + list(tr_dl_18k['text'].astype(str))
X_b_dl = sim_vec.fit_transform(texts_b_dl)
X_te2_dl = X_b_dl[:len(te_dl_82k)]
X_tr1_dl = X_b_dl[len(te_dl_82k):]

max_sim_te2_dl = np.zeros(len(te_dl_82k))
for i in range(0, X_te2_dl.shape[0], chunk_size):
    sim = linear_kernel(X_te2_dl[i:i+chunk_size], X_tr1_dl)
    max_sim_te2_dl[i:i+chunk_size] = np.max(sim, axis=1)

mask_te2_near80 = (max_sim_te2_dl >= 0.80)
mask_te2_near90 = (max_sim_te2_dl >= 0.90)
mask_te2_novel50 = (max_sim_te2_dl < 0.50)

dir_b_sim_stats = {
    "target_test_samples": len(te_dl_82k),
    "source_train_samples": len(tr_dl_18k),
    "exact_matches": int(te_dl_82k['text'].astype(str).isin(s_tr_18k).sum()),
    "near_duplicates_cos90_count": int(np.sum(mask_te2_near90)),
    "near_duplicates_cos90_pct": round(float(np.mean(mask_te2_near90) * 100), 2),
    "near_duplicates_cos80_count": int(np.sum(mask_te2_near80)),
    "near_duplicates_cos80_pct": round(float(np.mean(mask_te2_near80) * 100), 2),
    "moderate_similarity_cos50_count": int(np.sum(max_sim_te2_dl >= 0.50)),
    "moderate_similarity_cos50_pct": round(float(np.mean(max_sim_te2_dl >= 0.50) * 100), 2),
    "truly_novel_cos_lt50_count": int(np.sum(mask_te2_novel50)),
    "truly_novel_cos_lt50_pct": round(float(np.mean(mask_te2_novel50) * 100), 2)
}
print(f" [Direction B: D1 Train -> D2 Test (DL)] Near-dup (>=0.80): {dir_b_sim_stats['near_duplicates_cos80_count']} ({dir_b_sim_stats['near_duplicates_cos80_pct']}%) | Truly novel (<0.50): {dir_b_sim_stats['truly_novel_cos_lt50_count']} ({dir_b_sim_stats['truly_novel_cos_lt50_pct']}%)", flush=True)

# Direction B (ML): Target Test = D2 Test (te_ml_82k, N=16498) vs Source Train = D1 Train (tr_ml_18k, N=14907)
# Computed directly on ML split without any np.append hack
texts_b_ml = list(te_ml_82k['text'].astype(str)) + list(tr_ml_18k['text'].astype(str))
X_b_ml = sim_vec.fit_transform(texts_b_ml)
X_te2_ml = X_b_ml[:len(te_ml_82k)]
X_tr1_ml = X_b_ml[len(te_ml_82k):]

max_sim_te2_ml = np.zeros(len(te_ml_82k))
for i in range(0, X_te2_ml.shape[0], chunk_size):
    sim = linear_kernel(X_te2_ml[i:i+chunk_size], X_tr1_ml)
    max_sim_te2_ml[i:i+chunk_size] = np.max(sim, axis=1)

mask_ml_te2_near80 = (max_sim_te2_ml >= 0.80)
mask_ml_te2_novel50 = (max_sim_te2_ml < 0.50)
assert len(mask_ml_te2_near80) == len(te_ml_82k) == 16498, "FATAL: ML Direction B mask length mismatch!"
assert len(mask_te2_near80) == len(te_dl_82k) == 16497, "FATAL: DL Direction B mask length mismatch!"
assert len(mask_te1_near80) == len(te_dl_18k) == 3727, "FATAL: D1 mask length mismatch!"

# ==========================================
# 3. RECONSTRUCTION ASSERTIONS AGAINST DISK
# ==========================================
print("\n[STEP 3] Running Bit-for-Bit Sanity Check Assertions against Disk...", flush=True)
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
# 4. PREPARE CROSS-TOKENIZED INPUTS
# ==========================================
print("\n[STEP 4] Preparing Cross-Dataset Tokenized Inputs...", flush=True)
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
        "d1_total_samples": len(df_dl_18k),
        "d1_phishing_pct": round(float(df_dl_18k['label'].mean() * 100), 2),
        "d1_test_samples": len(te_dl_18k),
        "d2_total_samples": len(df_dl_82k),
        "d2_phishing_pct": round(float(df_dl_82k['label'].mean() * 100), 2),
        "d2_test_samples": len(te_dl_82k),
        "exact_string_overlap_global_count": exact_overlap_count,
        "exact_string_overlap_global_pct_d2": round(overlap_pct_82k, 4),
        "exact_string_overlap_global_pct_d1": round(overlap_pct_18k, 4),
        "d1_intra_dataset_leak_count": d1_intra_leak_count,
        "d1_intra_dataset_leak_pct": d1_intra_leak_pct,
        "d2_intra_dataset_leak_count": d2_intra_leak_count,
        "d2_intra_dataset_leak_pct": d2_intra_leak_pct,
        "similarity_direction_A_D2train_vs_D1test": dir_a_sim_stats,
        "similarity_direction_B_D1train_vs_D2test": dir_b_sim_stats,
        "conflicting_labels_audit": {
            "d1_unique_conflicting_texts": len(d1_conflicts),
            "d1_conflicting_rows_total": d1_conflict_rows,
            "d1_leaked_test_conflicting_count": conflicting_leaked_count,
            "d1_leaked_test_total": len(leaked_te_d1),
            "d1_leaked_test_conflicting_pct": conflicting_leaked_pct,
            "d2_unique_conflicting_texts": len(d2_conflicts),
            "d2_conflicting_rows_total": d2_conflict_rows
        },
        "empty_rows_audit": {
            "d1_empty_text_rows": int((df_dl_18k['text'] == 'empty').sum()),
            "d1_empty_text_phishing": int(((df_dl_18k['text'] == 'empty') & (df_dl_18k['label'] == 1)).sum()),
            "d1_empty_text_legitimate": int(((df_dl_18k['text'] == 'empty') & (df_dl_18k['label'] == 0)).sum()),
            "d1_empty_text_in_test": int((te_dl_18k['text'] == 'empty').sum()),
            "d1_empty_text_in_train": int((tr_dl_18k['text'] == 'empty').sum()),
            "d2_empty_text_rows": int((df_dl_82k['text'] == 'empty').sum())
        },
        "evaluation_sample_sizes": {
            "d1_test_samples": len(te_dl_18k),
            "d2_test_samples_dl": len(te_dl_82k),
            "d2_test_samples_ml": len(te_ml_82k),
            "note": "ML test split contains 16,498 samples (N=82,486 total) and DL test split contains 16,497 samples (N=82,485 total). Both pipelines use independent stratified splits and independent similarity masks."
        },
        "reconstruction_assert_dl_d2": True,
        "reconstruction_assert_dl_d1": True,
        "reconstruction_assert_ml_d2": True,
        "reconstruction_assert_ml_d1": True
    },
    "d1_deduplicated_in_domain_baseline": {},
    "native_structural_in_domain": {},
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
    },
    "subset_evaluation": {
        "D2_models_tested_on_D1": {},
        "D1_models_tested_on_D2": {}
    }
}

# Shared PyTorch Forward Function
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
    auc_val = roc_auc_score(y_true, pr) if len(np.unique(y_true)) > 1 else None
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "auc": round(float(auc_val), 4) if auc_val is not None else None
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
# 5. EVALUATE DEEP LEARNING (DIRECTION A: D2 -> D1)
# ==========================================
print("\n" + "="*80, flush=True)
print("DEEP LEARNING DIRECTION A: Trained on D2 (82.5K PyTorch) -> Evaluated on D1 (18.6K, 3,727 samples)", flush=True)
print("="*80, flush=True)

for m_key, m_file in dl_models:
    p = os.path.join(DL_82K_DIR, "models", m_file)
    if not os.path.exists(p):
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
        
        # Subset eval
        cross_near80 = run_dl_pytorch_eval(model, X_18k_for_dl82k[mask_te1_near80], y_dl_18k.values[mask_te1_near80])
        cross_novel50 = run_dl_pytorch_eval(model, X_18k_for_dl82k[mask_te1_novel50], y_dl_18k.values[mask_te1_novel50])
        
        report["DL_cross_dataset"]["D2_models_tested_on_D1"]["models"][m_key] = {
            "in_domain_accuracy": in_domain["accuracy"],
            "cross_domain_accuracy": cross_domain["accuracy"],
            "delta_accuracy": delta_acc,
            "precision": cross_domain["precision"],
            "recall": cross_domain["recall"],
            "f1": cross_domain["f1"],
            "auc": cross_domain["auc"]
        }
        report["subset_evaluation"]["D2_models_tested_on_D1"][m_key] = {
            "all": cross_domain,
            "near_duplicates_ge_080": cross_near80,
            "truly_novel_lt_050": cross_novel50
        }
        print(f"{m_key:12s} | In: {in_domain['accuracy']*100:.2f}% | Cross: {cross_domain['accuracy']*100:.2f}% (d: {delta_acc*100:+.2f}%) | Near: {cross_near80['accuracy']*100:.2f}% | Novel: {cross_novel50['accuracy']*100:.2f}% | AUC: {cross_domain['auc']:.4f}", flush=True)
    except Exception as e:
        report["DL_cross_dataset"]["D2_models_tested_on_D1"]["skipped"][m_key] = str(e)
        print(f"{m_key:12s} | Error: {e}", flush=True)

# ==========================================
# 6. EVALUATE DEEP LEARNING (DIRECTION B: D1 -> D2)
# ==========================================
print("\n" + "="*80, flush=True)
print("DEEP LEARNING DIRECTION B: Trained on D1 (18.6K PyTorch) -> Evaluated on D2 (82.5K, 16,497 samples)", flush=True)
print("="*80, flush=True)

for m_key, m_file in dl_models:
    p = os.path.join(DL_18K_DIR, "models", m_file)
    if not os.path.exists(p):
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
        
        # Subset eval
        cross_near80 = run_dl_pytorch_eval(model, X_82k_for_dl18k[mask_te2_near80], y_dl_82k.values[mask_te2_near80])
        cross_novel50 = run_dl_pytorch_eval(model, X_82k_for_dl18k[mask_te2_novel50], y_dl_82k.values[mask_te2_novel50])
        
        report["DL_cross_dataset"]["D1_models_tested_on_D2"]["models"][m_key] = {
            "in_domain_accuracy": in_domain["accuracy"],
            "cross_domain_accuracy": cross_domain["accuracy"],
            "delta_accuracy": delta_acc,
            "precision": cross_domain["precision"],
            "recall": cross_domain["recall"],
            "f1": cross_domain["f1"],
            "auc": cross_domain["auc"]
        }
        report["subset_evaluation"]["D1_models_tested_on_D2"][m_key] = {
            "all": cross_domain,
            "near_duplicates_ge_080": cross_near80,
            "truly_novel_lt_050": cross_novel50
        }
        print(f"{m_key:12s} | In: {in_domain['accuracy']*100:.2f}% | Cross: {cross_domain['accuracy']*100:.2f}% (d: {delta_acc*100:+.2f}%) | Near: {cross_near80['accuracy']*100:.2f}% | Novel: {cross_novel50['accuracy']*100:.2f}% | AUC: {cross_domain['auc']:.4f}", flush=True)
    except Exception as e:
        report["DL_cross_dataset"]["D1_models_tested_on_D2"]["skipped"][m_key] = str(e)
        print(f"{m_key:12s} | Error: {e}", flush=True)

# ==========================================
# 7. EVALUATE CLASSICAL ML (ALL 8 MODELS WITH CHUNKED PREDICTION)
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
    auc_val = roc_auc_score(y_true, scores) if scores is not None and len(np.unique(y_true)) > 1 else None
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc_val), 4) if auc_val is not None else None
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

# D1 Deduplicated In-Domain Evaluation
print("\n[STEP 7A] Evaluating D1 In-Domain on Deduplicated Clean Test Set...", flush=True)
for m_name, m_file in ml_models:
    p = os.path.join(ML_18K_DIR, "models", m_file)
    if os.path.exists(p):
        m = joblib.load(p)
        full_m = get_ml_metrics(m, x18_ml_saved, y_ml_18k)
        clean_m = get_ml_metrics(m, x18_ml_saved[clean_te1_mask_ml], y_ml_18k.values[clean_te1_mask_ml])
        report["d1_deduplicated_in_domain_baseline"][m_name] = {
            "full_test_accuracy": full_m["accuracy"],
            "clean_test_accuracy": clean_m["accuracy"],
            "difference_pp": round(float((clean_m["accuracy"] - full_m["accuracy"]) * 100), 2)
        }
        print(f"D1 Native {m_name:20s} | Full: {full_m['accuracy']*100:.2f}% | Clean: {clean_m['accuracy']*100:.2f}% (Diff: {(clean_m['accuracy'] - full_m['accuracy'])*100:+.2f} pp)", flush=True)

# Direction A: D2 -> D1 ML
print("\n" + "="*80, flush=True)
print("CLASSICAL HYBRID ML DIRECTION A: Trained on D2 (82.5K) -> Evaluated on D1 (18.6K)", flush=True)
print("="*80, flush=True)

for m_name, m_file in ml_models:
    p = os.path.join(ML_82K_DIR, "models", m_file)
    if not os.path.exists(p):
        continue
    try:
        m = joblib.load(p)
        in_m = get_ml_metrics(m, x82_ml_saved, y_ml_82k)
        cross_m = get_ml_metrics(m, X_ml_18k_for_82k, y_ml_18k)
        delta = round(cross_m["accuracy"] - in_m["accuracy"], 4)
        
        # Subset eval
        near_m = get_ml_metrics(m, X_ml_18k_for_82k[mask_te1_near80], y_ml_18k.values[mask_te1_near80])
        novel_m = get_ml_metrics(m, X_ml_18k_for_82k[mask_te1_novel50], y_ml_18k.values[mask_te1_novel50])
        
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
        report["subset_evaluation"]["D2_models_tested_on_D1"][m_name] = {
            "all": cross_m,
            "near_duplicates_ge_080": near_m,
            "truly_novel_lt_050": novel_m
        }
        print(f"{m_name:24s} | In: {in_m['accuracy']*100:.2f}% | Cross: {cross_m['accuracy']*100:.2f}% (d: {delta*100:+.2f}%) | Near: {near_m['accuracy']*100:.2f}% | Novel: {novel_m['accuracy']*100:.2f}% | AUC: {cross_m['auc']:.4f}", flush=True)
    except Exception as e:
        report["ML_hybrid_cross_dataset"]["D2_models_tested_on_D1"]["skipped"][m_name] = str(e)
        print(f"{m_name:24s} | Error: {e}", flush=True)

# Direction B: D1 -> D2 ML
print("\n" + "="*80, flush=True)
print("CLASSICAL HYBRID ML DIRECTION B: Trained on D1 (18.6K) -> Evaluated on D2 (82.5K)", flush=True)
print("="*80, flush=True)

for m_name, m_file in ml_models:
    p = os.path.join(ML_18K_DIR, "models", m_file)
    if not os.path.exists(p):
        continue
    try:
        m = joblib.load(p)
        in_m = get_ml_metrics(m, x18_ml_saved, y_ml_18k)
        cross_m = get_ml_metrics(m, X_ml_82k_for_18k, y_ml_82k)
        delta = round(cross_m["accuracy"] - in_m["accuracy"], 4)
        
        # Subset eval
        near_m = get_ml_metrics(m, X_ml_82k_for_18k[mask_ml_te2_near80], y_ml_82k.values[mask_ml_te2_near80])
        novel_m = get_ml_metrics(m, X_ml_82k_for_18k[mask_ml_te2_novel50], y_ml_82k.values[mask_ml_te2_novel50])
        
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
        report["subset_evaluation"]["D1_models_tested_on_D2"][m_name] = {
            "all": cross_m,
            "near_duplicates_ge_080": near_m,
            "truly_novel_lt_050": novel_m
        }
        print(f"{m_name:24s} | In: {in_m['accuracy']*100:.2f}% | Cross: {cross_m['accuracy']*100:.2f}% (d: {delta*100:+.2f}%) | Near: {near_m['accuracy']*100:.2f}% | Novel: {novel_m['accuracy']*100:.2f}% | AUC: {cross_m['auc']:.4f}", flush=True)
    except Exception as e:
        report["ML_hybrid_cross_dataset"]["D1_models_tested_on_D2"]["skipped"][m_name] = str(e)
        print(f"{m_name:24s} | Error: {e}", flush=True)

# ==========================================
# 8. NATIVE IN-DOMAIN STRUCTURAL-ONLY BASELINES
# ==========================================
print("\n" + "="*80, flush=True)
print("NATIVE IN-DOMAIN STRUCTURAL-ONLY BASELINES (D1 & D2)", flush=True)
print("="*80, flush=True)

with open(os.path.join(ML_82K_DIR, "best_params.json")) as f:
    bp_82k = json.load(f)
with open(os.path.join(ML_18K_DIR, "best_params.json")) as f:
    bp_18k = json.load(f)

# Native Structural D1
X_tr1_struct_native = scaler_ml_18k.transform(tr_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_te1_struct_native = scaler_ml_18k.transform(te_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)

for name, clf in [
    ("LogisticRegression", LogisticRegression(C=bp_18k['LogisticRegression']['C'], solver=bp_18k['LogisticRegression']['solver'], max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(C=bp_18k['LinearSVC']['C'], max_iter=2000, random_state=42))
]:
    clf.fit(X_tr1_struct_native, tr_ml_18k['label'])
    m_res = get_ml_metrics(clf, X_te1_struct_native, te_ml_18k['label'])
    report["native_structural_in_domain"][f"D1_{name}"] = m_res
    print(f"D1 Native Structural {name:18s} | Acc: {m_res['accuracy']*100:.2f}% | Rec: {m_res['recall']*100:.2f}% | F1: {m_res['f1']*100:.2f}% | AUC: {m_res['auc']:.4f}", flush=True)

# Native Structural D2
X_tr2_struct_native = scaler_ml_82k.transform(tr_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_te2_struct_native = scaler_ml_82k.transform(te_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)

for name, clf in [
    ("LogisticRegression", LogisticRegression(C=bp_82k['LogisticRegression']['C'], solver=bp_82k['LogisticRegression']['solver'], max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(C=bp_82k['LinearSVC']['C'], max_iter=2000, random_state=42))
]:
    clf.fit(X_tr2_struct_native, tr_ml_82k['label'])
    m_res = get_ml_metrics(clf, X_te2_struct_native, te_ml_82k['label'])
    report["native_structural_in_domain"][f"D2_{name}"] = m_res
    print(f"D2 Native Structural {name:18s} | Acc: {m_res['accuracy']*100:.2f}% | Rec: {m_res['recall']*100:.2f}% | F1: {m_res['f1']*100:.2f}% | AUC: {m_res['auc']:.4f}", flush=True)

# ==========================================
# 9. ML ABLATION STUDY: TF-IDF vs Struct vs Hybrid (WITH TUNED HYPERPARAMETERS & AUC)
# ==========================================
print("\n" + "="*80, flush=True)
print("ML ABLATION STUDY: TF-IDF only vs Structural only vs Hybrid (WITH ROC-AUC)", flush=True)
print("="*80, flush=True)

ablation_models_d2 = [
    ("LogisticRegression", LogisticRegression(C=bp_82k['LogisticRegression']['C'], solver=bp_82k['LogisticRegression']['solver'], max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(C=bp_82k['LinearSVC']['C'], max_iter=2000, random_state=42))
]

ablation_models_d1 = [
    ("LogisticRegression", LogisticRegression(C=bp_18k['LogisticRegression']['C'], solver=bp_18k['LogisticRegression']['solver'], max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(C=bp_18k['LinearSVC']['C'], max_iter=2000, random_state=42))
]

# Ablation Direction A: Train on D2, Test on D1
X_tr2_tfidf = vec_ml_82k.transform(tr_ml_82k['text'].astype(str))
X_te1_tfidf_for2 = vec_ml_82k.transform(te_ml_18k['text'].astype(str))
X_tr2_struct = scaler_ml_82k.transform(tr_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_te1_struct_for2 = scaler_ml_82k.transform(te_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_tr2_hyb = hstack([X_tr2_tfidf, X_tr2_struct]).tocsr()
X_te1_hyb_for2 = hstack([X_te1_tfidf_for2, X_te1_struct_for2]).tocsr()

for name, clf in ablation_models_d2:
    # TF-IDF
    clf.fit(X_tr2_tfidf, tr_ml_82k['label'])
    m_tf = get_ml_metrics(clf, X_te1_tfidf_for2, te_ml_18k['label'])
    
    # Struct
    clf.fit(X_tr2_struct, tr_ml_82k['label'])
    m_st = get_ml_metrics(clf, X_te1_struct_for2, te_ml_18k['label'])

    # Hybrid
    clf.fit(X_tr2_hyb, tr_ml_82k['label'])
    m_hy = get_ml_metrics(clf, X_te1_hyb_for2, te_ml_18k['label'])

    report["ML_ablation_study"]["D2_models_tested_on_D1"][name] = {
        "tfidf_only": m_tf,
        "struct_only": m_st,
        "hybrid": m_hy
    }
    print(f"D2->D1 {name:18s} | TF-IDF: {m_tf['accuracy']*100:.2f}% (AUC {m_tf['auc']}) | Struct: {m_st['accuracy']*100:.2f}% (AUC {m_st['auc']}) | Hybrid: {m_hy['accuracy']*100:.2f}% (AUC {m_hy['auc']})", flush=True)

# Ablation Direction B: Train on D1, Test on D2
X_tr1_tfidf = vec_ml_18k.transform(tr_ml_18k['text'].astype(str))
X_te2_tfidf_for1 = vec_ml_18k.transform(te_ml_82k['text'].astype(str))
X_tr1_struct = scaler_ml_18k.transform(tr_ml_18k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_te2_struct_for1 = scaler_ml_18k.transform(te_ml_82k[['url_count', 'has_ip', 'urgency_score', 'has_html']].values)
X_tr1_hyb = hstack([X_tr1_tfidf, X_tr1_struct]).tocsr()
X_te2_hyb_for1 = hstack([X_te2_tfidf_for1, X_te2_struct_for1]).tocsr()

for name, clf in ablation_models_d1:
    # TF-IDF
    clf.fit(X_tr1_tfidf, tr_ml_18k['label'])
    m_tf = get_ml_metrics(clf, X_te2_tfidf_for1, te_ml_82k['label'])
    
    # Struct
    clf.fit(X_tr1_struct, tr_ml_18k['label'])
    m_st = get_ml_metrics(clf, X_te2_struct_for1, te_ml_82k['label'])

    # Hybrid
    clf.fit(X_tr1_hyb, tr_ml_18k['label'])
    m_hy = get_ml_metrics(clf, X_te2_hyb_for1, te_ml_82k['label'])

    report["ML_ablation_study"]["D1_models_tested_on_D2"][name] = {
        "tfidf_only": m_tf,
        "struct_only": m_st,
        "hybrid": m_hy
    }
    print(f"D1->D2 {name:18s} | TF-IDF: {m_tf['accuracy']*100:.2f}% (AUC {m_tf['auc']}) | Struct: {m_st['accuracy']*100:.2f}% (AUC {m_st['auc']}) | Hybrid: {m_hy['accuracy']*100:.2f}% (AUC {m_hy['auc']})", flush=True)

# ==========================================
# 10. SAVE FINAL REPORT TO DISK
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

with open(ROOT_COPY_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80, flush=True)
print(f"[SUCCESS] All dynamic evaluations complete and saved to:\n  - {OUTPUT_JSON}\n  - {ROOT_COPY_JSON}", flush=True)
print("="*80, flush=True)
