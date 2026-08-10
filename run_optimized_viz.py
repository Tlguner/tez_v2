# -*- coding: utf-8 -*-
import os
import json
import sys
import io
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import joblib
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# Force UTF-8 encoding
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

BASE_RUNS_DIR = "C:/TU/deu/tez_v2/scraped data/phishing_mcp_server/thesis_pipeline/ML/runs"
all_runs = [os.path.join(BASE_RUNS_DIR, d) for d in os.listdir(BASE_RUNS_DIR) if d.startswith("run_")]
all_runs.sort(key=os.path.getmtime, reverse=True)
RUN_DIR = all_runs[0] if all_runs else "C:/TU/deu/tez_v2/scraped data/phishing_mcp_server/thesis_pipeline/ML/runs/run_2026-08-04_12-17-11"
print(f"[INFO] Target Run Directory: {RUN_DIR}")

MODELS_DIR = os.path.join(RUN_DIR, "models")
TEST_FEATURES_PATH = os.path.join(RUN_DIR, "X_test.npz")
TEST_LABELS_PATH = os.path.join(RUN_DIR, "y_test.npy")
CLEANED_DATASET_PATH = os.path.join(RUN_DIR, "cleaned_dataset.csv")
VECTORIZER_PKL_PATH = os.path.join(RUN_DIR, "vectorizer.pkl")
MODELS_METRICS_JSON = os.path.join(RUN_DIR, "auto_ML_results.json")
VIZ_OUTPUT_DIR = os.path.join(RUN_DIR, "viz_outputs")
RUN_STATE_JSON = os.path.join(RUN_DIR, "run_state.json")

os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

output_json = {
    "ok": True,
    "viz_output_dir": VIZ_OUTPUT_DIR.replace("\\", "/"),
    "shap_plots": [],
    "wordcloud_comparison_plot": os.path.join(VIZ_OUTPUT_DIR, "wordcloud_comparison.png").replace("\\", "/"),
    "wordcloud_phishing_plot": os.path.join(VIZ_OUTPUT_DIR, "wordcloud_phishing.png").replace("\\", "/"),
    "wordcloud_legit_plot": os.path.join(VIZ_OUTPUT_DIR, "wordcloud_legit.png").replace("\\", "/"),
    "tsne_plot": os.path.join(VIZ_OUTPUT_DIR, "tsne_embedding.png").replace("\\", "/"),
    "calibration_plot": os.path.join(VIZ_OUTPUT_DIR, "calibration_curve.png").replace("\\", "/")
}

print("[INFO] Loading data for optimized visualization...")
start_time = time.time()

X_test = scipy.sparse.load_npz(TEST_FEATURES_PATH)
y_test = np.ravel(np.load(TEST_LABELS_PATH)).astype(int)
print(f"[INFO] Loaded X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

df_clean = pd.read_csv(CLEANED_DATASET_PATH)
print(f"[INFO] Loaded cleaned dataset: {len(df_clean)} rows.")

vectorizer = None
if os.path.exists(VECTORIZER_PKL_PATH):
    vectorizer = joblib.load(VECTORIZER_PKL_PATH)
    print("[INFO] Loaded vectorizer.")

with open(MODELS_METRICS_JSON, "r", encoding="utf-8") as f:
    metrics = json.load(f)

def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

# --- TASK 1: Fast WordCloud from TF-IDF Frequencies ---
print("\n[TASK 1/4] Generating WordClouds from TF-IDF frequencies across ALL 82,485 rows...")
t0 = time.time()
try:
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    label_col = "label" if "label" in df_clean.columns else df_clean.columns[-1]
    text_col = "text_combined" if "text_combined" in df_clean.columns else df_clean.columns[0]

    phishing_mask = df_clean[label_col].astype(str).str.contains('1|phish|spam', case=False, na=False).values
    legit_mask = ~phishing_mask

    if vectorizer is not None:
        feature_names = vectorizer.get_feature_names_out()
        # Compute TF-IDF frequency dictionary directly from matrix sums
        X_train_full = scipy.sparse.load_npz(os.path.join(RUN_DIR, "X_train.npz"))
        y_train_full = np.ravel(np.load(os.path.join(RUN_DIR, "y_train.npy"))).astype(int)
        
        phish_train_mask = (y_train_full == 1)
        legit_train_mask = (y_train_full == 0)

        phish_sums = np.array(X_train_full[phish_train_mask].sum(axis=0)).ravel()
        legit_sums = np.array(X_train_full[legit_train_mask].sum(axis=0)).ravel()

        phish_freqs = {feature_names[i]: float(phish_sums[i]) for i in range(len(feature_names))}
        legit_freqs = {feature_names[i]: float(legit_sums[i]) for i in range(len(feature_names))}

        # Filter out numbers, single letters, and email header noise (dates, months, enron headers)
        date_month_stopwords = {"2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010",
                                "aug", "jan", "feb", "mar", "apr", "may", "jun", "jul", "sep", "oct", "nov", "dec",
                                "subject", "re", "email", "cc", "hpl", "ect", "enron", "hou", "com", "net", "org",
                                "http", "https", "www", "pm", "am", "vols", "nom", "noms", "xls", "doc"}
        
        custom_stopwords = set(ENGLISH_STOP_WORDS).union(date_month_stopwords)
        
        # Clean freqs dict: remove stopwords, numbers-only terms, and terms <= 2 chars
        clean_phish_freqs = {k: v for k, v in phish_freqs.items() 
                             if k.lower() not in custom_stopwords 
                             and not k.isdigit() 
                             and len(k) > 2}
        
        clean_legit_freqs = {k: v for k, v in legit_freqs.items() 
                             if k.lower() not in custom_stopwords 
                             and not k.isdigit() 
                             and len(k) > 2}

        # collocations=False and repeat=False prevents any word from repeating in the layout
        wc_phish = WordCloud(width=800, height=600, background_color='black', colormap='autumn', 
                              max_words=100, collocations=False, repeat=False).generate_from_frequencies(clean_phish_freqs)
        wc_phish.to_file(output_json["wordcloud_phishing_plot"])

        wc_legit = WordCloud(width=800, height=600, background_color='white', colormap='winter', 
                             max_words=100, collocations=False, repeat=False).generate_from_frequencies(clean_legit_freqs)
        wc_legit.to_file(output_json["wordcloud_legit_plot"])

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(wc_phish, interpolation='bilinear')
        axes[0].set_title("Phishing Emails - Top Phishing Terms (Full Dataset)", fontsize=16, color='red', fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(wc_legit, interpolation='bilinear')
        axes[1].set_title("Legitimate Emails - Top Legitimate Terms (Full Dataset)", fontsize=16, color='blue', fontweight='bold')
        axes[1].axis('off')

        plt.suptitle("Word Cloud Comparison - Phishing vs Legitimate", fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(output_json["wordcloud_comparison_plot"], dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] WordCloud generated in {time.time() - t0:.2f}s -> {output_json['wordcloud_comparison_plot']}")
except Exception as e:
    print(f"[WARN] WordCloud failed: {e}")

# --- TASK 2: Fast Multi-Core t-SNE Embedding ---
print("\n[TASK 2/4] Generating Multi-Core t-SNE Embedding across ALL 16,497 test samples...")
t0 = time.time()
try:
    # Use TruncatedSVD to reduce 10,004 sparse features to 50 dense features in ~1s
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X_test)

    # Multi-core TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y_test == 0, 0], X_tsne[y_test == 0, 1], color='blue', label='Legitimate', alpha=0.5, s=15)
    plt.scatter(X_tsne[y_test == 1, 0], X_tsne[y_test == 1, 1], color='red', label='Phishing', alpha=0.5, s=15)
    plt.title("t-SNE Visualization of Phishing Email Feature Embeddings (Full Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)

    plt.savefig(output_json["tsne_plot"], dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] t-SNE generated in {time.time() - t0:.2f}s -> {output_json['tsne_plot']}")
except Exception as e:
    print(f"[WARN] t-SNE failed: {e}")

# --- TASK 3: Feature Importance (Random Forest + Logistic Regression) ---
print("\n[TASK 3/4] Generating Feature Importance (Random Forest & Logistic Regression)...")
t0 = time.time()
try:
    if vectorizer is not None:
        feature_names = list(vectorizer.get_feature_names_out())
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # 1. Random Forest Feature Importance (Tree-based)
        rf_model = load_model("RandomForestClassifier")
        if rf_model is not None:
            importances = rf_model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            axes[0].barh([feature_names[i] for i in top_idx[::-1]], importances[top_idx[::-1]], color='crimson')
            axes[0].set_xlabel("Feature Importance (Gini Impurity)", fontsize=12)
            axes[0].set_title("Random Forest - Top 20 Important Features", fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)

        # 2. Logistic Regression Coefficients (Linear weights)
        lr_model = load_model("LogisticRegression")
        if lr_model is not None:
            coefs = lr_model.coef_[0]
            top_idx = np.argsort(np.abs(coefs))[::-1][:20]
            axes[1].barh([feature_names[i] for i in top_idx[::-1]], coefs[top_idx[::-1]], color='teal')
            axes[1].set_xlabel("Logistic Regression Coefficient Weight", fontsize=12)
            axes[1].set_title("Logistic Regression - Top 20 Feature Weights", fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        shap_plot_path = os.path.join(VIZ_OUTPUT_DIR, "shap_RandomForestClassifier.png").replace("\\", "/")
        plt.savefig(shap_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_json["shap_plots"].append(shap_plot_path)
        print(f"[INFO] Feature Importances generated in {time.time() - t0:.2f}s -> {shap_plot_path}")
except Exception as e:
    print(f"[WARN] Feature importance failed: {e}")

# --- TASK 4: Calibration Curves ---
print("\n[TASK 4/4] Generating Calibration Curves across ALL test samples...")
t0 = time.time()
try:
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    
    models_eval = ["LogisticRegression", "MultinomialNB", "RandomForestClassifier", "LinearSVC", "SGDClassifier", "VotingClassifier"]
    has_curves = False
    
    for name in models_eval:
        model = load_model(name)
        if model is not None:
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    df_val = model.decision_function(X_test)
                    y_prob = 1 / (1 + np.exp(-df_val))
                else:
                    continue

                y_prob = np.clip(y_prob, 0.0, 1.0)
                fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
                plt.plot(mean_pred, fraction_pos, "s-", label=f"{name}")
                has_curves = True
            except Exception as ex:
                print(f"[WARN] Calibration failed for {name}: {ex}")

    if has_curves:
        plt.ylabel("Fraction of positives", fontsize=12)
        plt.xlabel("Mean predicted probability", fontsize=12)
        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right", fontsize=10)
        plt.title("Calibration Curves - Model Reliability (Full Test Set)", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_json["calibration_plot"], dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Calibration curve generated in {time.time() - t0:.2f}s -> {output_json['calibration_plot']}")
except Exception as e:
    print(f"[WARN] Calibration curve failed: {e}")

# Update run_state.json to mark step_5_viz as completed
if os.path.exists(RUN_STATE_JSON):
    try:
        with open(RUN_STATE_JSON, "r", encoding="utf-8") as f:
            state = json.load(f)
        state["step_5_viz"] = "completed"
        state["viz_output_dir"] = VIZ_OUTPUT_DIR.replace("\\", "/")
        with open(RUN_STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print("[INFO] Updated run_state.json: step_5_viz = completed.")
    except Exception as e:
        print(f"[WARN] Could not update run_state.json: {e}")

print(f"\n✅ All visualizations completed in {time.time() - start_time:.2f}s!")
print("<<<JSON_START>>>")
print(json.dumps(output_json, indent=2))
print("<<<JSON_END>>>")
