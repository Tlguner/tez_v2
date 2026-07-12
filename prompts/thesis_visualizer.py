# -*- coding: utf-8 -*-
import os
import json
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
import sys
import io

# Force UTF-8 encoding
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

# Parameters (filled dynamically by client.js)
MODE = "${mode}"
MODELS_DIR = "${models_dir}"
TEST_FEATURES_PATH = "${test_features_path}"
TEST_LABELS_PATH = "${test_labels_path}"
CLEANED_DATASET_PATH = "${cleaned_dataset_path}"
VECTORIZER_PKL_PATH = "${vectorizer_pkl_path}"
MODELS_METRICS_JSON = "${models_metrics_json}"
VIZ_OUTPUT_DIR = "${viz_output_dir}"

os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

# Loaded data variables
X_test = None
y_test = None
df_clean = None
vectorizer = None
metrics = {}

# Output files tracking
output_json = {
    "ok": True,
    "viz_output_dir": VIZ_OUTPUT_DIR,
    "shap_plots": [],
    "error_analysis_plots": {},
    "error_summary_csv": os.path.join(VIZ_OUTPUT_DIR, "error_analysis_summary.csv").replace("\\", "/"),
    "wordcloud_comparison_plot": os.path.join(VIZ_OUTPUT_DIR, "wordcloud_comparison.png").replace("\\", "/"),
    "wordcloud_phishing_plot": os.path.join(VIZ_OUTPUT_DIR, "wordcloud_phishing.png").replace("\\", "/"),
    "wordcloud_legit_plot": os.path.join(VIZ_OUTPUT_DIR, "wordcloud_legit.png").replace("\\", "/"),
    "tsne_plot": os.path.join(VIZ_OUTPUT_DIR, "tsne_embedding.png").replace("\\", "/"),
    "calibration_plot": os.path.join(VIZ_OUTPUT_DIR, "calibration_curve.png").replace("\\", "/")
}

# --- Load Data ---
print("[INFO] Loading data...")
try:
    if MODE == "ML":
        X_test = scipy.sparse.load_npz(TEST_FEATURES_PATH)
    else:
        X_test = np.load(TEST_FEATURES_PATH)
    
    y_test = np.ravel(np.load(TEST_LABELS_PATH)).astype(int)
    print(f"[INFO] Loaded X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
except Exception as e:
    print(f"[ERROR] Failed to load features/labels: {e}", file=sys.stderr)
    sys.exit(1)

try:
    df_clean = pd.read_csv(CLEANED_DATASET_PATH)
    print(f"[INFO] Loaded cleaned dataset. Rows: {len(df_clean)}")
except Exception as e:
    print(f"[WARN] Failed to load dataset: {e}", file=sys.stderr)

try:
    if os.path.exists(VECTORIZER_PKL_PATH):
        vectorizer = joblib.load(VECTORIZER_PKL_PATH)
        print("[INFO] Loaded vectorizer.")
except Exception as e:
    print(f"[WARN] Failed to load vectorizer: {e}", file=sys.stderr)

try:
    with open(MODELS_METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("[INFO] Loaded models metrics.")
except Exception as e:
    print(f"[WARN] Failed to load metrics JSON: {e}", file=sys.stderr)

# Define BahdanauAttention and other custom layers for DL models loading
if MODE == "DL":
    import tensorflow as tf
    class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units, **kwargs):
            super(BahdanauAttention, self).__init__(**kwargs)
            self.units = units
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            query_with_time_axis = tf.expand_dims(query, 1)
            score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector

        def get_config(self):
            config = super(BahdanauAttention, self).get_config()
            config.update({"units": self.units})
            return config

    class ExpandDims(tf.keras.layers.Layer):
        def __init__(self, axis, **kwargs):
            super().__init__(**kwargs)
            self.axis = axis
        def call(self, inputs, **kwargs):
            return tf.expand_dims(inputs, axis=self.axis)
        def get_config(self):
            config = super().get_config()
            config.update({"axis": self.axis})
            return config

    class Sum(tf.keras.layers.Layer):
        def __init__(self, axis, keepdims=False, **kwargs):
            super().__init__(**kwargs)
            self.axis = axis
            self.keepdims = keepdims
        def call(self, inputs, **kwargs):
            return tf.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)
        def get_config(self):
            config = super().get_config()
            config.update({"axis": self.axis, "keepdims": self.keepdims})
            return config

    class PatchedMultiply(tf.keras.layers.Multiply):
        def __call__(self, *args, **kwargs):
            if len(args) > 1:
                args = (list(args),)
            return super().__call__(*args, **kwargs)

# Helper to load models
def load_model_by_name(name):
    try:
        if MODE == "ML":
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            return joblib.load(path)
        else:
            path = os.path.join(MODELS_DIR, f"{name}.h5")
            if name == "auto_model":
                return tf.keras.models.load_model(path, custom_objects={
                    'BahdanauAttention': BahdanauAttention,
                    'ExpandDims': ExpandDims,
                    'Sum': Sum,
                    'Multiply': PatchedMultiply
                })
            else:
                return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"[WARN] Failed to load model {name}: {e}", file=sys.stderr)
        return None

# Helper to make predictions robustly
def predict_model(model, name, features):
    if MODE == "ML":
        if name in ["GaussianNB", "auto_model"]:
            # Convert to dense
            features_dense = features.toarray()
            preds = model.predict(features_dense)
        else:
            preds = model.predict(features)
        return preds
    else:
        # DL predicts probabilities
        probs = model.predict(features, verbose=0).ravel()
        return (probs >= 0.5).astype(int)

# Helper to get predict probabilities robustly
def predict_proba_model(model, name, features):
    if MODE == "ML":
        if name in ["GaussianNB", "auto_model"]:
            features_dense = features.toarray()
            if hasattr(model, "predict_proba"):
                return model.predict_proba(features_dense)[:, 1]
        else:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(features)[:, 1]
        
        # Fallback for LinearSVC or SGDClassifier with hinge
        if hasattr(model, "decision_function"):
            decision = model.decision_function(features)
            # Apply sigmoid
            return 1 / (1 + np.exp(-decision))
        return None
    else:
        probs = model.predict(features, verbose=0).ravel()
        return probs

# --- TASK 1: SHAP ---
print("[INFO] TASK 1: Starting SHAP feature importance...")
try:
    import shap
    if MODE == "ML":
        # Trees: RandomForestClassifier, auto_model (HistGradientBoosting)
        shap_models = ["RandomForestClassifier", "auto_model"]
        for s_name in shap_models:
            model = load_model_by_name(s_name)
            if model is not None:
                try:
                    # Select max 300 test samples to keep it fast
                    slice_size = min(300, X_test.shape[0])
                    X_sample = X_test[:slice_size]
                    X_sample_dense = X_sample.toarray()
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample_dense)
                    
                    # For binary classification, shap_values can be list of arrays or single array
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        # RF shap_values is a list for [class 0, class 1]
                        shap_values_to_plot = shap_values[1]
                    else:
                        shap_values_to_plot = shap_values
                    
                    feature_names = None
                    if vectorizer is not None:
                        feature_names = list(vectorizer.get_feature_names_out())
                        
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_to_plot, X_sample_dense, feature_names=feature_names, max_display=15, show=False)
                    plt.title(f"SHAP Feature Importance - {s_name}", fontsize=14)
                    
                    plot_path = os.path.join(VIZ_OUTPUT_DIR, f"shap_{s_name}.png").replace("\\", "/")
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    output_json["shap_plots"].append(plot_path)
                    print(f"[INFO] Saved SHAP plot for {s_name} to {plot_path}")
                except Exception as ex:
                    print(f"[WARN] SHAP error for model {s_name}: {ex}", file=sys.stderr)
    else:
        # DL SHAP
        # Find best model
        best_dl_name = None
        best_score = -1
        # Metrics can be list of dicts or dict
        metrics_dict = {}
        if isinstance(metrics, list):
            for item in metrics:
                metrics_dict[item["model_name"]] = item
        else:
            metrics_dict = metrics
            
        for k, v in metrics_dict.items():
            acc = v.get("accuracy", 0)
            if acc > best_score and k != "auto_model": # Choose best baseline DL model
                best_score = acc
                best_dl_name = k
        if not best_dl_name:
            best_dl_name = "bilstm"
            
        model = load_model_by_name(best_dl_name)
        if model is not None:
            try:
                # DL shap is slow: use 50 background samples and explain 100 samples
                bg_size = min(50, X_test.shape[0])
                test_size = min(100, X_test.shape[0])
                
                background = X_test[:bg_size]
                X_sample = X_test[:test_size]
                
                try:
                    explainer = shap.DeepExplainer(model, background)
                    shap_values = explainer.shap_values(X_sample)
                except Exception:
                    try:
                        # Fallback to GradientExplainer
                        explainer = shap.GradientExplainer(model, background)
                        shap_values = explainer.shap_values(X_sample)
                    except Exception:
                        # Fallback to KernelExplainer (extremely robust model-agnostic fallback)
                        print("[INFO] Falling back to SHAP KernelExplainer...")
                        bg_kernel = X_test[:10]
                        sample_kernel = X_test[:15]
                        explainer = shap.KernelExplainer(lambda x: model.predict(x, verbose=0), bg_kernel)
                        shap_values = explainer.shap_values(sample_kernel, nsamples=50)
                
                # shap_values shape: (100, 200, 1) or (100, 200)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Calculate mean absolute importance per token position
                mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2)) if len(shap_values.shape) == 3 else np.mean(np.abs(shap_values), axis=0)
                
                plt.figure(figsize=(12, 5))
                plt.bar(range(len(mean_abs_shap)), mean_abs_shap, color='crimson')
                plt.title(f"Mean Absolute SHAP Value per Token Position ({best_dl_name})", fontsize=14)
                plt.xlabel("Token Position (0-199)", fontsize=12)
                plt.ylabel("Mean |SHAP Value|", fontsize=12)
                plt.grid(True, alpha=0.3)
                
                plot_path = os.path.join(VIZ_OUTPUT_DIR, "shap_dl_best_model.png").replace("\\", "/")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                output_json["shap_plots"].append(plot_path)
                print(f"[INFO] Saved DL SHAP plot to {plot_path}")
            except Exception as ex:
                print(f"[WARN] DL SHAP explanation failed: {ex}", file=sys.stderr)
except Exception as e:
    print(f"[WARN] SHAP execution skipped or failed: {e}", file=sys.stderr)


# --- TASK 2: Error Analysis ---
print("[INFO] TASK 2: Error Analysis skipped by user request.")


# --- TASK 3: Word Cloud ---
print("[INFO] TASK 3: Starting Word Cloud...")
try:
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    if df_clean is not None:
        # Detect text/label columns again
        text_col = None
        for col in ["text_combined", "Email Text", "text", "body", "message", "content", "email", "mail"]:
            if col in df_clean.columns:
                text_col = col
                break
        label_col = None
        for col in ["label", "Email Type", "labels", "target", "Category", "class", "y"]:
            if col in df_clean.columns:
                label_col = col
                break
                
        # Split texts
        phishing_df = df_clean[df_clean[label_col].astype(str).str.contains('1|phish|spam|malicious', case=False, na=False)]
        legit_df = df_clean[~df_clean.index.isin(phishing_df.index)]
        
        phishing_text = " ".join(phishing_df[text_col].fillna("").astype(str).values)
        legit_text = " ".join(legit_df[text_col].fillna("").astype(str).values)
        
        # Define stopwords
        custom_stopwords = set(ENGLISH_STOP_WORDS).union({"subject", "re", "email", "cc", "hpl", "ect", "enron", "hou", "com"})
        
        # WordCloud for Phishing
        wc_phish = WordCloud(width=800, height=600, background_color='black', colormap='autumn', stopwords=custom_stopwords, max_words=100).generate(phishing_text)
        wc_phish.to_file(output_json["wordcloud_phishing_plot"])
        
        # WordCloud for Legit
        wc_legit = WordCloud(width=800, height=600, background_color='white', colormap='winter', stopwords=custom_stopwords, max_words=100).generate(legit_text)
        wc_legit.to_file(output_json["wordcloud_legit_plot"])
        
        # Comparison Side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(wc_phish, interpolation='bilinear')
        axes[0].set_title("Phishing Emails - Top Words", fontsize=16, color='red', fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(wc_legit, interpolation='bilinear')
        axes[1].set_title("Legitimate Emails - Top Words", fontsize=16, color='blue', fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle("Word Cloud Comparison - Phishing vs Legitimate", fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(output_json["wordcloud_comparison_plot"], dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved Word Cloud comparison to {output_json['wordcloud_comparison_plot']}")
except Exception as e:
    print(f"[WARN] Word Cloud task failed: {e}", file=sys.stderr)


# --- TASK 4: t-SNE Embedding ---
print("[INFO] TASK 4: Starting t-SNE embedding visualization...")
try:
    # We sample 1000 items
    sample_size = min(1000, len(y_test))
    np.random.seed(42)
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    y_sample = y_test[indices]
    
    if MODE == "ML":
        X_sample = X_test[indices]
        # Dimensions reduction first
        svd = TruncatedSVD(n_components=min(50, X_sample.shape[1] - 1), random_state=42)
        X_reduced = svd.fit_transform(X_sample)
    else:
        # DL mode: load best model and extract embeddings from second-to-last layer
        X_sample = X_test[indices]
        best_dl_name = None
        best_score = -1
        
        metrics_dict = {}
        if isinstance(metrics, list):
            for item in metrics:
                metrics_dict[item["model_name"]] = item
        else:
            metrics_dict = metrics
            
        for k, v in metrics_dict.items():
            acc = v.get("accuracy", 0)
            if acc > best_score and k != "auto_model":
                best_score = acc
                best_dl_name = k
        if not best_dl_name:
            best_dl_name = "bilstm"
            
        model = load_model_by_name(best_dl_name)
        if model is not None:
            try:
                # Build feature extractor depending on model type
                if isinstance(model, tf.keras.Sequential):
                    extractor = tf.keras.Sequential(model.layers[:-1])
                else:
                    extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
                X_reduced = extractor.predict(X_sample, verbose=0)
                # Flatten in case of 3D LSTM outputs
                if len(X_reduced.shape) == 3:
                    X_reduced = X_reduced.mean(axis=1)
            except Exception as ex:
                print(f"[WARN] DL Embedding extraction failed: {ex}. Falling back to default t-SNE...", file=sys.stderr)
                X_reduced = X_sample.reshape(X_sample.shape[0], -1)
        else:
            X_reduced = X_sample.reshape(X_sample.shape[0], -1)
            
    # Apply TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_reduced)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y_sample == 0, 0], X_tsne[y_sample == 0, 1], color='blue', label='Legitimate', alpha=0.6, s=25)
    plt.scatter(X_tsne[y_sample == 1, 0], X_tsne[y_sample == 1, 1], color='red', label='Phishing', alpha=0.6, s=25)
    plt.title(f"t-SNE Visualization of Email Embeddings ({MODE} Mode)", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_json["tsne_plot"], dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved t-SNE visualization to {output_json['tsne_plot']}")
except Exception as e:
    print(f"[WARN] t-SNE task failed: {e}", file=sys.stderr)


# --- TASK 5: Calibration Curve ---
print("[INFO] TASK 5: Starting Calibration Curve...")
try:
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    
    models_to_eval = []
    if MODE == "ML":
        models_to_eval = ["LogisticRegression", "MultinomialNB", "GaussianNB", "RandomForestClassifier", "LinearSVC", "SGDClassifier", "VotingClassifier", "auto_model"]
    else:
        models_to_eval = ["lstm", "bilstm", "gru", "cnn", "hybrid", "text_cnn", "stacked_lstm", "cnn_lstm", "auto_model"]
        
    has_curves = False
    
    for name in models_to_eval:
        model = load_model_by_name(name)
        if model is not None:
            try:
                y_prob = predict_proba_model(model, name, X_test)
                if y_prob is not None:
                    # Filter NaN or Infinite values
                    y_prob = np.clip(y_prob, 0.0, 1.0)
                    
                    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
                    
                    # Format model label
                    desc = ""
                    if name in metrics:
                        desc = metrics[name].get("description", "")
                    elif isinstance(metrics, list):
                        for m_item in metrics:
                            if m_item["model_name"] == name:
                                desc = m_item.get("description", "")
                                break
                                
                    label_name = name
                    if "Voting" in name or "hybrid" in name:
                        label_name = f"{name} (Hybrid)"
                    elif name == "auto_model":
                        label_name = f"auto_model (HistGB)" if MODE == "ML" else "auto_model (BiGRU+Attn)"
                        
                    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=label_name)
                    has_curves = True
            except Exception as ex:
                print(f"[WARN] Calibration failed for {name}: {ex}", file=sys.stderr)
                
    if has_curves:
        plt.ylabel("Fraction of positives", fontsize=12)
        plt.xlabel("Mean predicted probability", fontsize=12)
        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right", fontsize=10)
        plt.title(f"Calibration Curves - Probability Reliability ({MODE} Mode)", fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_json["calibration_plot"], dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved Calibration Curve to {output_json['calibration_plot']}")
    else:
        plt.close()
        output_json["calibration_plot"] = None
        print("[WARN] No calibration curves plotted.")
except Exception as e:
    print(f"[WARN] Calibration Curve task failed: {e}", file=sys.stderr)


# --- Print Final Output ---
print("<<<JSON_START>>>")
print(json.dumps(output_json, indent=2))
print("<<<JSON_END>>>")
