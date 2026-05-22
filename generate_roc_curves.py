"""
generate_roc_curves.py
======================
Phishing Detection - ROC Curve Generator
Makalede kullanmak üzere Dataset 1 ve Dataset 2 için
ML ve DL modellerinin ROC eğrilerini üretir.

Çalıştırmak için:
  python generate_roc_curves.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlar için (sunucu/script)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ── Ayarlar ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET1_PATH = os.path.join(SCRIPT_DIR, "Phishing_Email.csv")
DATASET2_PATH = os.path.join(SCRIPT_DIR, "test_dataset", "phishing_email.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "roc_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FEATURES_TFIDF = 5000
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────

def detect_columns(df):
    """Otomatik metin ve etiket sütunları tespiti."""
    text_candidates  = ["text_combined", "Email Text", "email_text", "text",
                        "body", "message", "content", "email", "mail"]
    label_candidates = ["label", "Email Type", "labels", "target",
                        "Category", "class", "y", "is_phishing", "spam"]

    text_col  = next((c for c in text_candidates  if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None:
        raise ValueError("Metin sütunu bulunamadı.")
    if label_col is None:
        raise ValueError("Etiket sütunu bulunamadı.")
    return text_col, label_col


def map_labels(df, label_col):
    """Etiketleri 0/1'e dönüştürür (numeric veya string)."""
    s = df[label_col]
    # Numeric deneme
    y_num = pd.to_numeric(s, errors="coerce")
    if y_num.notna().all() and set(y_num.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
        return y_num.astype(int)

    # String mapping
    phishing_labels = {"phishing", "phishing email", "spam", "malicious"}
    safe_labels     = {"ham", "safe", "safe email", "legitimate", "normal", "benign"}
    mapping = {}
    for val in s.dropna().unique():
        v = str(val).strip().lower()
        if v in phishing_labels:
            mapping[val] = 1
        elif v in safe_labels:
            mapping[val] = 0
    return s.map(mapping)


def load_dataset(path):
    """CSV dosyasını yükler, ön işler ve X, y döndürür."""
    print(f"[INFO] Yükleniyor: {path}")
    df = pd.read_csv(path)
    text_col, label_col = detect_columns(df)
    print(f"[INFO] Metin sütunu: {text_col} | Etiket sütunu: {label_col}")

    df[text_col]  = df[text_col].fillna("")
    df[label_col] = map_labels(df, label_col)
    df = df.dropna(subset=[label_col])
    df = df[df[label_col].isin([0, 1])]

    if len(df) < 10:
        raise ValueError("Yeterli satır yok.")
    if df[label_col].nunique() < 2:
        raise ValueError("Her iki sınıf da gerekli.")

    print(f"[INFO] Toplam satır: {len(df)} | Sınıf dağılımı: {df[label_col].value_counts().to_dict()}")
    return df[text_col].tolist(), df[label_col].tolist()


def get_ml_models():
    """ROC için olasılık desteği olan ML modellerini döndürür.
    LinearSVC olasılık üretmez → CalibratedClassifierCV ile sarar."""
    return [
        ("LogisticRegression",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("MultinomialNB",         MultinomialNB()),
        ("RandomForest",          RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        ("LinearSVC (Calibrated)",CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_STATE))),
        ("GradientBoosting",      GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ]

# ── ROC Çizim Fonksiyonu ─────────────────────────────────────────────────────

# Publication-ready renk paleti (IEEE siyah-beyaz baskıda ayırt edilebilir)
MODEL_STYLES = {
    "LogisticRegression":     {"color": "#2196F3", "ls": "-",  "lw": 2.0},
    "MultinomialNB":          {"color": "#FF9800", "ls": "--", "lw": 1.8},
    "RandomForest":           {"color": "#4CAF50", "ls": "-.", "lw": 2.0},
    "LinearSVC (Calibrated)": {"color": "#9C27B0", "ls": ":", "lw": 2.2},
    "GradientBoosting":       {"color": "#F44336", "ls": "-",  "lw": 1.8},
}


def plot_roc(ax, model_curves, title):
    """
    model_curves: list of (model_name, fpr, tpr, roc_auc)
    """
    for name, fpr, tpr, roc_auc in model_curves:
        style = MODEL_STYLES.get(name, {"color": "gray", "ls": "-", "lw": 1.5})
        label = f"{name}  (AUC = {roc_auc:.4f})"
        ax.plot(fpr, tpr, color=style["color"],
                linestyle=style["ls"], linewidth=style["lw"], label=label)

    # Rastgele sınıflandırıcı referans çizgisi
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)


def generate_ml_roc(texts, labels, dataset_label, out_filename):
    """ML modelleri için ROC eğrisi üretir ve kaydeder."""
    print(f"\n[ML-ROC] {dataset_label} — TF-IDF vektörleştirme başlıyor...")

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    vec = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF)
    X_train = vec.fit_transform(X_train_txt)
    X_test  = vec.transform(X_test_txt)
    y_test  = np.array(y_test)

    model_curves = []
    for name, clf in get_ml_models():
        print(f"    Eğitiliyor: {name}...")
        clf.fit(X_train, y_train)
        # Olasılık skoru al
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_test)[:, 1]
        else:
            scores = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        model_curves.append((name, fpr, tpr, roc_auc))
        print(f"    → AUC: {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_roc(ax, model_curves, f"ROC Curves — {dataset_label} (Classical ML)")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return model_curves


def generate_dl_roc_from_history(dl_json_path, dataset_label, out_filename):
    """
    DL modellerinin validation_accuracy_history kullanarak
    yaklaşık AUC değerleriyle ROC eğrisini çizer.

    NOT: Gerçek probability çıktıları kaydedilmediği için,
    confusion matrix'ten TPR/FPR hesaplıyoruz + smooth eğri çiziyoruz.
    Bu yöntem tezde "approximate ROC from confusion matrix" olarak belirtilmeli.
    """
    print(f"\n[DL-ROC] {dataset_label} — JSON okunuyor: {dl_json_path}")
    with open(dl_json_path, "r") as f:
        data = json.load(f)

    model_display_names = {
        "lstm":       "LSTM",
        "bilstm":     "BiLSTM",
        "gru":        "GRU",
        "cnn":        "CNN (Conv1D)",
        "hybrid":     "Hybrid (CNN+Dense)",
        "auto_model": "AutoModel",
    }

    model_curves = []
    for key, display in model_display_names.items():
        if key not in data:
            continue
        m = data[key]
        cm = m.get("confusion_matrix", None)
        if cm is None:
            continue
        # cm = [[TN, FP], [FN, TP]]
        TN, FP = cm[0]
        FN, TP = cm[1]

        # Gerçek ROC noktaları yerine 3 noktalı yaklaşım:
        # (0,0) → eşik=1  → (TPR_actual, FPR_actual) → (1,1) eşik=0
        tpr_pt = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr_pt = FP / (FP + TN) if (FP + TN) > 0 else 0
        roc_auc = m.get("f1", 0)  # F1 yaklaşık AUC proxy

        fpr = np.array([0.0, fpr_pt, 1.0])
        tpr = np.array([0.0, tpr_pt, 1.0])
        roc_auc_calc = auc(fpr, tpr)

        model_curves.append((display, fpr, tpr, roc_auc_calc))
        print(f"    {display}: TPR={tpr_pt:.4f}, FPR={fpr_pt:.4f}, AUC≈{roc_auc_calc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    # DL için farklı renk paleti
    dl_colors = {
        "LSTM":              "#1A237E",
        "BiLSTM":            "#1976D2",
        "GRU":               "#00BCD4",
        "CNN (Conv1D)":      "#00897B",
        "Hybrid (CNN+Dense)":"#E53935",
        "AutoModel":         "#F57C00",
    }
    dl_styles = ["solid", "dashed", "dashdot", "dotted", (0,(5,2)), (0,(3,1,1,1))]
    for i, (name, fpr, tpr, roc_auc) in enumerate(model_curves):
        color = dl_colors.get(name, "gray")
        ls    = dl_styles[i % len(dl_styles)]
        label = f"{name}  (AUC ≈ {roc_auc:.4f})"
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.0, label=label, marker="o", markersize=4)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"ROC Curves — {dataset_label} (Deep Learning)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def generate_combined_roc(ml_curves_d1, ml_curves_d2, out_filename):
    """ML modellerini iki dataset için yan yana tek figürde gösterir."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_roc(ax1, ml_curves_d1, "Dataset 1 (~18.6k) — Classical ML")
    plot_roc(ax2, ml_curves_d2, "Dataset 2 (~82k) — Classical ML")
    fig.suptitle("ROC Curves: Classical ML Models — Phishing Detection",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def generate_dl_combined_roc(dl_json_d1, dl_json_d2, out_filename):
    """DL modellerini iki dataset için yan yana tek figürde gösterir."""
    dl_colors = {
        "LSTM":              "#1A237E",
        "BiLSTM":            "#1976D2",
        "GRU":               "#00BCD4",
        "CNN (Conv1D)":      "#00897B",
        "Hybrid (CNN+Dense)":"#E53935",
        "AutoModel":         "#F57C00",
    }
    dl_styles = ["solid", "dashed", "dashdot", "dotted", (0,(5,2)), (0,(3,1,1,1))]
    model_display_names = {
        "lstm": "LSTM", "bilstm": "BiLSTM", "gru": "GRU",
        "cnn": "CNN (Conv1D)", "hybrid": "Hybrid (CNN+Dense)", "auto_model": "AutoModel",
    }

    def parse_dl_curves(json_path, label):
        print(f"\n[DL-Combined] {label} — JSON: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        curves = []
        for key, display in model_display_names.items():
            if key not in data:
                continue
            cm = data[key].get("confusion_matrix")
            if not cm:
                continue
            TN, FP = cm[0]; FN, TP = cm[1]
            tpr_pt = TP / (TP + FN) if (TP + FN) > 0 else 0
            fpr_pt = FP / (FP + TN) if (FP + TN) > 0 else 0
            fpr = np.array([0.0, fpr_pt, 1.0])
            tpr = np.array([0.0, tpr_pt, 1.0])
            roc_auc = auc(fpr, tpr)
            curves.append((display, fpr, tpr, roc_auc))
            print(f"    {display}: AUC≈{roc_auc:.4f}")
        return curves

    curves_d1 = parse_dl_curves(dl_json_d1, "Dataset 1") if os.path.exists(dl_json_d1) else []
    curves_d2 = parse_dl_curves(dl_json_d2, "Dataset 2") if os.path.exists(dl_json_d2) else []

    if not curves_d1 and not curves_d2:
        print("[WARN] DL combined: hiç veri bulunamadı, atlanıyor.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, curves, title in [
        (axes[0], curves_d1, "Dataset 1 (~18.6k) — Deep Learning"),
        (axes[1], curves_d2, "Dataset 2 (~82k) — Deep Learning"),
    ]:
        for i, (name, fpr, tpr, roc_auc) in enumerate(curves):
            color = dl_colors.get(name, "gray")
            ls    = dl_styles[i % len(dl_styles)]
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.0,
                    label=f"{name}  (AUC ≈ {roc_auc:.4f})", marker="o", markersize=4)
        ax.plot([0,1],[0,1],"k--",linewidth=1.0,alpha=0.5,label="Random Classifier")
        ax.set_xlim([0.0,1.0]); ax.set_ylim([0.0,1.02])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"ROC Curves: Deep Learning Models — Phishing Detection",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ── Ana Akış ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phishing Detection — ROC Curve Generator")
    print("=" * 60)

    # ── Dataset 1 ML ──────────────────────────────────────────
    if os.path.exists(DATASET1_PATH):
        texts1, labels1 = load_dataset(DATASET1_PATH)
        ml_curves_d1 = generate_ml_roc(
            texts1, labels1,
            dataset_label="Dataset 1 (~18.6k)",
            out_filename="roc_ml_dataset1.png"
        )
    else:
        print(f"[WARN] Dataset 1 bulunamadı: {DATASET1_PATH}")
        ml_curves_d1 = []

    # ── Dataset 2 ML ──────────────────────────────────────────
    # Eğer Dataset 2 farklı bir dosyaysa DATASET2_PATH'i güncelleyin.
    # Şu an aynı dosyayı kullanıyoruz (tek dataset varsa).
    if DATASET2_PATH != DATASET1_PATH and os.path.exists(DATASET2_PATH):
        texts2, labels2 = load_dataset(DATASET2_PATH)
        ml_curves_d2 = generate_ml_roc(
            texts2, labels2,
            dataset_label="Dataset 2 (~82k)",
            out_filename="roc_ml_dataset2.png"
        )
    else:
        print("[INFO] Dataset 2 için ayrı yol tanımlanmamış, atlanıyor.")
        ml_curves_d2 = []

    # ── Kombine ML ROC ────────────────────────────────────────
    if ml_curves_d1 and ml_curves_d2:
        generate_combined_roc(
            ml_curves_d1, ml_curves_d2,
            out_filename="roc_ml_combined.png"
        )

    # ── DL ROC (JSON'dan) ─────────────────────────────────────
    # Dataset 2 DL sonuçları — en son çalışan dosyayı seçin:
    dl_json_d2 = os.path.join(SCRIPT_DIR, "results_dl", "autodl_1773588188934.json")
    if os.path.exists(dl_json_d2):
        generate_dl_roc_from_history(
            dl_json_d2,
            dataset_label="Dataset 2 (~82k)",
            out_filename="roc_dl_dataset2.png"
        )

    # Dataset 1 DL sonuçları — varsa buraya ekleyin:
    dl_json_d1 = os.path.join(SCRIPT_DIR, "results_dl", "autodl_1769568679271.json")
    if os.path.exists(dl_json_d1):
        generate_dl_roc_from_history(
            dl_json_d1,
            dataset_label="Dataset 1 (~18.6k)",
            out_filename="roc_dl_dataset1.png"
        )

    # ── Kombine DL ROC ────────────────────────────────────────
    generate_dl_combined_roc(
        dl_json_d1=dl_json_d1,
        dl_json_d2=dl_json_d2,
        out_filename="roc_dl_combined.png"
    )

    print("\n" + "=" * 60)
    print(f"  Tüm ROC grafikleri kaydedildi: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
