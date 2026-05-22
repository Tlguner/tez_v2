import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "new_roc_result")

# Golden JSON Paths (Absolute Truth)
JSON_FILES = {
    "ML_Dataset1": os.path.join(BASE_DIR, "results", "auto_1774130332291.json"),
    "DL_Dataset1": os.path.join(BASE_DIR, "results_dl", "autodl_1773498676067.json"),
    "ML_Dataset2": os.path.join(BASE_DIR, "results", "auto_1774131381250.json"),
    "DL_Dataset2": os.path.join(BASE_DIR, "results_dl", "autodl_1773588188934.json")
}

# Styles to keep visuals professional
MODEL_STYLING = {
    "LogisticRegression":    {"color": "#1f77b4", "ls": "-",  "label": "Logistic Regression"},
    "MultinomialNB":         {"color": "#ff7f0e", "ls": "--", "label": "Naive Bayes"},
    "RandomForestClassifier":{"color": "#2ca02c", "ls": "-.", "label": "Random Forest"},
    "LinearSVC":             {"color": "#d62728", "ls": ":",  "label": "Linear SVM"},
    "GradientBoostingClassifier": {"color": "#9467bd", "ls": "-", "label": "Gradient Boosting"},
    "lstm":       {"color": "#1a237e", "label": "LSTM", "ls": "-"},
    "bilstm":     {"color": "#1976d2", "label": "BiLSTM", "ls": "--"},
    "gru":        {"color": "#00bcd4", "label": "GRU", "ls": "-."},
    "cnn":        {"color": "#00897b", "label": "CNN (Conv1D)", "ls": ":"},
    "hybrid":     {"color": "#e53935", "label": "Hybrid (CNN+Dense)", "ls": "-"},
    "auto_model": {"color": "#f57c00", "label": "AutoModel", "ls": "--"},
}

def get_roc_points(cm):
    """
    Confusion Matrix'ten (TN, FP, FN, TP) ROC noktalarını (0,0 -> FPR,TPR -> 1,1) üretir.
    cm formatı: [[TN, FP], [FN, TP]]
    """
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    # ROC eğrisi için 3 nokta (Discrete approach)
    fpr_pts = [0.0, fpr, 1.0]
    tpr_pts = [0.0, tpr, 1.0]
    return fpr_pts, tpr_pts

def draw_roc_from_json(json_path, title, out_name):
    print(f"Reading: {os.path.basename(json_path)}")
    if not os.path.exists(json_path):
        print(f"[ERROR] Found no file at {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
    
    # ML ve DL JSON yapıları biraz farklı, ikisini de destekleyelim
    models_to_plot = []
    if "all_models" in data: # ML JSON yapısı
        for m in data["all_models"]:
            models_to_plot.append((m["model_name"], m["confusion_matrix"]))
    else: # DL JSON yapısı
        for m_key, m_val in data.items():
            if isinstance(m_val, dict) and "confusion_matrix" in m_val:
                models_to_plot.append((m_key, m_val["confusion_matrix"]))

    plt.figure(figsize=(8, 6))
    
    for name, cm in models_to_plot:
        fpr, tpr = get_roc_points(cm)
        roc_auc = auc(fpr, tpr)
        
        style = MODEL_STYLING.get(name, {"color": "gray", "ls": "-", "label": name})
        label_text = f"{style['label']} (AUC ≈ {roc_auc:.4f})"
        
        plt.plot(fpr, tpr, color=style["color"], linestyle=style.get("ls", "-"), 
                 lw=2, marker='o', markersize=4, label=label_text)

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(alpha=0.2)
    
    save_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {save_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    draw_roc_from_json(JSON_FILES["ML_Dataset1"], "ROC Curves - Dataset 1 (ML) - Pure JSON Source", "roc_ml_d1_pure.png")
    draw_roc_from_json(JSON_FILES["DL_Dataset1"], "ROC Curves - Dataset 1 (DL) - Pure JSON Source", "roc_dl_d1_pure.png")
    draw_roc_from_json(JSON_FILES["ML_Dataset2"], "ROC Curves - Dataset 2 (ML) - Pure JSON Source", "roc_ml_d2_pure.png")
    draw_roc_from_json(JSON_FILES["DL_Dataset2"], "ROC Curves - Dataset 2 (DL) - Pure JSON Source", "roc_dl_d2_pure.png")

    print("\n[SUCCESS] %100 JSON-based ROC curves are ready in 'new_roc_result' folder.")
