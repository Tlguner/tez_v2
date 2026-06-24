# compare_ml_dl.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import io

# Force stdout/stderr to use UTF-8 encoding
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def format_model_label(model_name, description):
    if not description:
        return model_name
    if "(" in model_name:
        return model_name
        
    model_name_lower = model_name.lower()
    desc_lower = description.lower()
    
    if "voting" in model_name_lower or "hybrid" in model_name_lower or "ensemble" in model_name_lower:
        submodels = []
        if "logistic" in desc_lower or "lr" in desc_lower:
            submodels.append("LR")
        if "randomforest" in desc_lower or "rf" in desc_lower or "random forest" in desc_lower:
            submodels.append("RF")
        if "linearsvc" in desc_lower or "svc" in desc_lower:
            submodels.append("SVC")
        if "multinomialnb" in desc_lower or "nb" in desc_lower:
            submodels.append("NB")
        if "conv1d" in desc_lower or "cnn" in desc_lower:
            submodels.append("CNN")
        if "lstm" in desc_lower:
            submodels.append("LSTM")
        if "gru" in desc_lower:
            submodels.append("GRU")
            
        if submodels:
            return f"{model_name} ({' + '.join(submodels)})"
            
    if "auto_model" in model_name_lower:
        layers = []
        if "conv1d" in desc_lower or "cnn" in desc_lower:
            layers.append("CNN")
        if "bilstm" in desc_lower:
            layers.append("BiLSTM")
        elif "lstm" in desc_lower:
            layers.append("LSTM")
        if "bigru" in desc_lower:
            layers.append("BiGRU")
        elif "gru" in desc_lower:
            layers.append("GRU")
        if "gradientboosting" in desc_lower or "gbm" in desc_lower:
            layers.append("GBM")
        if "xgboost" in desc_lower or "xgb" in desc_lower:
            layers.append("XGBoost")
        if "lightgbm" in desc_lower or "lgb" in desc_lower:
            layers.append("LightGBM")
            
        if layers:
            return f"{model_name} ({' + '.join(layers)})"
            
    return model_name

def load_latest_results(mode):
    path = f"thesis_pipeline/{mode}/latest/run_state.json"
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {mode} state from latest: {e}")
            
    # Fallback to the latest folder in runs/
    runs_dir = f"thesis_pipeline/{mode}/runs"
    if os.path.exists(runs_dir):
        try:
            folders = [d for d in os.listdir(runs_dir) if d.startswith("run_")]
            if folders:
                # Sort folders by modification time
                folders_with_time = []
                for f in folders:
                    folder_path = os.path.join(runs_dir, f)
                    try:
                        mtime = os.path.getmtime(folder_path)
                        folders_with_time.append((f, mtime))
                    except Exception:
                        pass
                if folders_with_time:
                    folders_with_time.sort(key=lambda x: x[1])
                    latest_folder = folders_with_time[-1][0]
                    fallback_path = os.path.join(runs_dir, latest_folder, "run_state.json")
                    if os.path.exists(fallback_path):
                        print(f"Fallback: loading {mode} state from {fallback_path}")
                        with open(fallback_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
        except Exception as e:
            print(f"Error loading fallback {mode} state: {e}")
    return None

def main():
    ml_state = load_latest_results("ML")
    dl_state = load_latest_results("DL")
    
    if not ml_state or not dl_state:
        print("Makine Öğrenmesi (ML) ve Derin Öğrenme (DL) sonuçlarının ikisi de henüz tamamlanmamış. İleride karşılaştırma yapılabilmesi için her iki modun da en az bir kez çalıştırılmış olması gerekir.")
        return

    # Extract best models
    ml_train = ml_state.get("trainResult", {})
    dl_train = dl_state.get("trainResult", {})

    ml_best_raw = ml_train.get("best_model")
    dl_best_raw = dl_train.get("best_model")

    if not ml_best_raw or not dl_best_raw:
        print("En iyi model bilgileri eksik.")
        return

    ml_metrics = ml_train.get("metrics", {}).get(ml_best_raw, {})
    dl_metrics = dl_train.get("metrics", {}).get(dl_best_raw, {})

    if not ml_metrics or not dl_metrics:
        print("Metrikler bulunamadı.")
        return

    ml_best_desc = ml_train.get("model_descriptions", {}).get(ml_best_raw, "")
    dl_best_desc = dl_train.get("model_descriptions", {}).get(dl_best_raw, "")

    ml_best_name = format_model_label(ml_best_raw, ml_best_desc)
    dl_best_name = format_model_label(dl_best_raw, dl_best_desc)

    print(f"ML Best Model: {ml_best_name}")
    print(f"DL Best Model: {dl_best_name}")

    # Metrics to compare
    metrics = ["accuracy", "f1", "precision", "recall"]
    ml_vals = [
        ml_metrics.get("accuracy", 0),
        ml_metrics.get("f1", ml_metrics.get("f1_score", 0)),
        ml_metrics.get("precision", 0),
        ml_metrics.get("recall", 0)
    ]
    dl_vals = [
        dl_metrics.get("accuracy", 0),
        dl_metrics.get("f1", dl_metrics.get("f1_score", 0)),
        dl_metrics.get("precision", 0),
        dl_metrics.get("recall", 0)
    ]

    # Create plot directory
    os.makedirs("thesis_pipeline/plots", exist_ok=True)

    # 1. Performance Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, ml_vals, width, label=f'Best ML: {ml_best_name.upper()}', color='#4CAF50')
    bars2 = ax.bar(x + width/2, dl_vals, width, label=f'Best DL: {dl_best_name.upper()}', color='#2196F3')

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Machine Learning (ML) vs. Deep Learning (DL)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("thesis_pipeline/plots/ml_vs_dl_comparison.png", dpi=300)
    plt.close()

    # 2. Time & Resource Usage Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training Time
    ml_time = ml_metrics.get("time_s", 0)
    dl_time = dl_metrics.get("time_s", 0)
    
    ax1.bar([f'ML: {ml_best_name}', f'DL: {dl_best_name}'], [ml_time, dl_time], color=['#a5d6a7', '#90caf9'], width=0.5)
    ax1.set_ylabel('Training Time (Seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    max_time = max(ml_time, dl_time)
    for i, v in enumerate([ml_time, dl_time]):
        ax1.text(i, v + (max_time*0.01 if max_time > 0 else 1), f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')

    # Resource usage (RAM)
    ml_ram = ml_metrics.get("max_ram_mb", 0)
    dl_ram = dl_metrics.get("max_ram_mb", 0)
    
    ax2.bar([f'ML: {ml_best_name}', f'DL: {dl_best_name}'], [ml_ram, dl_ram], color=['#81c784', '#64b5f6'], width=0.5)
    ax2.set_ylabel('Maximum Memory Usage (MB)')
    ax2.set_title('Hardware Memory (RAM) Consumption')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    max_ram = max(ml_ram, dl_ram)
    for i, v in enumerate([ml_ram, dl_ram]):
        ax2.text(i, v + (max_ram*0.01 if max_ram > 0 else 1), f'{v:.1f} MB', ha='center', va='bottom', fontweight='bold')

    fig.tight_layout()
    plt.savefig("thesis_pipeline/plots/ml_vs_dl_resources.png", dpi=300)
    plt.close()
    
    print("✅ ML vs DL comparison graphs successfully saved to 'thesis_pipeline/plots/' directory.")

if __name__ == "__main__":
    main()
