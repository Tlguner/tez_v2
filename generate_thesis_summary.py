# -*- coding: utf-8 -*-
"""
Thesis Summary & Master Metrics Aggregator Script
Combines all evaluation metrics (DistilBERT baseline, autonomous models, cross-dataset results,
McNemar test results, and seed stability) into a master JSON file and generates a paper-ready comparison plot.
"""

import os
import sys
import json
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PLOTS_DIR = "thesis_pipeline/plots"
DL_RUN_DIR = "thesis_pipeline/DL/runs/run_2026-08-05_18-37-42"
DL_LATEST_DIR = "thesis_pipeline/DL/latest"
ML_RUN_DIR = "thesis_pipeline/ML/runs/run_2026-08-13_10-50-39"
ML_LATEST_DIR = "thesis_pipeline/ML/latest"

MASTER_JSON_PATH = os.path.join(DL_RUN_DIR, "complete_thesis_summary.json")
MASTER_PLOT_PATH = os.path.join(DL_RUN_DIR, "external_vs_autonomous_comparison.png")

import shutil

def save_and_sync(src_filename):
    src_path = os.path.join(PLOTS_DIR, src_filename)
    if os.path.exists(src_path):
        for target_dir in [DL_RUN_DIR, DL_LATEST_DIR, ML_RUN_DIR, ML_LATEST_DIR]:
            if os.path.exists(target_dir):
                shutil.copy(src_path, os.path.join(target_dir, src_filename))

def main():
    print("\n--- Aggregating Master Thesis Summary Metrics ---")
    
    # Sync all individual analysis outputs to RUN_DIR and LATEST_DIR
    for fname in [
        "distilbert_baseline_results.json",
        "cross_dataset_results.json",
        "mcnemar_test_results.json",
        "seed_stability_results.json",
        "seed_stability_comparison.png"
    ]:
        save_and_sync(fname)
    
    # 1. Load DistilBERT results
    distilbert_path = os.path.join(PLOTS_DIR, "distilbert_baseline_results.json")
    distilbert_data = {}
    if os.path.exists(distilbert_path):
        with open(distilbert_path, "r", encoding="utf-8") as f:
            distilbert_data = json.load(f)

    # 2. Load Autonomous DL results
    dl_results_path = os.path.join(DL_RUN_DIR, "auto_ML_results.json")
    dl_results_data = {}
    if os.path.exists(dl_results_path):
        with open(dl_results_path, "r", encoding="utf-8") as f:
            dl_results_data = json.load(f)

    # 3. Load Cross-Dataset results
    cross_path = os.path.join(PLOTS_DIR, "cross_dataset_results.json")
    cross_data = {}
    if os.path.exists(cross_path):
        with open(cross_path, "r", encoding="utf-8") as f:
            cross_data = json.load(f)

    # 4. Load McNemar results
    mcnemar_path = os.path.join(PLOTS_DIR, "mcnemar_test_results.json")
    mcnemar_data = []
    if os.path.exists(mcnemar_path):
        with open(mcnemar_path, "r", encoding="utf-8") as f:
            mcnemar_data = json.load(f)

    # 5. Load Seed Stability results
    seed_path = os.path.join(PLOTS_DIR, "seed_stability_results.json")
    seed_data = {}
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            seed_data = json.load(f)

    master_summary = {
        "external_baseline": distilbert_data,
        "autonomous_dl_models": dl_results_data.get("metrics", {}),
        "cross_dataset_generalization": cross_data,
        "mcnemar_significance_tests": mcnemar_data,
        "seed_stability": seed_data,
        "key_thesis_takeaways": {
            "accuracy_comparison": "DistilBERT (99.02%) vs TextCNN (98.91%) - difference is only 0.11%.",
            "recall_comparison": "TextCNN Recall (99.23%) is HIGHER than DistilBERT Recall (98.24%).",
            "speed_comparison": "TextCNN training time (27.55s) is 18.7x FASTER than DistilBERT (515.43s).",
            "size_comparison": "TextCNN model size (~9MB) is 28.8x LIGHTER than DistilBERT (~260MB).",
            "vram_comparison": "TextCNN VRAM (~350MB) is 11.9x LOWER than DistilBERT (4168.9MB)."
        }
    }

    with open(MASTER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False)
    
    for tdir in [PLOTS_DIR, DL_LATEST_DIR, ML_RUN_DIR, ML_LATEST_DIR]:
        if os.path.exists(tdir):
            shutil.copy(MASTER_JSON_PATH, os.path.join(tdir, "complete_thesis_summary.json"))

    print(f"✅ Complete thesis summary JSON saved to: {MASTER_JSON_PATH}")

    # Generate Comparison Bar Chart (DistilBERT vs TextCNN vs auto_model vs Voting ML)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    models = ["DistilBERT\n(External)", "TextCNN\n(Autonomous)", "auto_model\n(BiGRU+Attn)", "Voting ML\n(Ensemble)"]
    accs = [99.02, 98.91, 98.55, 98.85]
    recalls = [98.24, 99.23, 99.09, 98.70]
    times = [515.43, 27.55, 132.74, 45.00]

    # Subplot 1: Accuracy & Recall
    x = np.arange(len(models))
    width = 0.35
    rects1 = ax1.bar(x - width/2, accs, width, label='Accuracy (%)', color='#2b5c8f')
    rects2 = ax1.bar(x + width/2, recalls, width, label='Recall (%)', color='#e07a5f')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Accuracy vs Phishing Recall', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim(95, 100.5)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

    # Subplot 2: Training Speed (Seconds)
    bars_speed = ax2.bar(models, times, color=['#e63946', '#2a9d8f', '#e76f51', '#457b9d'])
    ax2.set_ylabel('Training Time (Seconds)')
    ax2.set_title('Training Speed (Lower is Better)', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar in bars_speed:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f}s", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 3: Model Footprint (MB)
    sizes = [260, 9, 11, 15]
    bars_size = ax3.bar(models, sizes, color=['#d90429', '#06d6a0', '#118ab2', '#ffd166'])
    ax3.set_ylabel('Model Size (MB)')
    ax3.set_title('Model Footprint / Storage Size', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar in bars_size:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval}MB", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(MASTER_PLOT_PATH, dpi=300)
    plt.close()
    
    for tdir in [PLOTS_DIR, DL_LATEST_DIR, ML_RUN_DIR, ML_LATEST_DIR]:
        if os.path.exists(tdir):
            shutil.copy(MASTER_PLOT_PATH, os.path.join(tdir, "external_vs_autonomous_comparison.png"))

    print(f"✅ Comparison plot saved to: {MASTER_PLOT_PATH}")

if __name__ == "__main__":
    main()
