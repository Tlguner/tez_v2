# -*- coding: utf-8 -*-
"""
Post-Training Evaluation & Thesis Metrics Suite
Automatically triggered after any ML or DL pipeline run completes.
Executes:
1. Cross-Dataset Generalization (D1 82.5k <-> D2 18.6k)
2. McNemar Statistical Significance Testing
3. Multiple Random Seeds Stability Validation
4. Thesis Summary & Comparison Aggregation

Saves ALL outputs directly into the current run directory (runDir) and latest directory.
"""

import os
import sys
import json
import io
import shutil
import subprocess

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

py_cmd = sys.executable

def run_cmd(cmd_list):
    try:
        res = subprocess.run(cmd_list, check=True, capture_output=True, text=True, encoding='utf-8')
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Step failed: {e.cmd}\n{e.stderr}")

def main():
    target_run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    mode = sys.argv[2] if len(sys.argv) > 2 else "ML"
    
    print(f"\n=======================================================")
    print(f"🚀 [POST-TRAINING SUITE] Starting Thesis Evaluation for Mode: {mode}")
    if target_run_dir:
        print(f"📂 Run Directory: {target_run_dir}")
    print(f"=======================================================\n")
    
    # 1. Run Cross-Dataset Evaluation
    print("📌 [1/4] Running Bidirectional Cross-Dataset Evaluation (D1 <-> D2)...")
    run_cmd([py_cmd, "cross_dataset_eval.py"])
    
    # 2. Run McNemar Statistical Significance Test
    print("\n📌 [2/4] Running McNemar Statistical Significance Test...")
    run_cmd([py_cmd, "mcnemar_test.py"])
    
    # 3. Run Multiple Random Seeds Stability Evaluation
    print("\n📌 [3/4] Running Multiple Random Seeds Stability Evaluation (5 Seeds)...")
    run_cmd([py_cmd, "multiple_seed_eval.py"])

    # 4. Generate Master Summary & Plot
    print("\n📌 [4/4] Generating Master Thesis Summary & Comparison Charts...")
    run_cmd([py_cmd, "generate_thesis_summary.py"])

    # If specific run_dir is provided, sync all generated artifacts into it
    plots_dir = "thesis_pipeline/plots"
    if target_run_dir and os.path.exists(target_run_dir):
        print(f"\n📁 Copying all evaluation artifacts into run directory: {target_run_dir}...")
        artifacts = [
            "cross_dataset_results.json",
            "mcnemar_test_results.json",
            "seed_stability_results.json",
            "seed_stability_comparison.png",
            "distilbert_baseline_results.json",
            "complete_thesis_summary.json",
            "external_vs_autonomous_comparison.png"
        ]
        for art in artifacts:
            src = os.path.join(plots_dir, art)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(target_run_dir, art))
                print(f"  -> Synced: {art}")

    print("\n✅ [POST-TRAINING SUITE COMPLETE] All evaluation metrics & plots updated!\n")

if __name__ == "__main__":
    main()
