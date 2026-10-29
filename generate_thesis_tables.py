import os, sys, json, math, numpy as np

REPO_DIR = r"c:\TU\deu\tez_v2\scraped data\phishing_mcp_server"
JSON_PATH = os.path.join(REPO_DIR, "cross_dataset_results_proper.json")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return (center - margin) * 100, (center + margin) * 100

md_output = []

def p(text=""):
    md_output.append(text)

# Header
p("# Cross-Dataset Generalization and Domain Transfer Analysis (Table 6)")
p()
p("To evaluate model transfer performance across differing email distributions, models trained on one dataset were evaluated directly against the holdout test set of the counterpart benchmark without domain adaptation, fine-tuning, or retraining.")
p()
p("Both benchmark collections aggregate public email corpora (including Enron, SpamAssassin, Nazario, and CEAS):")
p(f"* **Dataset 1 ({data['metadata']['d1_total_samples']:,} emails, test set N = {data['metadata']['d1_test_samples']:,}):** Academic benchmark containing {data['metadata']['d1_phishing_pct']:.2f}% phishing emails.")
p(f"* **Dataset 2 ({data['metadata']['d2_total_samples']:,} emails, test set N = {data['metadata']['d2_test_samples']:,} DL / 16,498 ML):** Larger-scale benchmark containing {data['metadata']['d2_phishing_pct']:.2f}% phishing emails.")
p()
p(f"An exact-string comparison between the full corpora identified {data['metadata']['exact_string_overlap_global_count']} identical email texts across datasets, while between train and test partitions only {data['metadata']['similarity_direction_A_D2train_vs_D1test']['exact_matches']} exact matches were present. However, inspecting matched emails revealed that this near-zero exact count stems from differing token normalization: Dataset 1 preserved stop words and punctuation symbols, whereas Dataset 2 stripped both. A lexical similarity audit using TF-IDF n-gram cosine similarity revealed substantial cross-corpus overlap:")
p(f"* **Dataset 2 Train vs. Dataset 1 Test (N = {data['metadata']['d1_test_samples']:,}):** {data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']:.2f}% of test emails ({data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_count']:,} samples) share high lexical similarity (cosine similarity >= 0.80) with training samples, while {data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_pct']:.2f}% ({data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_count']:,} samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises {data['metadata']['similarity_direction_A_D2train_vs_D1test']['moderate_similarity_cos50_pct'] - data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']:.2f}%.")
p(f"* **Dataset 1 Train vs. Dataset 2 Test (N = {data['metadata']['d2_test_samples']:,}):** {data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']:.2f}% of test emails ({data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_count']:,} samples) share high lexical similarity (cosine similarity >= 0.80), while {data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_pct']:.2f}% ({data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_count']:,} samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises {data['metadata']['similarity_direction_B_D1train_vs_D2test']['moderate_similarity_cos50_pct'] - data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']:.2f}%.")
p()
p("The transfer benchmark therefore assesses model resilience across differing corpus compositions, label priors (39.24% vs. 52.00%), and token pre-processing pipelines derived from shared public sources.")
p()

# Table 6a
p("### Table 6a: Deep Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1, N = 3,727)")
p("| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d2_d1_dl = data["DL_cross_dataset"]["D2_models_tested_on_D1"]["models"]
# Order models by cross accuracy descending
for m_key, m_data in sorted(d2_d1_dl.items(), key=lambda x: x[1]["cross_domain_accuracy"], reverse=True):
    name_map = {
        "BILSTM": "BiLSTM", "GRU": "GRU", "TEXT_CNN": "TextCNN", "CNN": "1D-CNN",
        "AUTO_MODEL": "AutoModel-DL (BiGRU + Attn.)", "HYBRID": "Hybrid (CNN + Dense)",
        "LSTM": "LSTM", "CNN_LSTM": "CNN-LSTM", "STACKED_LSTM": "Stacked LSTM"
    }
    disp_name = name_map.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['cross_domain_accuracy']*100:.2f}% | {m_data['delta_accuracy']*100:+.2f} pp | {m_data['precision']*100:.2f}% | {m_data['recall']*100:.2f}% | {m_data['f1']*100:.2f}% | {m_data['auc']:.4f} |")
p()

# Table 6b
p("### Table 6b: Deep Learning Transfer (Trained on Dataset 1 -> Tested on Dataset 2, N = 16,497)")
p("| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d1_d2_dl = data["DL_cross_dataset"]["D1_models_tested_on_D2"]["models"]
for m_key, m_data in sorted(d1_d2_dl.items(), key=lambda x: x[1]["cross_domain_accuracy"], reverse=True):
    name_map = {
        "TEXT_CNN": "TextCNN", "AUTO_MODEL": "AutoModel-DL (BiGRU + Attn.)",
        "BILSTM": "BiLSTM", "CNN": "1D-CNN", "CNN_LSTM": "CNN-LSTM",
        "HYBRID": "Hybrid (CNN + Dense)", "GRU": "GRU",
        "STACKED_LSTM": "Stacked LSTM (Non-converged)*", "LSTM": "LSTM (Non-converged)*"
    }
    disp_name = name_map.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['cross_domain_accuracy']*100:.2f}% | {m_data['delta_accuracy']*100:+.2f} pp | {m_data['precision']*100:.2f}% | {m_data['recall']*100:.2f}% | {m_data['f1']*100:.2f}% | {m_data['auc']:.4f} |")
p("*Note: LSTM and Stacked LSTM did not learn an effective decision rule during training on Dataset 1 (native ROC-AUC ≈ 0.52–0.53); their cross-dataset collapse directly reflects in-domain optimization failure.*")
p()

# Table 6c
p("### Table 6c: Classical Machine Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1, N = 3,727)")
p("| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d2_d1_ml = data["ML_hybrid_cross_dataset"]["D2_models_tested_on_D1"]["models"]
for m_key, m_data in sorted(d2_d1_ml.items(), key=lambda x: x[1]["cross_domain_accuracy"], reverse=True):
    name_map = {
        "auto_model": "AutoModel-ML (HistGradBoost)",
        "LogisticRegression": "Logistic Regression",
        "LinearSVC": "LinearSVC",
        "VotingClassifier": "Voting Classifier",
        "SGDClassifier": "SGDClassifier",
        "GaussianNB": "GaussianNB",
        "MultinomialNB": "Multinomial NB",
        "RandomForestClassifier": "Random Forest"
    }
    disp_name = name_map.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['cross_domain_accuracy']*100:.2f}% | {m_data['delta_accuracy']*100:+.2f} pp | {m_data['precision']*100:.2f}% | {m_data['recall']*100:.2f}% | {m_data['f1']*100:.2f}% | {m_data['auc']:.4f} |")
p()

# Table 6d
p("### Table 6d: Classical Machine Learning Transfer (Trained on Dataset 1 -> Tested on Dataset 2, N = 16,498)")
p("| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d1_d2_ml = data["ML_hybrid_cross_dataset"]["D1_models_tested_on_D2"]["models"]
for m_key, m_data in sorted(d1_d2_ml.items(), key=lambda x: x[1]["cross_domain_accuracy"], reverse=True):
    name_map = {
        "auto_model": "AutoModel-ML (HistGradBoost)",
        "LogisticRegression": "Logistic Regression",
        "LinearSVC": "LinearSVC",
        "VotingClassifier": "Voting Classifier",
        "SGDClassifier": "SGDClassifier",
        "GaussianNB": "GaussianNB",
        "MultinomialNB": "Multinomial NB",
        "RandomForestClassifier": "Random Forest"
    }
    disp_name = name_map.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['cross_domain_accuracy']*100:.2f}% | {m_data['delta_accuracy']*100:+.2f} pp | {m_data['precision']*100:.2f}% | {m_data['recall']*100:.2f}% | {m_data['f1']*100:.2f}% | {m_data['auc']:.4f} |")
p("*Note: The ML test split contains 16,498 samples (corresponding to total N = 82,486, matching Chapter 4), whereas the DL split contains 16,497 samples due to an excluded malformed row.*")
p()

# Table 6e
p("### Table 6e: Feature Space Ablation Study (With ROC-AUC)")
p("All models are trained with their exact Optuna-tuned hyperparameters. Native in-domain baselines are dynamically computed and included for reference.")
p()
p("| Setting & Model | Feature Configuration | Accuracy | Phishing Recall | F1-Score | ROC-AUC |")
p("| :--- | :--- | :---: | :---: | :---: | :---: |")

# Native in-domain structural
p("| **Native In-Domain Baselines (Reference)** | | | | | |")
d1_lr_st = data["native_structural_in_domain"]["D1_LogisticRegression"]
d1_svc_st = data["native_structural_in_domain"]["D1_LinearSVC"]
d2_lr_st = data["native_structural_in_domain"]["D2_LogisticRegression"]
d2_svc_st = data["native_structural_in_domain"]["D2_LinearSVC"]

p(f"| Dataset 1 Native (Logistic Regression) | Structural Only (4 features) | {d1_lr_st['accuracy']*100:.2f}% | {d1_lr_st['recall']*100:.2f}% | {d1_lr_st['f1']*100:.2f}% | {d1_lr_st['auc']:.4f} |")
p(f"| Dataset 1 Native (LinearSVC) | Structural Only (4 features) | {d1_svc_st['accuracy']*100:.2f}% | {d1_svc_st['recall']*100:.2f}% | {d1_svc_st['f1']*100:.2f}% | {d1_svc_st['auc']:.4f} |")
p(f"| Dataset 2 Native (Logistic Regression) | Structural Only (4 features) | {d2_lr_st['accuracy']*100:.2f}% | {d2_lr_st['recall']*100:.2f}% | {d2_lr_st['f1']*100:.2f}% | {d2_lr_st['auc']:.4f} |")
p(f"| Dataset 2 Native (LinearSVC) | Structural Only (4 features) | {d2_svc_st['accuracy']*100:.2f}% | {d2_svc_st['recall']*100:.2f}% | {d2_svc_st['f1']*100:.2f}% | {d2_svc_st['auc']:.4f} |")

# D2 -> D1 Ablation
p("| **Dataset 2 -> Dataset 1 (Transfer)** | | | | | |")
for clf in ["LogisticRegression", "LinearSVC"]:
    c_data = data["ML_ablation_study"]["D2_models_tested_on_D1"][clf]
    c_disp = "Logistic Regression" if clf == "LogisticRegression" else "LinearSVC"
    p(f"| {c_disp} | TF-IDF Only (10,000 features) | {c_data['tfidf_only']['accuracy']*100:.2f}% | {c_data['tfidf_only']['recall']*100:.2f}% | {c_data['tfidf_only']['f1']*100:.2f}% | {c_data['tfidf_only']['auc']:.4f} |")
    p(f"| | Structural Only (4 features) | {c_data['struct_only']['accuracy']*100:.2f}% | {c_data['struct_only']['recall']*100:.2f}% | {c_data['struct_only']['f1']*100:.2f}% | {c_data['struct_only']['auc']:.4f} |")
    p(f"| | Hybrid (TF-IDF + Structural) | {c_data['hybrid']['accuracy']*100:.2f}% | {c_data['hybrid']['recall']*100:.2f}% | {c_data['hybrid']['f1']*100:.2f}% | {c_data['hybrid']['auc']:.4f} |")

# D1 -> D2 Ablation
p("| **Dataset 1 -> Dataset 2 (Transfer)** | | | | | |")
for clf in ["LogisticRegression", "LinearSVC"]:
    c_data = data["ML_ablation_study"]["D1_models_tested_on_D2"][clf]
    c_disp = "Logistic Regression" if clf == "LogisticRegression" else "LinearSVC"
    p(f"| {c_disp} | TF-IDF Only (10,000 features) | {c_data['tfidf_only']['accuracy']*100:.2f}% | {c_data['tfidf_only']['recall']*100:.2f}% | {c_data['tfidf_only']['f1']*100:.2f}% | {c_data['tfidf_only']['auc']:.4f} |")
    p(f"| | Structural Only (4 features) | {c_data['struct_only']['accuracy']*100:.2f}% | {c_data['struct_only']['recall']*100:.2f}% | {c_data['struct_only']['f1']*100:.2f}% | {c_data['struct_only']['auc']:.4f} |")
    p(f"| | Hybrid (TF-IDF + Structural) | {c_data['hybrid']['accuracy']*100:.2f}% | {c_data['hybrid']['recall']*100:.2f}% | {c_data['hybrid']['f1']*100:.2f}% | {c_data['hybrid']['auc']:.4f} |")

p()

# Table 6f
p("### Table 6f: Stratified Transfer Evaluation Across Lexical Overlap Subsets")
p("Performance evaluated across distinct similarity strata to isolate genuine out-of-distribution transfer from near-duplicate overlap. Accuracies for novel partitions include 95% Wilson score confidence intervals.")
p()
p("| Transfer Direction & Model | Full Test Set Accuracy | Near-Duplicate Subset Accuracy (cos >= 0.80) | Truly Novel Subset Accuracy [95% CI] (cos < 0.50) | Truly Novel Phishing Recall |")
p("| :--- | :---: | :---: | :---: | :---: |")

# D2 -> D1 Subsets
n_d1_full = data['metadata']['d1_test_samples']
n_d1_near = data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_count']
pct_d1_near = data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']
n_d1_nov = data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_count']
pct_d1_nov = data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_pct']

p(f"| **Dataset 2 -> Dataset 1** | *(N = {n_d1_full:,})* | *(N = {n_d1_near:,}, {pct_d1_near:.1f}%)* | *(N = {n_d1_nov:,}, {pct_d1_nov:.1f}%)* | |")

models_6f_a = [
    ("LogisticRegression", "Logistic Regression"),
    ("LinearSVC", "LinearSVC"),
    ("SGDClassifier", "SGDClassifier"),
    ("VotingClassifier", "Voting Classifier"),
    ("auto_model", "AutoModel-ML (HistGradBoost)"),
    ("BILSTM", "BiLSTM"),
    ("TEXT_CNN", "TextCNN"),
    ("AUTO_MODEL", "AutoModel-DL (BiGRU + Attn.)")
]

for m_key, disp_name in models_6f_a:
    m_sub = data["subset_evaluation"]["D2_models_tested_on_D1"][m_key]
    acc_all = m_sub["all"]["accuracy"] * 100
    acc_near = m_sub["near_duplicates_ge_080"]["accuracy"] * 100
    acc_nov = m_sub["truly_novel_lt_050"]["accuracy"]
    rec_nov = m_sub["truly_novel_lt_050"]["recall"] * 100
    ci_low, ci_high = wilson_ci(acc_nov, n_d1_nov)
    p(f"| {disp_name} | {acc_all:.2f}% | {acc_near:.2f}% | {acc_nov*100:.2f}% [{ci_low:.1f}–{ci_high:.1f}%] | {rec_nov:.2f}% |")

# D1 -> D2 Subsets
n_d2_full = data['metadata']['d2_test_samples']
n_d2_near = data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_count']
pct_d2_near = data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']
n_d2_nov = data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_count']
pct_d2_nov = data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_pct']

p(f"| **Dataset 1 -> Dataset 2** | *(N = {n_d2_full:,})* | *(N = {n_d2_near:,}, {pct_d2_near:.1f}%)* | *(N = {n_d2_nov:,}, {pct_d2_nov:.1f}%)* | |")

models_6f_b = [
    ("SGDClassifier", "SGDClassifier"),
    ("VotingClassifier", "Voting Classifier"),
    ("LogisticRegression", "Logistic Regression"),
    ("LinearSVC", "LinearSVC"),
    ("auto_model", "AutoModel-ML (HistGradBoost)"),
    ("TEXT_CNN", "TextCNN"),
    ("AUTO_MODEL", "AutoModel-DL (BiGRU + Attn.)"),
    ("BILSTM", "BiLSTM")
]

for m_key, disp_name in models_6f_b:
    m_sub = data["subset_evaluation"]["D1_models_tested_on_D2"][m_key]
    acc_all = m_sub["all"]["accuracy"] * 100
    acc_near = m_sub["near_duplicates_ge_080"]["accuracy"] * 100
    acc_nov = m_sub["truly_novel_lt_050"]["accuracy"]
    rec_nov = m_sub["truly_novel_lt_050"]["recall"] * 100
    ci_low, ci_high = wilson_ci(acc_nov, n_d2_nov)
    p(f"| {disp_name} | {acc_all:.2f}% | {acc_near:.2f}% | {acc_nov*100:.2f}% [{ci_low:.1f}–{ci_high:.1f}%] | {rec_nov:.2f}% |")

# Save markdown to file
out_path = os.path.join(REPO_DIR, "table6_final_verified.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_output))

print(f"[SUCCESS] Generated verified tables to: {out_path}", flush=True)
