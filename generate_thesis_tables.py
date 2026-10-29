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

# Dynamic variables from JSON metadata
d1_tot = data['metadata']['d1_total_samples']
d1_te = data['metadata']['d1_test_samples']
d1_phish_pct = data['metadata']['d1_phishing_pct']
d2_tot = data['metadata']['d2_total_samples']
d2_phish_pct = data['metadata']['d2_phishing_pct']
d2_te_dl = data['metadata']['evaluation_sample_sizes']['d2_test_samples_dl']
d2_te_ml = data['metadata']['evaluation_sample_sizes']['d2_test_samples_ml']
d1_empty_test = data['metadata']['empty_rows_audit']['d1_empty_text_in_test']
d1_clean_te = d1_te - d1_empty_test

prep = data['metadata']['preprocessing_differences_audit']
d1_prep = prep['d1_academic_mailing_lists']
d2_prep = prep['d2_corporate_threat_intelligence']

# Header
p("# Cross-Dataset Generalization and Domain Transfer Analysis (Table 6)")
p()
p("To evaluate model transfer performance across differing email distributions, models trained on one dataset were evaluated directly against the holdout test set of the counterpart benchmark without domain adaptation, fine-tuning, or retraining.")
p()
p("Both benchmark collections aggregate public email corpora (including Enron, SpamAssassin, Nazario, and CEAS):")
p(f"* **Dataset 1 ({d1_tot:,} emails, test set N = {d1_te:,}):** Academic benchmark containing {d1_phish_pct:.2f}% phishing emails.")
p(f"* **Dataset 2 ({d2_tot:,} emails, test set N = {d2_te_dl:,} DL / {d2_te_ml:,} ML):** Larger-scale benchmark containing {d2_phish_pct:.2f}% phishing emails.")
p()
p(f"An exact-string comparison between the full corpora identified {data['metadata']['exact_string_overlap_global_count']} identical email texts across datasets, while between train and test partitions only {data['metadata']['similarity_direction_A_D2train_vs_D1test']['exact_matches']} exact matches were present. An empirical audit of token normalization confirmed that this near-zero exact count is driven primarily by differing upstream preprocessing: Dataset 1 preserves common English stop words ({d1_prep['stop_words_pct']:.2f}% of words) and punctuation symbols ({d1_prep['punct_chars_pct']:.2f}% of characters), whereas Dataset 2 aggressively strips both ({d2_prep['stop_words_pct']:.2f}% stop words and {d2_prep['punct_chars_pct']:.2f}% punctuation characters), reflecting a {prep['stop_words_ratio_d1_vs_d2']:.1f}-fold disparity in stop-word density and a {prep['punctuation_ratio_d1_vs_d2']:.1f}-fold disparity in punctuation density. A lexical similarity audit using TF-IDF n-gram cosine similarity revealed substantial cross-corpus overlap:")
p(f"* **Dataset 2 Train vs. Dataset 1 Test (N = {d1_te:,}):** {data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']:.2f}% of test emails ({data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_count']:,} samples) share high lexical similarity (cosine similarity >= 0.80) with training samples, while {data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_pct']:.2f}% ({data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_count']:,} samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises {data['metadata']['similarity_direction_A_D2train_vs_D1test']['moderate_similarity_cos50_pct'] - data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']:.2f}%.")
p(f"* **Dataset 1 Train vs. Dataset 2 Test (N = {d2_te_dl:,}):** {data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']:.2f}% of test emails ({data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_count']:,} samples) share high lexical similarity (cosine similarity >= 0.80), while {data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_pct']:.2f}% ({data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_count']:,} samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises {data['metadata']['similarity_direction_B_D1train_vs_D2test']['moderate_similarity_cos50_pct'] - data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']:.2f}%.")
p()
p(f"The transfer benchmark therefore assesses model resilience across differing corpus compositions, label priors ({d1_phish_pct:.2f}% vs. {d2_phish_pct:.2f}%), and token pre-processing pipelines derived from shared public sources.")
p()

# Table 6a: Deep Learning Transfer (D2 -> D1)
p(f"### Table 6a: Deep Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1 Clean Test Set, N = {d1_clean_te:,})")
p("| Model | In-Domain Accuracy | Clean Cross Accuracy (N=3,613) | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC | Full Test Acc (N=3,727)* |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d2_d1_dl = data["DL_cross_dataset"]["D2_models_tested_on_D1"]["models"]
dl_name_map = {
    "BILSTM": "BiLSTM", "GRU": "GRU", "TEXT_CNN": "TextCNN", "CNN": "1D-CNN",
    "AUTO_MODEL": "AutoModel-DL (BiGRU + Attn.)", "HYBRID": "Hybrid (CNN + Dense)",
    "LSTM": "LSTM", "CNN_LSTM": "CNN-LSTM", "STACKED_LSTM": "Stacked LSTM"
}
for m_key, m_data in sorted(d2_d1_dl.items(), key=lambda x: x[1]["clean_cross_domain_accuracy"], reverse=True):
    disp_name = dl_name_map.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['clean_cross_domain_accuracy']*100:.2f}% | {m_data['clean_delta_accuracy']*100:+.2f} pp | {m_data['clean_precision']*100:.2f}% | {m_data['clean_recall']*100:.2f}% | {m_data['clean_f1']*100:.2f}% | {m_data['clean_auc']:.4f} | {m_data['cross_domain_accuracy']*100:.2f}% |")
p(f"*Note: The clean target test set (N = {d1_clean_te:,}) excludes {d1_empty_test} uninformative rows containing only the literal string 'empty' ({d1_empty_test / d1_te * 100:.2f}% of Dataset 1's test split). Full uncleaned test set accuracy (N = {d1_te:,}) is reported in the rightmost column for reference.*")
p()

# Table 6b: Deep Learning Transfer (D1 -> D2)
p(f"### Table 6b: Deep Learning Transfer (Trained on Dataset 1 -> Tested on Dataset 2, N = {d2_te_dl:,})")
p("| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d1_d2_dl = data["DL_cross_dataset"]["D1_models_tested_on_D2"]["models"]
dl_d1_names = {
    "TEXT_CNN": "TextCNN", "AUTO_MODEL": "AutoModel-DL (BiGRU + Attn.)",
    "BILSTM": "BiLSTM", "CNN": "1D-CNN", "CNN_LSTM": "CNN-LSTM",
    "HYBRID": "Hybrid (CNN + Dense)", "GRU": "GRU",
    "STACKED_LSTM": "Stacked LSTM (Non-converged)*", "LSTM": "LSTM (Non-converged)*"
}
for m_key, m_data in sorted(d1_d2_dl.items(), key=lambda x: x[1]["cross_domain_accuracy"], reverse=True):
    disp_name = dl_d1_names.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['cross_domain_accuracy']*100:.2f}% | {m_data['delta_accuracy']*100:+.2f} pp | {m_data['precision']*100:.2f}% | {m_data['recall']*100:.2f}% | {m_data['f1']*100:.2f}% | {m_data['auc']:.4f} |")
p("*Note: LSTM and Stacked LSTM did not learn an effective decision rule during training on Dataset 1 (native ROC-AUC ≈ 0.52–0.53); their cross-dataset collapse directly reflects in-domain optimization failure.*")
p()

# Table 6c: Classical ML Transfer (D2 -> D1)
p(f"### Table 6c: Classical Machine Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1 Clean Test Set, N = {d1_clean_te:,})")
p("| Model | In-Domain Accuracy | Clean Cross Accuracy (N=3,613) | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC | Full Test Acc (N=3,727)* |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d2_d1_ml = data["ML_hybrid_cross_dataset"]["D2_models_tested_on_D1"]["models"]
ml_name_map = {
    "auto_model": "AutoModel-ML (HistGradBoost)",
    "LogisticRegression": "Logistic Regression",
    "LinearSVC": "LinearSVC",
    "VotingClassifier": "Voting Classifier",
    "SGDClassifier": "SGDClassifier",
    "GaussianNB": "GaussianNB",
    "MultinomialNB": "Multinomial NB",
    "RandomForestClassifier": "Random Forest"
}
for m_key, m_data in sorted(d2_d1_ml.items(), key=lambda x: x[1]["clean_cross_domain_accuracy"], reverse=True):
    disp_name = ml_name_map.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['clean_cross_domain_accuracy']*100:.2f}% | {m_data['clean_delta_accuracy']*100:+.2f} pp | {m_data['clean_precision']*100:.2f}% | {m_data['clean_recall']*100:.2f}% | {m_data['clean_f1']*100:.2f}% | {m_data['clean_auc']:.4f} | {m_data['cross_domain_accuracy']*100:.2f}% |")
p(f"*Note: The clean target test set (N = {d1_clean_te:,}) excludes {d1_empty_test} uninformative rows containing only the literal string 'empty'. Full uncleaned test set accuracy (N = {d1_te:,}) is reported in the rightmost column for reference.*")
p()

# Table 6d: Classical ML Transfer (D1 -> D2)
p(f"### Table 6d: Classical Machine Learning Transfer (Trained on Dataset 1 -> Tested on Dataset 2, N = {d2_te_ml:,})")
p("| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
d1_d2_ml = data["ML_hybrid_cross_dataset"]["D1_models_tested_on_D2"]["models"]
for m_key, m_data in sorted(d1_d2_ml.items(), key=lambda x: x[1]["cross_domain_accuracy"], reverse=True):
    disp_name = ml_name_map.get(m_key, m_key)
    p(f"| **{disp_name}** | {m_data['in_domain_accuracy']*100:.2f}% | {m_data['cross_domain_accuracy']*100:.2f}% | {m_data['delta_accuracy']*100:+.2f} pp | {m_data['precision']*100:.2f}% | {m_data['recall']*100:.2f}% | {m_data['f1']*100:.2f}% | {m_data['auc']:.4f} |")
p(f"*Note: The ML test split contains {d2_te_ml:,} samples (from the ML pipeline run with N = 82,486 total), whereas the DL test split contains {d2_te_dl:,} samples (from the DL pipeline run with N = 82,485 total). Both pipelines use independent stratified splits and independent similarity masks.*")
p()

# Table 6e: Feature Space Ablation Study
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

# Table 6f: Stratified Transfer Evaluation Across Lexical Overlap Subsets
p("### Table 6f: Stratified Transfer Evaluation Across Lexical Overlap Subsets (All Models)")
p("Performance evaluated across distinct similarity strata to isolate genuine out-of-distribution transfer from near-duplicate overlap. Confidence intervals (95% Wilson score) are provided for novel partitions. In Direction A (D2 -> D1), metrics are reported on both the full novel partition (N = 588) and the clean novel partition (N = 474, excluding 114 uninformative 'empty' rows).")
p()

# D2 -> D1 Subsets
n_d1_near = data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_count']
pct_d1_near = data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']
n_d1_nov = data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_count']
pct_d1_nov = data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_pct']
n_d1_clean_nov = n_d1_nov - d1_empty_test

p(f"#### Direction A: Trained on Dataset 2 -> Evaluated on Dataset 1 Subsets")
p("| Model | Clean Test Acc (N=3,613) | Near-Dup Acc (cos >= 0.80, N=2,582) | Clean Novel Acc [95% CI] (cos < 0.50, N=474) | Full Novel Recall (N=588) | Clean Novel Recall (N=474) | Novel ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

# Ordered list of all 17 models for D2 -> D1
all_models_a = [
    ("LogisticRegression", "Logistic Regression"),
    ("LinearSVC", "LinearSVC"),
    ("VotingClassifier", "Voting Classifier"),
    ("SGDClassifier", "SGDClassifier"),
    ("auto_model", "AutoModel-ML (HistGradBoost)"),
    ("GaussianNB", "GaussianNB"),
    ("MultinomialNB", "Multinomial NB"),
    ("RandomForestClassifier", "Random Forest"),
    ("BILSTM", "BiLSTM"),
    ("TEXT_CNN", "TextCNN"),
    ("AUTO_MODEL", "AutoModel-DL (BiGRU + Attn.)"),
    ("GRU", "GRU"),
    ("CNN", "1D-CNN"),
    ("HYBRID", "Hybrid (CNN + Dense)"),
    ("LSTM", "LSTM"),
    ("CNN_LSTM", "CNN-LSTM"),
    ("STACKED_LSTM", "Stacked LSTM")
]

for m_key, disp_name in all_models_a:
    m_sub = data["subset_evaluation"]["D2_models_tested_on_D1"][m_key]
    acc_clean_test = m_sub["clean_test_no_empty"]["accuracy"] * 100
    acc_near = m_sub["near_duplicates_ge_080"]["accuracy"] * 100
    acc_clean_nov = m_sub["clean_novel_lt_050_no_empty"]["accuracy"]
    rec_clean_nov = m_sub["clean_novel_lt_050_no_empty"]["recall"] * 100
    rec_full_nov = m_sub["truly_novel_lt_050"]["recall"] * 100
    auc_nov = m_sub["clean_novel_lt_050_no_empty"]["auc"] if m_sub["clean_novel_lt_050_no_empty"]["auc"] is not None else m_sub["truly_novel_lt_050"]["auc"]
    ci_low, ci_high = wilson_ci(acc_clean_nov, n_d1_clean_nov)
    auc_str = f"{auc_nov:.4f}" if auc_nov is not None else "N/A"
    p(f"| {disp_name} | {acc_clean_test:.2f}% | {acc_near:.2f}% | {acc_clean_nov*100:.2f}% [{ci_low:.1f}–{ci_high:.1f}%] | {rec_full_nov:.2f}% | {rec_clean_nov:.2f}% | {auc_str} |")

p()

# D1 -> D2 Subsets
n_d2_near = data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_count']
pct_d2_near = data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']
n_d2_nov = data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_count']
pct_d2_nov = data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_pct']

p(f"#### Direction B: Trained on Dataset 1 -> Evaluated on Dataset 2 Subsets (Holdout N = {d2_te_ml:,} ML / {d2_te_dl:,} DL)")
p("| Model | Full Test Set Accuracy | Near-Duplicate Subset Accuracy (cos >= 0.80, N=3,534) | Truly Novel Subset Accuracy [95% CI] (cos < 0.50, N=11,191) | Truly Novel Phishing Recall (N=11,191) | Truly Novel ROC-AUC |")
p("| :--- | :---: | :---: | :---: | :---: | :---: |")

all_models_b = [
    ("SGDClassifier", "SGDClassifier"),
    ("VotingClassifier", "Voting Classifier"),
    ("LogisticRegression", "Logistic Regression"),
    ("LinearSVC", "LinearSVC"),
    ("auto_model", "AutoModel-ML (HistGradBoost)"),
    ("GaussianNB", "GaussianNB"),
    ("MultinomialNB", "Multinomial NB"),
    ("RandomForestClassifier", "Random Forest"),
    ("TEXT_CNN", "TextCNN"),
    ("AUTO_MODEL", "AutoModel-DL (BiGRU + Attn.)"),
    ("BILSTM", "BiLSTM"),
    ("CNN", "1D-CNN"),
    ("CNN_LSTM", "CNN-LSTM"),
    ("HYBRID", "Hybrid (CNN + Dense)"),
    ("GRU", "GRU"),
    ("STACKED_LSTM", "Stacked LSTM (Non-converged)*"),
    ("LSTM", "LSTM (Non-converged)*")
]

for m_key, disp_name in all_models_b:
    m_sub = data["subset_evaluation"]["D1_models_tested_on_D2"][m_key]
    acc_all = m_sub["all"]["accuracy"] * 100
    acc_near = m_sub["near_duplicates_ge_080"]["accuracy"] * 100
    acc_nov = m_sub["truly_novel_lt_050"]["accuracy"]
    rec_nov = m_sub["truly_novel_lt_050"]["recall"] * 100
    auc_nov = m_sub["truly_novel_lt_050"]["auc"]
    ci_low, ci_high = wilson_ci(acc_nov, n_d2_nov)
    auc_str = f"{auc_nov:.4f}" if auc_nov is not None else "N/A"
    p(f"| {disp_name} | {acc_all:.2f}% | {acc_near:.2f}% | {acc_nov*100:.2f}% [{ci_low:.1f}–{ci_high:.1f}%] | {rec_nov:.2f}% | {auc_str} |")

p("*Note: Dataset 2 contains zero empty rows; all holdout samples possess non-empty textual content.*")
p()
p("---")
p()
p("## Comprehensive Cross-Dataset Discussion & Analysis")
p()
p("The bidirectional transfer evaluation between Dataset 1 and Dataset 2 uncovers several empirical dynamics governing cross-corpus generalization in email phishing detection:")
p()

# Section 1: Near-duplicate analysis
sub_a = data["subset_evaluation"]["D2_models_tested_on_D1"]
near_accs_a = {k: v["near_duplicates_ge_080"]["accuracy"] * 100 for k, v in sub_a.items()}
min_k_a = min(near_accs_a, key=near_accs_a.get)
max_k_a = max(near_accs_a, key=near_accs_a.get)

sub_b = data["subset_evaluation"]["D1_models_tested_on_D2"]
near_accs_b = {k: v["near_duplicates_ge_080"]["accuracy"] * 100 for k, v in sub_b.items()}
conv_b = {k: v for k, v in near_accs_b.items() if k not in ["LSTM", "STACKED_LSTM"]}
min_k_b = min(conv_b, key=conv_b.get)
max_k_b = max(conv_b, key=conv_b.get)

ml_keys = ["LogisticRegression", "LinearSVC", "SGDClassifier", "VotingClassifier", "auto_model", "GaussianNB", "MultinomialNB", "RandomForestClassifier"]
ml_near_a = [sub_a[k]['near_duplicates_ge_080']['accuracy']*100 for k in ml_keys if k in sub_a]

p("### 1. Generalization Under High Lexical Overlap (Near-Duplicate Subset, cos >= 0.80)")
p("When tested on email samples that share high lexical overlap with the training distribution (cosine similarity >= 0.80), both deep learning and machine learning models demonstrate robust classification accuracy. Specifically:")
p(f"* In the **Dataset 2 -> Dataset 1** direction (testing on {data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_count']:,} near-duplicate samples, {data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']:.1f}% of Dataset 1 test set), near-duplicate accuracy spans from **{near_accs_a[min_k_a]:.2f}%** ({min_k_a}) to **{near_accs_a[max_k_a]:.2f}%** ({max_k_a}). Classical ML models achieve between {min(ml_near_a):.2f}% and {max(ml_near_a):.2f}%.")
p(f"* In the **Dataset 1 -> Dataset 2** direction (testing on {data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_count']:,} near-duplicate samples, {data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']:.1f}% of Dataset 2 test set), near-duplicate accuracy among architectures that converged in-domain spans from **{conv_b[min_k_b]:.2f}%** ({min_k_b}) to **{conv_b[max_k_b]:.2f}%** ({max_k_b}).")
p(f"* **Note on Non-Converged Baselines:** LSTM and Stacked LSTM models trained on Dataset 1 attained near-duplicate accuracies of only {near_accs_b['LSTM']:.2f}% and {near_accs_b['STACKED_LSTM']:.2f}%. Because these architectures failed to optimize during in-domain training on Dataset 1 (achieving in-domain ROC-AUC of only ~0.52–0.53), their low cross-dataset performance represents an in-domain underfitting failure rather than a domain-shift transfer collapse.")
p()

# Section 2: Model-Specific Transfer Performance, Threshold Calibration, and Uninformative Email Bodies on Truly Novel Emails
drops_a = {k: v["all"]["accuracy"]*100 - v["truly_novel_lt_050"]["accuracy"]*100 for k, v in sub_a.items()}
min_dr_a = min(drops_a, key=drops_a.get)
max_dr_a = max(drops_a, key=drops_a.get)

drops_b = {k: v["all"]["accuracy"]*100 - v["truly_novel_lt_050"]["accuracy"]*100 for k, v in sub_b.items()}
min_dr_b = min(drops_b, key=drops_b.get)
max_dr_b = max(drops_b, key=drops_b.get)

empt = data["metadata"]["empty_rows_audit"]

p("### 2. Model-Specific Transfer Performance, Threshold Calibration, and Uninformative Email Bodies on Truly Novel Emails (cos < 0.50)")
p("Evaluating models strictly on the partition of emails exhibiting genuine lexical novelty (cosine similarity < 0.50) isolates authentic out-of-distribution transfer resilience from vocabulary memorization:")
p(f"* **Transfer Direction Asymmetry in Accuracy Drops:** When moving from the aggregate test set to the truly novel partition, accuracy drops differ markedly across transfer directions:")
p(f"  - In the **Dataset 2 -> Dataset 1** direction (novel partition N = {data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_count']:,}), accuracy drops across models range from **{drops_a[min_dr_a]:.2f} pp** ({min_dr_a}) to **{drops_a[max_dr_a]:.2f} pp** ({max_dr_a}). For instance, Logistic Regression drops by {drops_a['LogisticRegression']:.2f} pp (from {sub_a['LogisticRegression']['all']['accuracy']*100:.2f}% to {sub_a['LogisticRegression']['truly_novel_lt_050']['accuracy']*100:.2f}%).")
p(f"  - In the **Dataset 1 -> Dataset 2** direction (novel partition N = {data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_count']:,}), accuracy drops are substantially more moderate across linear architectures, dropping only **{drops_b['SGDClassifier']:.2f} pp** for SGDClassifier (from {sub_b['SGDClassifier']['all']['accuracy']*100:.2f}% to {sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy']*100:.2f}%), **{drops_b['LogisticRegression']:.2f} pp** for Logistic Regression, and **{drops_b['LinearSVC']:.2f} pp** for LinearSVC.")
p(f"* **Model-Specific Recall Variation (D2 -> D1):** On the full Dataset 2 -> Dataset 1 novel partition (N = {n_d1_nov:,}), phishing recall exhibits pronounced model-specific variation across different architectural paradigms, rather than a universal architectural limitation of sequence models:")
p(f"  - Several models suffer substantial recall degradation at the default 0.5 decision threshold, missing approximately half of novel phishing emails: Voting Classifier ({sub_a['VotingClassifier']['truly_novel_lt_050']['recall']*100:.2f}%), GaussianNB ({sub_a['GaussianNB']['truly_novel_lt_050']['recall']*100:.2f}%), Multinomial NB ({sub_a['MultinomialNB']['truly_novel_lt_050']['recall']*100:.2f}%), AutoModel-DL ({sub_a['AUTO_MODEL']['truly_novel_lt_050']['recall']*100:.2f}%), BiLSTM ({sub_a['BILSTM']['truly_novel_lt_050']['recall']*100:.2f}%), TextCNN ({sub_a['TEXT_CNN']['truly_novel_lt_050']['recall']*100:.2f}%), GRU ({sub_a['GRU']['truly_novel_lt_050']['recall']*100:.2f}%), and Hybrid ({sub_a['HYBRID']['truly_novel_lt_050']['recall']*100:.2f}%).")
p(f"  - In sharp contrast, non-linear tree-based ensembles and specific deep architectures retain high novel phishing recall without collapsing: Random Forest ({sub_a['RandomForestClassifier']['truly_novel_lt_050']['recall']*100:.2f}%), AutoModel-ML ({sub_a['auto_model']['truly_novel_lt_050']['recall']*100:.2f}%), LSTM ({sub_a['LSTM']['truly_novel_lt_050']['recall']*100:.2f}%), and 1D-CNN ({sub_a['CNN']['truly_novel_lt_050']['recall']*100:.2f}%), alongside regularized linear models (Logistic Regression: {sub_a['LogisticRegression']['truly_novel_lt_050']['recall']*100:.2f}%, LinearSVC: {sub_a['LinearSVC']['truly_novel_lt_050']['recall']*100:.2f}%, SGDClassifier: {sub_a['SGDClassifier']['truly_novel_lt_050']['recall']*100:.2f}%).")
p(f"* **Impact of Uninformative 'Empty' Email Bodies on Novel Recall:** Forensic inspection of the {n_d1_nov:,} novel test emails revealed that **{empt['d1_empty_in_novel_subset_count']} samples ({empt['d1_empty_in_novel_subset_pct']:.2f}% of the novel partition)** consist solely of the uninformative literal string `'empty'`. Of these empty rows, **{empt['d1_empty_in_novel_phishing']} are labeled phishing** (accounting for **37.56% of all 197 novel phishing emails**) and {empt['d1_empty_in_novel_legitimate']} are labeled legitimate. This creates an extreme inductive bias artifact:")
p(f"  - When evaluated on empty token sequences (sequences consisting entirely of padding tokens), deep sequence models (BiLSTM, TextCNN, AutoModel-DL) and naive Bayes models default to predicting class 0 (legitimate) at the 0.5 decision threshold, consequently misclassifying all {empt['d1_empty_in_novel_phishing']} empty phishing emails by default (0% recall on empty rows).")
p(f"  - Conversely, regularized linear models (Logistic Regression, LinearSVC, SGDClassifier) and tree ensembles possess positive model intercepts/biases for uninformative sparse vectors that cause them to default to predicting class 1 (phishing), thereby trivially capturing all {empt['d1_empty_in_novel_phishing']} empty phishing emails.")
p(f"* **Clean Novel Subset Evaluation (N = {n_d1_clean_nov:,}):** When these {empt['d1_empty_in_novel_subset_count']} uninformative empty email rows are excluded, the apparent 'sequence recall collapse' largely evaporates across the evaluation benchmark:")
p(f"  - **BiLSTM** clean novel recall surges from {sub_a['BILSTM']['truly_novel_lt_050']['recall']*100:.2f}% to **{sub_a['BILSTM']['clean_novel_lt_050_no_empty']['recall']*100:.2f}%** (with {sub_a['BILSTM']['clean_novel_lt_050_no_empty']['accuracy']*100:.2f}% overall accuracy and a ROC-AUC of **{sub_a['BILSTM']['clean_novel_lt_050_no_empty']['auc']:.4f}**).")
p(f"  - **TextCNN** clean novel recall surges from {sub_a['TEXT_CNN']['truly_novel_lt_050']['recall']*100:.2f}% to **{sub_a['TEXT_CNN']['clean_novel_lt_050_no_empty']['recall']*100:.2f}%** (with {sub_a['TEXT_CNN']['clean_novel_lt_050_no_empty']['accuracy']*100:.2f}% overall accuracy and a ROC-AUC of **{sub_a['TEXT_CNN']['clean_novel_lt_050_no_empty']['auc']:.4f}**).")
p(f"  - **AutoModel-DL** clean novel recall surges from {sub_a['AUTO_MODEL']['truly_novel_lt_050']['recall']*100:.2f}% to **{sub_a['AUTO_MODEL']['clean_novel_lt_050_no_empty']['recall']*100:.2f}%** (with {sub_a['AUTO_MODEL']['clean_novel_lt_050_no_empty']['accuracy']*100:.2f}% overall accuracy).")
p(f"  - **Voting Classifier** clean novel recall surges from {sub_a['VotingClassifier']['truly_novel_lt_050']['recall']*100:.2f}% to **{sub_a['VotingClassifier']['clean_novel_lt_050_no_empty']['recall']*100:.2f}%** (with {sub_a['VotingClassifier']['clean_novel_lt_050_no_empty']['accuracy']*100:.2f}% overall accuracy and a ROC-AUC of **{sub_a['VotingClassifier']['clean_novel_lt_050_no_empty']['auc']:.4f}**).")
p(f"  - Linear models maintain solid performance on informative novel emails: Logistic Regression achieves {sub_a['LogisticRegression']['clean_novel_lt_050_no_empty']['accuracy']*100:.2f}% accuracy, {sub_a['LogisticRegression']['clean_novel_lt_050_no_empty']['recall']*100:.2f}% recall, and {sub_a['LogisticRegression']['clean_novel_lt_050_no_empty']['auc']:.4f} ROC-AUC; LinearSVC achieves {sub_a['LinearSVC']['clean_novel_lt_050_no_empty']['accuracy']*100:.2f}% accuracy, {sub_a['LinearSVC']['clean_novel_lt_050_no_empty']['recall']*100:.2f}% recall, and {sub_a['LinearSVC']['clean_novel_lt_050_no_empty']['auc']:.4f} ROC-AUC.")
p(f"* **Threshold Calibration vs. Discriminative Loss:** Even on the uncleaned novel subset (N = 588), BiLSTM preserves a high threshold-independent novel ROC-AUC of **{sub_a['BILSTM']['truly_novel_lt_050']['auc']:.4f}** (and **{sub_a['BILSTM']['clean_novel_lt_050_no_empty']['auc']:.4f}** on the clean novel subset), and TextCNN achieves **{sub_a['TEXT_CNN']['truly_novel_lt_050']['auc']:.4f}** ({sub_a['TEXT_CNN']['clean_novel_lt_050_no_empty']['auc']:.4f} clean). These high AUC values demonstrate that the models maintain strong relative ranking discriminability. The degradation observed at the static 0.5 decision threshold indicates a calibration shift (where predicted probabilities for out-of-distribution phishing samples systematically shifted below 0.5) combined with the uninformative body artifact, rather than a collapse of representation learning.")
p(f"* **Confidence Interval Overlap and Statistical Equivalence:** Across the clean novel subset (N = {n_d1_clean_nov:,}), 95% Wilson confidence intervals span ±3 to 4 pp (BiLSTM: [{wilson_ci(sub_a['BILSTM']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[0]:.1f}–{wilson_ci(sub_a['BILSTM']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[1]:.1f}%], Voting Classifier: [{wilson_ci(sub_a['VotingClassifier']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[0]:.1f}–{wilson_ci(sub_a['VotingClassifier']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[1]:.1f}%], Logistic Regression: [{wilson_ci(sub_a['LogisticRegression']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[0]:.1f}–{wilson_ci(sub_a['LogisticRegression']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[1]:.1f}%], SGDClassifier: [{wilson_ci(sub_a['SGDClassifier']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[0]:.1f}–{wilson_ci(sub_a['SGDClassifier']['clean_novel_lt_050_no_empty']['accuracy'], n_d1_clean_nov)[1]:.1f}%]). Consequently, these top classifiers display overlapping uncertainty intervals rather than a statistically strict hierarchical superiority.")
p(f"* **Robust Large-Scale Transfer (D1 -> D2):** On the Dataset 1 -> Dataset 2 novel partition (N = {n_d2_nov:,}), where no empty rows exist, SGDClassifier achieves **{sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy']*100:.2f}%** accuracy [95% CI: {wilson_ci(sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy'], n_d2_nov)[0]:.1f}–{wilson_ci(sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy'], n_d2_nov)[1]:.1f}%] and **{sub_b['SGDClassifier']['truly_novel_lt_050']['recall']*100:.2f}%** recall (ROC-AUC {sub_b['SGDClassifier']['truly_novel_lt_050']['auc']:.4f}). Among deep architectures, TextCNN maintains **{sub_b['TEXT_CNN']['truly_novel_lt_050']['accuracy']*100:.2f}%** accuracy [95% CI: {wilson_ci(sub_b['TEXT_CNN']['truly_novel_lt_050']['accuracy'], n_d2_nov)[0]:.1f}–{wilson_ci(sub_b['TEXT_CNN']['truly_novel_lt_050']['accuracy'], n_d2_nov)[1]:.1f}%] and **{sub_b['TEXT_CNN']['truly_novel_lt_050']['recall']*100:.2f}%** recall (ROC-AUC {sub_b['TEXT_CNN']['truly_novel_lt_050']['auc']:.4f}), and AutoModel-DL achieves **{sub_b['AUTO_MODEL']['truly_novel_lt_050']['accuracy']*100:.2f}%** accuracy and **{sub_b['AUTO_MODEL']['truly_novel_lt_050']['recall']*100:.2f}%** recall (ROC-AUC {sub_b['AUTO_MODEL']['truly_novel_lt_050']['auc']:.4f}).")
p()

# Section 3: Ablation analysis
ab_a = data["ML_ablation_study"]["D2_models_tested_on_D1"]
ab_b = data["ML_ablation_study"]["D1_models_tested_on_D2"]
p("### 3. Representation Space Ablation: Lexical vs. Engineered Structural Features")
p("The Optuna-tuned feature space ablation (Table 6e) provides conclusive evidence on the relative transferability of lexical versus structural representations:")
p(f"* **Lexical TF-IDF Dominance:** Across both transfer directions, TF-IDF representations alone drive nearly all discriminative capacity. In D2 -> D1, TF-IDF achieves {ab_a['LogisticRegression']['tfidf_only']['accuracy']*100:.2f}% accuracy and {ab_a['LogisticRegression']['tfidf_only']['auc']:.4f} ROC-AUC for Logistic Regression. In D1 -> D2, TF-IDF achieves {ab_b['LogisticRegression']['tfidf_only']['accuracy']*100:.2f}% accuracy and {ab_b['LogisticRegression']['tfidf_only']['auc']:.4f} ROC-AUC.")
p(f"* **Structural Feature Inefficacy:** Structural features alone (URL count, IP presence, urgency score, HTML flag) collapse toward majority-class baselines, achieving only {ab_a['LogisticRegression']['struct_only']['accuracy']*100:.2f}% accuracy (ROC-AUC {ab_a['LogisticRegression']['struct_only']['auc']:.4f}) in D2 -> D1, and {ab_b['LogisticRegression']['struct_only']['accuracy']*100:.2f}% accuracy (ROC-AUC {ab_b['LogisticRegression']['struct_only']['auc']:.4f}, with 0.00% phishing recall) in D1 -> D2.")
p(f"* **Ranking Metric Ablation:** In the D1 -> D2 transfer direction, TF-IDF alone achieved slightly higher ROC-AUC than the Hybrid configuration for both Logistic Regression ({ab_b['LogisticRegression']['tfidf_only']['auc']:.4f} vs. {ab_b['LogisticRegression']['hybrid']['auc']:.4f}) and LinearSVC ({ab_b['LinearSVC']['tfidf_only']['auc']:.4f} vs. {ab_b['LinearSVC']['hybrid']['auc']:.4f}), confirming that incorporating engineered structural features does not improve cross-dataset ranking discriminability.")
p()

# Section 4: Leakage and Conflicting Labels
conf = data["metadata"]["conflicting_labels_audit"]
clean_d1 = data["d1_deduplicated_in_domain_baseline"]
p("### 4. Intra-Dataset Train-Test Leakage and the 'Empty' Email Body Audit")
p("A rigorous forensic audit of intra-dataset duplicate rows was conducted to assess whether duplicate leakage between training and test sets artificially inflated in-domain benchmarks:")
p(f"* **Dataset 1 Conflicting Labels:** A global scan revealed {conf['d1_unique_conflicting_texts']} unique text string across {empt['d1_empty_text_rows']} rows with conflicting binary labels ({empt['d1_empty_text_phishing']} labeled phishing and {empt['d1_empty_text_legitimate']} labeled legitimate). Specifically, all {empt['d1_empty_text_rows']} conflicting rows contain the identical uninformative string `'empty'`. In the 20% stratified test split, {conf['d1_leaked_test_total']} duplicate rows were present from the training set, of which **{conf['d1_leaked_test_conflicting_count']} ({conf['d1_leaked_test_conflicting_pct']:.2f}%)** were these identical empty email bodies with contradictory labels. Rather than representing semantic label ambiguity across distinct textual narratives, this finding reflects label noise arising from uninformative blank email bodies.")
p(f"* **Impact of Test Set Deduplication:** Evaluating native Dataset 1 models on a strictly deduplicated clean test set (N = {d1_te - conf['d1_leaked_test_total']:,}) revealed that in-domain accuracy actually **increased** by +0.81 to +0.97 pp across all classifiers (Logistic Regression: {clean_d1['LogisticRegression']['full_test_accuracy']*100:.2f}% -> {clean_d1['LogisticRegression']['clean_test_accuracy']*100:.2f}%; LinearSVC: {clean_d1['LinearSVC']['full_test_accuracy']*100:.2f}% -> {clean_d1['LinearSVC']['clean_test_accuracy']*100:.2f}%; SGDClassifier: {clean_d1['SGDClassifier']['full_test_accuracy']*100:.2f}% -> {clean_d1['SGDClassifier']['clean_test_accuracy']*100:.2f}%). This verifies that duplicate leakage did not artificially inflate in-domain accuracy; rather, contradictory labels on duplicate rows had imposed an artificial error ceiling.")
p(f"* **Dataset 2 Integrity:** The same audit conducted on Dataset 2 confirmed {empt['d2_empty_text_rows']} empty text rows and {conf['d2_conflicting_rows_total']} conflicting label rows across all {d2_tot:,} samples, confirming full label consistency.")
p()

# Section 5: Limitations
p("## Limitations and Threats to Validity")
p("Several methodological boundaries should be considered when interpreting these findings:")
p(f"1. **Token Pre-Processing Discrepancies:** An empirical quantification revealed substantial token normalization disparities between the benchmarks: Dataset 1 preserves common English stop words ({d1_prep['stop_words_pct']:.2f}% of words) and punctuation characters ({d1_prep['punct_chars_pct']:.2f}% of characters), whereas Dataset 2 stripped both ({d2_prep['stop_words_pct']:.2f}% stop words and {d2_prep['punct_chars_pct']:.2f}% punctuation characters), resulting in a {prep['stop_words_ratio_d1_vs_d2']:.1f}x stop-word and {prep['punctuation_ratio_d1_vs_d2']:.1f}x punctuation density disparity. While target emails were evaluated through the source models' tokenizers/vectorizers, this pre-processing divergence is empirically identified as a primary contributing factor to why raw exact-string deduplication identified only {data['metadata']['exact_string_overlap_global_count']} overlapping emails across datasets (and {data['metadata']['similarity_direction_A_D2train_vs_D1test']['exact_matches']} between train and test splits), whereas TF-IDF cosine similarity detected substantial underlying lexical overlap ({data['metadata']['similarity_direction_A_D2train_vs_D1test']['near_duplicates_cos80_pct']:.2f}% in D1 and {data['metadata']['similarity_direction_B_D1train_vs_D2test']['near_duplicates_cos80_pct']:.2f}% in D2 at cos >= 0.80).")
p("2. **In-Domain Optimization Failure of Specific Architectures:** Deep learning LSTM and Stacked LSTM models failed to converge during in-domain training on Dataset 1 (native ROC-AUC ≈ 0.52–0.53). Consequently, their low cross-dataset scores reflect in-domain optimization failure rather than domain shift transfer vulnerability.")
p("3. **Zero-Shot Decision Boundaries and Calibration:** All transfer evaluations were conducted strictly zero-shot at the default 0.5 decision threshold without post-hoc threshold adjustment, temperature scaling, or domain adaptation. The discrepancy between high novel ROC-AUC scores (> 0.92) and lower thresholded recall demonstrates that domain transfer models require post-deployment calibration to realign decision thresholds with out-of-distribution score distributions.")
p()

# Save markdown to file
out_path = os.path.join(REPO_DIR, "table6_final_verified.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_output))

print(f"[SUCCESS] Generated verified tables to: {out_path}", flush=True)
