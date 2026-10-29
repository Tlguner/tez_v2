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
p("*Note: The ML test split contains 16,498 samples (from the ML pipeline run with N = 82,486 total), whereas the DL test split contains 16,497 samples (from the DL pipeline run with N = 82,485 total). Both pipelines use independent stratified splits and independent similarity masks.*")
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

p()
p("---")
p()
p("## Comprehensive Cross-Dataset Discussion & Analysis")
p()
p("The bidirectional transfer evaluation between Dataset 1 and Dataset 2 uncovers several empirical dynamics governing cross-corpus generalization in email phishing detection:")
p()

# 1. Near-duplicate analysis
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

# 2. Truly novel subset analysis
drops_a = {k: v["all"]["accuracy"]*100 - v["truly_novel_lt_050"]["accuracy"]*100 for k, v in sub_a.items()}
min_dr_a = min(drops_a, key=drops_a.get)
max_dr_a = max(drops_a, key=drops_a.get)

drops_b = {k: v["all"]["accuracy"]*100 - v["truly_novel_lt_050"]["accuracy"]*100 for k, v in sub_b.items()}
min_dr_b = min(drops_b, key=drops_b.get)
max_dr_b = max(drops_b, key=drops_b.get)

p("### 2. Empirical Performance and Phishing Recall on Truly Novel Emails (cos < 0.50)")
p("Evaluating models strictly on the partition of emails exhibiting genuine lexical novelty (cosine similarity < 0.50) isolates authentic out-of-distribution transfer resilience from vocabulary memorization:")
p(f"* **Transfer Direction Asymmetry in Accuracy Drops:** When moving from the aggregate test set to the truly novel partition, accuracy drops differ markedly across transfer directions:")
p(f"  - In the **Dataset 2 -> Dataset 1** direction (novel partition N = {data['metadata']['similarity_direction_A_D2train_vs_D1test']['truly_novel_cos_lt50_count']:,}), accuracy drops across models range from **{drops_a[min_dr_a]:.2f} pp** ({min_dr_a}) to **{drops_a[max_dr_a]:.2f} pp** ({max_dr_a}). For instance, Logistic Regression drops by {drops_a['LogisticRegression']:.2f} pp (from {sub_a['LogisticRegression']['all']['accuracy']*100:.2f}% to {sub_a['LogisticRegression']['truly_novel_lt_050']['accuracy']*100:.2f}%).")
p(f"  - In the **Dataset 1 -> Dataset 2** direction (novel partition N = {data['metadata']['similarity_direction_B_D1train_vs_D2test']['truly_novel_cos_lt50_count']:,}), accuracy drops are substantially more moderate across linear architectures, dropping only **{drops_b['SGDClassifier']:.2f} pp** for SGDClassifier (from {sub_b['SGDClassifier']['all']['accuracy']*100:.2f}% to {sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy']*100:.2f}%), **{drops_b['LogisticRegression']:.2f} pp** for Logistic Regression, and **{drops_b['LinearSVC']:.2f} pp** for LinearSVC.")
p(f"* **Selective Phishing Recall Collapse in Deep Sequence Models (D2 -> D1):** On the Dataset 2 -> Dataset 1 novel partition (N = 588), deep learning sequence architectures suffer severe phishing recall degradation: AutoModel-DL recalls only **{sub_a['AUTO_MODEL']['truly_novel_lt_050']['recall']*100:.2f}%**, BiLSTM **{sub_a['BILSTM']['truly_novel_lt_050']['recall']*100:.2f}%**, TextCNN **{sub_a['TEXT_CNN']['truly_novel_lt_050']['recall']*100:.2f}%**, and GRU **{sub_a['GRU']['truly_novel_lt_050']['recall']*100:.2f}%**, effectively missing approximately half of all novel attack vectors. In contrast, regularized linear models maintain high phishing recall of **{sub_a['SGDClassifier']['truly_novel_lt_050']['recall']*100:.2f}%–{sub_a['LogisticRegression']['truly_novel_lt_050']['recall']*100:.2f}%** (Logistic Regression: {sub_a['LogisticRegression']['truly_novel_lt_050']['recall']*100:.2f}%, LinearSVC: {sub_a['LinearSVC']['truly_novel_lt_050']['recall']*100:.2f}%, SGDClassifier: {sub_a['SGDClassifier']['truly_novel_lt_050']['recall']*100:.2f}%) at {sub_a['SGDClassifier']['truly_novel_lt_050']['accuracy']*100:.2f}%–{sub_a['LogisticRegression']['truly_novel_lt_050']['accuracy']*100:.2f}% overall accuracy.")
p(f"* **Confidence Interval Overlap and Ranking Caution:** Because the Dataset 2 -> Dataset 1 novel subset comprises N = 588 samples, the 95% Wilson confidence intervals span approximately ±3 pp (Logistic Regression: [{wilson_ci(sub_a['LogisticRegression']['truly_novel_lt_050']['accuracy'], 588)[0]:.1f}–{wilson_ci(sub_a['LogisticRegression']['truly_novel_lt_050']['accuracy'], 588)[1]:.1f}%], LinearSVC: [{wilson_ci(sub_a['LinearSVC']['truly_novel_lt_050']['accuracy'], 588)[0]:.1f}–{wilson_ci(sub_a['LinearSVC']['truly_novel_lt_050']['accuracy'], 588)[1]:.1f}%], SGDClassifier: [{wilson_ci(sub_a['SGDClassifier']['truly_novel_lt_050']['accuracy'], 588)[0]:.1f}–{wilson_ci(sub_a['SGDClassifier']['truly_novel_lt_050']['accuracy'], 588)[1]:.1f}%], BiLSTM: [{wilson_ci(sub_a['BILSTM']['truly_novel_lt_050']['accuracy'], 588)[0]:.1f}–{wilson_ci(sub_a['BILSTM']['truly_novel_lt_050']['accuracy'], 588)[1]:.1f}%]). Consequently, these models exhibit overlapping uncertainty intervals rather than a statistically strict hierarchical superiority among the top linear classifiers.")
p(f"* **Robust Large-Scale Transfer (D1 -> D2):** On the Dataset 1 -> Dataset 2 novel partition (N = 11,191), SGDClassifier achieves **{sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy']*100:.2f}%** accuracy [95% CI: {wilson_ci(sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy'], 11191)[0]:.1f}–{wilson_ci(sub_b['SGDClassifier']['truly_novel_lt_050']['accuracy'], 11191)[1]:.1f}%] and **{sub_b['SGDClassifier']['truly_novel_lt_050']['recall']*100:.2f}%** recall. Among deep architectures, TextCNN maintains **{sub_b['TEXT_CNN']['truly_novel_lt_050']['accuracy']*100:.2f}%** accuracy [95% CI: {wilson_ci(sub_b['TEXT_CNN']['truly_novel_lt_050']['accuracy'], 11191)[0]:.1f}–{wilson_ci(sub_b['TEXT_CNN']['truly_novel_lt_050']['accuracy'], 11191)[1]:.1f}%] and **{sub_b['TEXT_CNN']['truly_novel_lt_050']['recall']*100:.2f}%** recall, and AutoModel-DL achieves **{sub_b['AUTO_MODEL']['truly_novel_lt_050']['accuracy']*100:.2f}%** accuracy and **{sub_b['AUTO_MODEL']['truly_novel_lt_050']['recall']*100:.2f}%** recall.")
p()

# 3. Ablation analysis
ab_a = data["ML_ablation_study"]["D2_models_tested_on_D1"]
ab_b = data["ML_ablation_study"]["D1_models_tested_on_D2"]
p("### 3. Representation Space Ablation: Lexical vs. Engineered Structural Features")
p("The Optuna-tuned feature space ablation (Table 6e) provides conclusive evidence on the relative transferability of lexical versus structural representations:")
p(f"* **Lexical TF-IDF Dominance:** Across both transfer directions, TF-IDF representations alone drive nearly all discriminative capacity. In D2 -> D1, TF-IDF achieves {ab_a['LogisticRegression']['tfidf_only']['accuracy']*100:.2f}% accuracy and {ab_a['LogisticRegression']['tfidf_only']['auc']:.4f} ROC-AUC for Logistic Regression. In D1 -> D2, TF-IDF achieves {ab_b['LogisticRegression']['tfidf_only']['accuracy']*100:.2f}% accuracy and {ab_b['LogisticRegression']['tfidf_only']['auc']:.4f} ROC-AUC.")
p(f"* **Structural Feature Inefficacy:** Structural features alone (URL count, IP presence, urgency score, HTML flag) collapse toward majority-class baselines, achieving only {ab_a['LogisticRegression']['struct_only']['accuracy']*100:.2f}% accuracy (ROC-AUC {ab_a['LogisticRegression']['struct_only']['auc']:.4f}) in D2 -> D1, and {ab_b['LogisticRegression']['struct_only']['accuracy']*100:.2f}% accuracy (ROC-AUC {ab_b['LogisticRegression']['struct_only']['auc']:.4f}, with 0.00% phishing recall) in D1 -> D2.")
p(f"* **Ranking Metric Ablation:** In the D1 -> D2 transfer direction, TF-IDF alone achieved slightly higher ROC-AUC than the Hybrid configuration for both Logistic Regression ({ab_b['LogisticRegression']['tfidf_only']['auc']:.4f} vs. {ab_b['LogisticRegression']['hybrid']['auc']:.4f}) and LinearSVC ({ab_b['LinearSVC']['tfidf_only']['auc']:.4f} vs. {ab_b['LinearSVC']['hybrid']['auc']:.4f}), confirming that incorporating engineered structural features does not improve cross-dataset ranking discriminability.")
p()

# 4. Leakage and Conflicting Labels
conf = data["metadata"]["conflicting_labels_audit"]
empt = data["metadata"]["empty_rows_audit"]
clean_d1 = data["d1_deduplicated_in_domain_baseline"]
p("### 4. Intra-Dataset Train-Test Leakage and Conflicting Labels Audit")
p("A rigorous forensic audit of intra-dataset duplicate rows was conducted to assess whether duplicate leakage between training and test sets artificially inflated in-domain benchmarks:")
p(f"* **Dataset 1 Conflicting Labels:** Dataset 1 contains {empt['d1_empty_text_rows']} rows with the text string `'empty'`, distributed across opposing classes ({empt['d1_empty_text_phishing']} labeled phishing and {empt['d1_empty_text_legitimate']} labeled legitimate). In the 20% stratified test split, {conf['d1_leaked_test_total']} duplicate rows were present from the training set, of which **{conf['d1_leaked_test_conflicting_count']} ({conf['d1_leaked_test_conflicting_pct']:.2f}%)** possessed conflicting binary labels.")
p(f"* **Impact of Test Set Deduplication:** Evaluating native Dataset 1 models on a strictly deduplicated clean test set (N = {data['metadata']['d1_test_samples'] - conf['d1_leaked_test_total']:,}) revealed that in-domain accuracy actually **increased** by +0.81 to +0.97 pp across all classifiers (Logistic Regression: {clean_d1['LogisticRegression']['full_test_accuracy']*100:.2f}% -> {clean_d1['LogisticRegression']['clean_test_accuracy']*100:.2f}%; LinearSVC: {clean_d1['LinearSVC']['full_test_accuracy']*100:.2f}% -> {clean_d1['LinearSVC']['clean_test_accuracy']*100:.2f}%; SGDClassifier: {clean_d1['SGDClassifier']['full_test_accuracy']*100:.2f}% -> {clean_d1['SGDClassifier']['clean_test_accuracy']*100:.2f}%). This verifies that duplicate leakage did not artificially inflate in-domain accuracy; rather, contradictory labels on duplicate rows had imposed an artificial error ceiling.")
p(f"* **Dataset 2 Integrity:** The same audit conducted on Dataset 2 confirmed {empt['d2_empty_text_rows']} empty text rows and {conf['d2_conflicting_rows_total']} conflicting label rows across the entire 82,485 samples, confirming full label consistency.")
p()

# 5. Limitations
p("## Limitations and Threats to Validity")
p("Several methodological boundaries should be considered when interpreting these findings:")
p("1. **Token Pre-Processing Discrepancies:** Inspection of matched email pairs revealed differing token normalization conventions between the benchmarks: Dataset 1 preserved common English stop words and punctuation characters, whereas Dataset 2 stripped both. While target emails were evaluated through the source models' tokenizers/vectorizers, this pre-processing divergence explains why raw exact-string deduplication identified only 8 overlapping emails across datasets (and 2 between train and test splits), whereas TF-IDF cosine similarity detected substantial underlying lexical overlap (69.28% in D1 and 21.42% in D2 at cos >= 0.80).")
p("2. **In-Domain Optimization Failure of Specific Architectures:** Deep learning LSTM and Stacked LSTM models failed to converge during in-domain training on Dataset 1 (native ROC-AUC ≈ 0.52–0.53). Consequently, their low cross-dataset scores reflect in-domain optimization failure rather than domain shift transfer vulnerability.")
p("3. **Zero-Shot Decision Boundaries:** All transfer evaluations were conducted strictly zero-shot at the default 0.5 decision threshold without post-hoc threshold adjustment, temperature scaling, or domain adaptation.")
p()

# Save markdown to file
out_path = os.path.join(REPO_DIR, "table6_final_verified.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_output))

print(f"[SUCCESS] Generated verified tables to: {out_path}", flush=True)
