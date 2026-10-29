# Cross-Dataset Generalization and Domain Transfer Analysis (Table 6)

To evaluate model transfer performance across differing email distributions, models trained on one dataset were evaluated directly against the holdout test set of the counterpart benchmark without domain adaptation, fine-tuning, or retraining.

Both benchmark collections aggregate public email corpora (including Enron, SpamAssassin, Nazario, and CEAS):
* **Dataset 1 (18,634 emails, test set N = 3,727):** Academic benchmark containing 39.24% phishing emails.
* **Dataset 2 (82,485 emails, test set N = 16,497 DL / 16,498 ML):** Larger-scale benchmark containing 52.00% phishing emails.

An exact-string comparison between the full corpora identified 8 identical email texts across datasets, while between train and test partitions only 2 exact matches were present. However, inspecting matched emails revealed that this near-zero exact count stems from differing token normalization: Dataset 1 preserved stop words and punctuation symbols, whereas Dataset 2 stripped both. A lexical similarity audit using TF-IDF n-gram cosine similarity revealed substantial cross-corpus overlap:
* **Dataset 2 Train vs. Dataset 1 Test (N = 3,727):** 69.28% of test emails (2,582 samples) share high lexical similarity (cosine similarity >= 0.80) with training samples, while 15.78% (588 samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises 14.94%.
* **Dataset 1 Train vs. Dataset 2 Test (N = 16,497):** 21.42% of test emails (3,534 samples) share high lexical similarity (cosine similarity >= 0.80), while 67.84% (11,191 samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises 10.74%.

The transfer benchmark therefore assesses model resilience across differing corpus compositions, label priors (39.24% vs. 52.00%), and token pre-processing pipelines derived from shared public sources.

### Table 6a: Deep Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1, N = 3,727)
| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM** | 98.52% | 94.31% | -4.21 pp | 95.62% | 89.60% | 92.51% | 0.9913 |
| **GRU** | 98.79% | 91.49% | -7.30 pp | 94.28% | 83.38% | 88.49% | 0.9801 |
| **TextCNN** | 98.91% | 90.50% | -8.41 pp | 84.50% | 92.82% | 88.46% | 0.9761 |
| **1D-CNN** | 98.76% | 90.15% | -8.61 pp | 90.35% | 83.86% | 86.98% | 0.9738 |
| **AutoModel-DL (BiGRU + Attn.)** | 98.43% | 89.08% | -9.35 pp | 93.85% | 77.22% | 84.73% | 0.9763 |
| **Hybrid (CNN + Dense)** | 98.36% | 88.54% | -9.82 pp | 82.08% | 90.56% | 86.11% | 0.9586 |
| **LSTM** | 96.98% | 87.23% | -9.75 pp | 83.58% | 83.93% | 83.75% | 0.9012 |
| **CNN-LSTM** | 98.04% | 85.35% | -12.69 pp | 86.29% | 74.49% | 79.96% | 0.9339 |
| **Stacked LSTM** | 96.47% | 83.58% | -12.89 pp | 85.59% | 69.90% | 76.96% | 0.9265 |

### Table 6b: Deep Learning Transfer (Trained on Dataset 1 -> Tested on Dataset 2, N = 16,497)
| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TextCNN** | 95.87% | 90.49% | -5.38 pp | 90.35% | 91.48% | 90.91% | 0.9602 |
| **AutoModel-DL (BiGRU + Attn.)** | 96.03% | 89.65% | -6.38 pp | 85.49% | 96.47% | 90.65% | 0.9565 |
| **BiLSTM** | 95.98% | 88.77% | -7.21 pp | 83.80% | 97.18% | 90.00% | 0.9514 |
| **1D-CNN** | 96.03% | 87.63% | -8.40 pp | 88.99% | 86.97% | 87.97% | 0.9571 |
| **CNN-LSTM** | 93.43% | 87.37% | -6.06 pp | 86.02% | 90.39% | 88.15% | 0.9409 |
| **Hybrid (CNN + Dense)** | 95.68% | 84.83% | -10.85 pp | 86.32% | 84.17% | 85.23% | 0.9294 |
| **GRU** | 90.18% | 79.97% | -10.21 pp | 80.24% | 81.56% | 80.89% | 0.8535 |
| **Stacked LSTM (Non-converged)*** | 64.13% | 48.67% | -15.46 pp | 64.78% | 2.81% | 5.39% | 0.5317 |
| **LSTM (Non-converged)*** | 61.79% | 47.92% | -13.87 pp | 48.96% | 3.58% | 6.67% | 0.5218 |
*Note: LSTM and Stacked LSTM did not learn an effective decision rule during training on Dataset 1 (native ROC-AUC ≈ 0.52–0.53); their cross-dataset collapse directly reflects in-domain optimization failure.*

### Table 6c: Classical Machine Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1, N = 3,727)
| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 98.85% | 95.33% | -3.52 pp | 90.30% | 98.70% | 94.31% | 0.9932 |
| **LinearSVC** | 98.90% | 95.25% | -3.65 pp | 90.13% | 98.70% | 94.22% | 0.9932 |
| **Voting Classifier** | 98.59% | 94.69% | -3.90 pp | 94.44% | 91.86% | 93.13% | 0.9928 |
| **SGDClassifier** | 98.51% | 93.88% | -4.63 pp | 88.04% | 97.67% | 92.61% | 0.9881 |
| **GaussianNB** | 96.25% | 91.79% | -4.46 pp | 95.16% | 83.31% | 88.84% | 0.9118 |
| **Multinomial NB** | 96.19% | 91.49% | -4.70 pp | 96.96% | 80.85% | 88.18% | 0.9866 |
| **Random Forest** | 96.71% | 89.48% | -7.23 pp | 79.36% | 98.91% | 88.06% | 0.9799 |
| **AutoModel-ML (HistGradBoost)** | 96.10% | 86.66% | -9.44 pp | 75.28% | 98.29% | 85.26% | 0.9568 |

### Table 6d: Classical Machine Learning Transfer (Trained on Dataset 1 -> Tested on Dataset 2, N = 16,498)
| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
*Note: The ML test split contains 16,498 samples (from the ML pipeline run with N = 82,486 total), whereas the DL test split contains 16,497 samples (from the DL pipeline run with N = 82,485 total). Both pipelines use independent stratified splits and independent similarity masks.*

### Table 6e: Feature Space Ablation Study (With ROC-AUC)
All models are trained with their exact Optuna-tuned hyperparameters. Native in-domain baselines are dynamically computed and included for reference.

| Setting & Model | Feature Configuration | Accuracy | Phishing Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Native In-Domain Baselines (Reference)** | | | | | |
| Dataset 1 Native (Logistic Regression) | Structural Only (4 features) | 61.12% | 1.78% | 3.46% | 0.6507 |
| Dataset 1 Native (LinearSVC) | Structural Only (4 features) | 61.12% | 1.78% | 3.46% | 0.6507 |
| Dataset 2 Native (Logistic Regression) | Structural Only (4 features) | 52.38% | 97.77% | 68.11% | 0.5167 |
| Dataset 2 Native (LinearSVC) | Structural Only (4 features) | 52.38% | 97.75% | 68.10% | 0.5169 |
| **Dataset 2 -> Dataset 1 (Transfer)** | | | | | |
| Logistic Regression | TF-IDF Only (10,000 features) | 95.41% | 98.63% | 94.40% | 0.9934 |
| | Structural Only (4 features) | 57.18% | 81.81% | 59.98% | 0.6278 |
| | Hybrid (TF-IDF + Structural) | 95.33% | 98.70% | 94.31% | 0.9932 |
| LinearSVC | TF-IDF Only (10,000 features) | 95.30% | 98.70% | 94.28% | 0.9933 |
| | Structural Only (4 features) | 57.34% | 81.74% | 60.05% | 0.6294 |
| | Hybrid (TF-IDF + Structural) | 95.25% | 98.70% | 94.22% | 0.9932 |
| **Dataset 1 -> Dataset 2 (Transfer)** | | | | | |
| Logistic Regression | TF-IDF Only (10,000 features) | 88.46% | 84.36% | 88.37% | 0.9697 |
| | Structural Only (4 features) | 48.00% | 0.00% | 0.00% | 0.5174 |
| | Hybrid (TF-IDF + Structural) | 88.53% | 85.66% | 88.59% | 0.9674 |
| LinearSVC | TF-IDF Only (10,000 features) | 88.59% | 84.80% | 88.54% | 0.9691 |
| | Structural Only (4 features) | 48.00% | 0.00% | 0.00% | 0.5174 |
| | Hybrid (TF-IDF + Structural) | 88.48% | 85.81% | 88.57% | 0.9672 |

### Table 6f: Stratified Transfer Evaluation Across Lexical Overlap Subsets
Performance evaluated across distinct similarity strata to isolate genuine out-of-distribution transfer from near-duplicate overlap. Accuracies for novel partitions include 95% Wilson score confidence intervals.

| Transfer Direction & Model | Full Test Set Accuracy | Near-Duplicate Subset Accuracy (cos >= 0.80) | Truly Novel Subset Accuracy [95% CI] (cos < 0.50) | Truly Novel Phishing Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Dataset 2 -> Dataset 1** | *(N = 3,727)* | *(N = 2,582, 69.3%)* | *(N = 588, 15.8%)* | |
| Logistic Regression | 95.33% | 98.84% | 84.01% [80.8–86.8%] | 95.94% |
| LinearSVC | 95.25% | 99.03% | 83.33% [80.1–86.1%] | 95.43% |
| SGDClassifier | 93.88% | 97.91% | 80.95% [77.6–83.9%] | 94.92% |
| Voting Classifier | 94.69% | 98.33% | 80.27% [76.9–83.3%] | 55.84% |
| AutoModel-ML (HistGradBoost) | 86.66% | 92.99% | 71.94% [68.2–75.4%] | 98.48% |
| BiLSTM | 94.31% | 97.56% | 81.12% [77.8–84.1%] | 55.33% |
| TextCNN | 90.50% | 97.21% | 69.39% [65.5–73.0%] | 59.39% |
| AutoModel-DL (BiGRU + Attn.) | 89.08% | 91.32% | 76.02% [72.4–79.3%] | 48.73% |
| **Dataset 1 -> Dataset 2** | *(N = 16,497)* | *(N = 3,534, 21.4%)* | *(N = 11,191, 67.8%)* | |
| SGDClassifier | 91.85% | 98.36% | 88.87% [88.3–89.4%] | 88.97% |
| Voting Classifier | 89.59% | 99.17% | 85.15% [84.5–85.8%] | 78.85% |
| Logistic Regression | 88.54% | 98.82% | 83.85% [83.2–84.5%] | 79.94% |
| LinearSVC | 88.48% | 98.85% | 83.78% [83.1–84.5%] | 80.12% |
| AutoModel-ML (HistGradBoost) | 86.95% | 94.32% | 83.88% [83.2–84.5%] | 92.37% |
| TextCNN | 90.49% | 96.15% | 87.83% [87.2–88.4%] | 90.56% |
| AutoModel-DL (BiGRU + Attn.) | 89.65% | 97.45% | 86.13% [85.5–86.8%] | 95.40% |
| BiLSTM | 88.77% | 96.32% | 85.46% [84.8–86.1%] | 96.57% |

---

## Comprehensive Cross-Dataset Discussion & Analysis

The bidirectional transfer evaluation between Dataset 1 and Dataset 2 uncovers several empirical dynamics governing cross-corpus generalization in email phishing detection:

### 1. Generalization Under High Lexical Overlap (Near-Duplicate Subset, cos >= 0.80)
When tested on email samples that share high lexical overlap with the training distribution (cosine similarity >= 0.80), both deep learning and machine learning models demonstrate robust classification accuracy. Specifically:
* In the **Dataset 2 -> Dataset 1** direction (testing on 2,582 near-duplicate samples, 69.3% of Dataset 1 test set), near-duplicate accuracy spans from **84.82%** (STACKED_LSTM) to **99.03%** (LinearSVC). Classical ML models achieve between 92.99% and 99.03%.
* In the **Dataset 1 -> Dataset 2** direction (testing on 3,534 near-duplicate samples, 21.4% of Dataset 2 test set), near-duplicate accuracy among architectures that converged in-domain spans from **88.79%** (GRU) to **99.17%** (VotingClassifier).
* **Note on Non-Converged Baselines:** LSTM and Stacked LSTM models trained on Dataset 1 attained near-duplicate accuracies of only 56.45% and 56.06%. Because these architectures failed to optimize during in-domain training on Dataset 1 (achieving in-domain ROC-AUC of only ~0.52–0.53), their low cross-dataset performance represents an in-domain underfitting failure rather than a domain-shift transfer collapse.

### 2. Empirical Performance and Phishing Recall on Truly Novel Emails (cos < 0.50)
Evaluating models strictly on the partition of emails exhibiting genuine lexical novelty (cosine similarity < 0.50) isolates authentic out-of-distribution transfer resilience from vocabulary memorization:
* **Transfer Direction Asymmetry in Accuracy Drops:** When moving from the aggregate test set to the truly novel partition, accuracy drops differ markedly across transfer directions:
  - In the **Dataset 2 -> Dataset 1** direction (novel partition N = 588), accuracy drops across models range from **8.41 pp** (STACKED_LSTM) to **21.11 pp** (TEXT_CNN). For instance, Logistic Regression drops by 11.32 pp (from 95.33% to 84.01%).
  - In the **Dataset 1 -> Dataset 2** direction (novel partition N = 11,191), accuracy drops are substantially more moderate across linear architectures, dropping only **2.98 pp** for SGDClassifier (from 91.85% to 88.87%), **4.69 pp** for Logistic Regression, and **4.70 pp** for LinearSVC.
* **Selective Phishing Recall Collapse in Deep Sequence Models (D2 -> D1):** On the Dataset 2 -> Dataset 1 novel partition (N = 588), deep learning sequence architectures suffer severe phishing recall degradation: AutoModel-DL recalls only **48.73%**, BiLSTM **55.33%**, TextCNN **59.39%**, and GRU **51.27%**, effectively missing approximately half of all novel attack vectors. In contrast, regularized linear models maintain high phishing recall of **94.92%–95.94%** (Logistic Regression: 95.94%, LinearSVC: 95.43%, SGDClassifier: 94.92%) at 80.95%–84.01% overall accuracy.
* **Confidence Interval Overlap and Ranking Caution:** Because the Dataset 2 -> Dataset 1 novel subset comprises N = 588 samples, the 95% Wilson confidence intervals span approximately ±3 pp (Logistic Regression: [80.8–86.8%], LinearSVC: [80.1–86.1%], SGDClassifier: [77.6–83.9%], BiLSTM: [77.8–84.1%]). Consequently, these models exhibit overlapping uncertainty intervals rather than a statistically strict hierarchical superiority among the top linear classifiers.
* **Robust Large-Scale Transfer (D1 -> D2):** On the Dataset 1 -> Dataset 2 novel partition (N = 11,191), SGDClassifier achieves **88.87%** accuracy [95% CI: 88.3–89.4%] and **88.97%** recall. Among deep architectures, TextCNN maintains **87.83%** accuracy [95% CI: 87.2–88.4%] and **90.56%** recall, and AutoModel-DL achieves **86.13%** accuracy and **95.40%** recall.

### 3. Representation Space Ablation: Lexical vs. Engineered Structural Features
The Optuna-tuned feature space ablation (Table 6e) provides conclusive evidence on the relative transferability of lexical versus structural representations:
* **Lexical TF-IDF Dominance:** Across both transfer directions, TF-IDF representations alone drive nearly all discriminative capacity. In D2 -> D1, TF-IDF achieves 95.41% accuracy and 0.9934 ROC-AUC for Logistic Regression. In D1 -> D2, TF-IDF achieves 88.46% accuracy and 0.9697 ROC-AUC.
* **Structural Feature Inefficacy:** Structural features alone (URL count, IP presence, urgency score, HTML flag) collapse toward majority-class baselines, achieving only 57.18% accuracy (ROC-AUC 0.6278) in D2 -> D1, and 48.00% accuracy (ROC-AUC 0.5174, with 0.00% phishing recall) in D1 -> D2.
* **Ranking Metric Ablation:** In the D1 -> D2 transfer direction, TF-IDF alone achieved slightly higher ROC-AUC than the Hybrid configuration for both Logistic Regression (0.9697 vs. 0.9674) and LinearSVC (0.9691 vs. 0.9672), confirming that incorporating engineered structural features does not improve cross-dataset ranking discriminability.

### 4. Intra-Dataset Train-Test Leakage and Conflicting Labels Audit
A rigorous forensic audit of intra-dataset duplicate rows was conducted to assess whether duplicate leakage between training and test sets artificially inflated in-domain benchmarks:
* **Dataset 1 Conflicting Labels:** Dataset 1 contains 533 rows with the text string `'empty'`, distributed across opposing classes (335 labeled phishing and 198 labeled legitimate). In the 20% stratified test split, 279 duplicate rows were present from the training set, of which **114 (40.86%)** possessed conflicting binary labels.
* **Impact of Test Set Deduplication:** Evaluating native Dataset 1 models on a strictly deduplicated clean test set (N = 3,448) revealed that in-domain accuracy actually **increased** by +0.81 to +0.97 pp across all classifiers (Logistic Regression: 97.10% -> 98.03%; LinearSVC: 97.05% -> 97.97%; SGDClassifier: 96.83% -> 97.80%). This verifies that duplicate leakage did not artificially inflate in-domain accuracy; rather, contradictory labels on duplicate rows had imposed an artificial error ceiling.
* **Dataset 2 Integrity:** The same audit conducted on Dataset 2 confirmed 0 empty text rows and 0 conflicting label rows across the entire 82,485 samples, confirming full label consistency.

## Limitations and Threats to Validity
Several methodological boundaries should be considered when interpreting these findings:
1. **Token Pre-Processing Discrepancies:** Inspection of matched email pairs revealed differing token normalization conventions between the benchmarks: Dataset 1 preserved common English stop words and punctuation characters, whereas Dataset 2 stripped both. While target emails were evaluated through the source models' tokenizers/vectorizers, this pre-processing divergence explains why raw exact-string deduplication identified only 8 overlapping emails across datasets (and 2 between train and test splits), whereas TF-IDF cosine similarity detected substantial underlying lexical overlap (69.28% in D1 and 21.42% in D2 at cos >= 0.80).
2. **In-Domain Optimization Failure of Specific Architectures:** Deep learning LSTM and Stacked LSTM models failed to converge during in-domain training on Dataset 1 (native ROC-AUC ≈ 0.52–0.53). Consequently, their low cross-dataset scores reflect in-domain optimization failure rather than domain shift transfer vulnerability.
3. **Zero-Shot Decision Boundaries:** All transfer evaluations were conducted strictly zero-shot at the default 0.5 decision threshold without post-hoc threshold adjustment, temperature scaling, or domain adaptation.
