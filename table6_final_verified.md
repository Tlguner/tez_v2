# Cross-Dataset Generalization and Domain Transfer Analysis (Table 6)

To evaluate model transfer performance across differing email distributions, models trained on one dataset were evaluated directly against the holdout test set of the counterpart benchmark without domain adaptation, fine-tuning, or retraining.

Both benchmark collections aggregate public email corpora (including Enron, SpamAssassin, Nazario, and CEAS):
* **Dataset 1 (18,634 emails, test set N = 3,727):** Academic benchmark containing 39.24% phishing emails.
* **Dataset 2 (82,485 emails, test set N = 16,497 DL / 16,498 ML):** Larger-scale benchmark containing 52.00% phishing emails.

An exact-string comparison between the full corpora identified 8 identical email texts across datasets, while between train and test partitions only 2 exact matches were present. An empirical audit of token normalization confirmed that differing upstream preprocessing is a likely contributing factor: Dataset 1 preserves common English stop words (29.63% of words) and punctuation symbols (6.52% of characters), whereas Dataset 2 aggressively strips both (6.61% stop words and 0.67% punctuation characters), reflecting a 4.5-fold disparity in stop-word density and a 9.7-fold disparity in punctuation density. A lexical similarity audit using TF-IDF n-gram cosine similarity revealed substantial cross-corpus overlap:
* **Dataset 2 Train vs. Dataset 1 Test (N = 3,727):** 69.28% of test emails (2,582 samples) share high lexical similarity (cosine similarity >= 0.80) with training samples, while 15.78% (588 samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises 14.94%.
* **Dataset 1 Train vs. Dataset 2 Test (N = 16,497):** 21.42% of test emails (3,534 samples) share high lexical similarity (cosine similarity >= 0.80), while 67.84% (11,191 samples) represent genuinely novel content (cosine similarity < 0.50). The intermediate similarity region (0.50 to 0.80) comprises 10.74%.

The transfer benchmark therefore assesses model resilience across differing corpus compositions, label priors (39.24% vs. 52.00%), and token pre-processing pipelines derived from shared public sources.

### Table 6a: Deep Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1 Clean Test Set, N = 3,613)
| Model | In-Domain Accuracy | Clean Cross Accuracy (N=3,613) | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC | Full Test Acc (N=3,727)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM** | 98.52% | 96.18% | -2.34 pp | 95.62% | 94.38% | 95.00% | 0.9940 | 94.31% |
| **GRU** | 98.79% | 93.27% | -5.52 pp | 94.28% | 87.82% | 90.94% | 0.9832 | 91.49% |
| **TextCNN** | 98.91% | 92.25% | -6.66 pp | 84.50% | 97.77% | 90.65% | 0.9871 | 90.50% |
| **1D-CNN** | 98.76% | 90.95% | -7.81 pp | 92.68% | 83.00% | 87.57% | 0.9783 | 90.15% |
| **AutoModel-DL (BiGRU + Attn.)** | 98.43% | 90.78% | -7.65 pp | 93.85% | 81.34% | 87.15% | 0.9799 | 89.08% |
| **Hybrid (CNN + Dense)** | 98.36% | 90.23% | -8.13 pp | 82.08% | 95.39% | 88.24% | 0.9710 | 88.54% |
| **LSTM** | 96.98% | 87.93% | -9.05 pp | 85.16% | 83.07% | 84.10% | 0.9016 | 87.23% |
| **CNN-LSTM** | 98.04% | 86.00% | -12.04 pp | 88.41% | 73.13% | 80.05% | 0.9395 | 85.35% |
| **Stacked LSTM** | 96.47% | 84.17% | -12.30 pp | 87.78% | 68.30% | 76.82% | 0.9320 | 83.58% |
*Note: The clean target test set (N = 3,613) excludes 114 uninformative rows containing only the literal string 'empty' (3.06% of Dataset 1's test split). Full uncleaned test set accuracy (N = 3,727) is reported in the rightmost column for reference.*

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

### Table 6c: Classical Machine Learning Transfer (Trained on Dataset 2 -> Tested on Dataset 1 Clean Test Set, N = 3,613)
| Model | In-Domain Accuracy | Clean Cross Accuracy (N=3,613) | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC | Full Test Acc (N=3,727)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Voting Classifier** | 98.59% | 96.57% | -2.02 pp | 94.44% | 96.76% | 95.59% | 0.9954 | 94.69% |
| **Logistic Regression** | 98.85% | 96.29% | -2.56 pp | 92.25% | 98.63% | 95.33% | 0.9957 | 95.33% |
| **LinearSVC** | 98.90% | 96.21% | -2.69 pp | 92.06% | 98.63% | 95.23% | 0.9957 | 95.25% |
| **SGDClassifier** | 98.51% | 94.80% | -3.71 pp | 89.79% | 97.55% | 93.51% | 0.9917 | 93.88% |
| **GaussianNB** | 96.25% | 93.58% | -2.67 pp | 95.16% | 87.75% | 91.30% | 0.9344 | 91.79% |
| **Multinomial NB** | 96.19% | 93.27% | -2.92 pp | 96.96% | 85.16% | 90.68% | 0.9894 | 91.49% |
| **Random Forest** | 96.71% | 90.26% | -6.45 pp | 80.33% | 98.85% | 88.63% | 0.9834 | 89.48% |
| **AutoModel-ML (HistGradBoost)** | 96.10% | 87.35% | -8.75 pp | 75.93% | 98.20% | 85.64% | 0.9621 | 86.66% |
*Note: The clean target test set (N = 3,613) excludes 114 uninformative rows containing only the literal string 'empty'. Full uncleaned test set accuracy (N = 3,727) is reported in the rightmost column for reference.*

### Table 6d: Classical Machine Learning Transfer (Trained on Dataset 1 -> Tested on Dataset 2, N = 16,498)
| Model | In-Domain Accuracy | Cross-Dataset Accuracy | Accuracy Change (pp) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SGDClassifier** | 96.83% | 91.85% | -4.98 pp | 92.34% | 91.96% | 92.15% | 0.9704 |
| **Voting Classifier** | 97.05% | 89.59% | -7.46 pp | 94.57% | 84.86% | 89.45% | 0.9764 |
| **Logistic Regression** | 97.10% | 88.54% | -8.56 pp | 91.74% | 85.69% | 88.61% | 0.9674 |
| **LinearSVC** | 97.05% | 88.48% | -8.57 pp | 91.51% | 85.81% | 88.57% | 0.9672 |
| **AutoModel-ML (HistGradBoost)** | 95.89% | 86.95% | -8.94 pp | 83.41% | 93.50% | 88.17% | 0.9360 |
| **GaussianNB** | 94.93% | 85.63% | -9.30 pp | 92.19% | 79.08% | 85.13% | 0.8644 |
| **Random Forest** | 96.59% | 85.38% | -11.21 pp | 83.45% | 89.66% | 86.45% | 0.9437 |
| **Multinomial NB** | 95.87% | 83.32% | -12.55 pp | 95.54% | 71.24% | 81.62% | 0.9663 |
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

### Table 6f: Stratified Transfer Evaluation Across Lexical Overlap Subsets (All Models)
Performance evaluated across distinct similarity strata to isolate genuine out-of-distribution transfer from near-duplicate overlap. Confidence intervals (95% Wilson score) are provided for novel partitions. In Direction A (D2 -> D1), metrics are reported on both the full novel partition (N = 588) and the clean novel partition (N = 474, excluding 114 uninformative 'empty' rows).

#### Direction A: Trained on Dataset 2 -> Evaluated on Dataset 1 Subsets
| Model | Clean Test Acc (N=3,613) | Near-Dup Acc (cos >= 0.80, N=2,582) | Clean Novel Acc [95% CI] (cos < 0.50, N=474) | Full Novel Recall (N=588) | Clean Novel Recall (N=474) | Novel ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 96.29% | 98.84% | 88.61% [85.4–91.2%] | 95.94% | 93.50% | 0.9669 |
| LinearSVC | 96.21% | 99.03% | 87.76% [84.5–90.4%] | 95.43% | 92.68% | 0.9673 |
| Voting Classifier | 96.57% | 98.33% | 91.14% [88.2–93.4%] | 55.84% | 89.43% | 0.9704 |
| SGDClassifier | 94.80% | 97.91% | 84.81% [81.3–87.8%] | 94.92% | 91.87% | 0.9533 |
| AutoModel-ML (HistGradBoost) | 87.35% | 92.99% | 73.63% [69.5–77.4%] | 98.48% | 97.56% | 0.8796 |
| GaussianNB | 93.58% | 94.31% | 91.14% [88.2–93.4%] | 49.75% | 79.67% | 0.8917 |
| Multinomial NB | 93.27% | 93.57% | 91.35% [88.5–93.6%] | 46.19% | 73.98% | 0.9581 |
| Random Forest | 90.26% | 95.00% | 77.43% [73.5–81.0%] | 97.97% | 96.75% | 0.9202 |
| BiLSTM | 96.18% | 97.56% | 92.19% [89.4–94.3%] | 55.33% | 88.62% | 0.9692 |
| TextCNN | 92.25% | 97.21% | 77.64% [73.7–81.2%] | 59.39% | 95.12% | 0.9393 |
| AutoModel-DL (BiGRU + Attn.) | 90.78% | 91.32% | 85.86% [82.4–88.7%] | 48.73% | 78.05% | 0.9266 |
| GRU | 93.27% | 94.23% | 90.51% [87.5–92.8%] | 51.27% | 82.11% | 0.9417 |
| 1D-CNN | 90.95% | 92.22% | 83.33% [79.7–86.4%] | 84.77% | 75.61% | 0.9149 |
| Hybrid (CNN + Dense) | 90.23% | 95.12% | 77.00% [73.0–80.6%] | 57.36% | 91.87% | 0.9074 |
| LSTM | 87.93% | 90.32% | 79.11% [75.2–82.5%] | 87.82% | 80.49% | 0.8487 |
| CNN-LSTM | 86.00% | 86.87% | 78.48% [74.6–81.9%] | 81.73% | 70.73% | 0.8358 |
| Stacked LSTM | 84.17% | 84.82% | 77.64% [73.7–81.2%] | 78.68% | 65.85% | 0.8599 |

#### Direction B: Trained on Dataset 1 -> Evaluated on Dataset 2 Subsets (Holdout N = 16,498 ML / 16,497 DL)
| Model | Full Test Set Accuracy | Near-Duplicate Subset Accuracy (cos >= 0.80, N=3,534) | Truly Novel Subset Accuracy [95% CI] (cos < 0.50, N=11,191) | Truly Novel Phishing Recall (N=11,191) | Truly Novel ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SGDClassifier | 91.85% | 98.36% | 88.87% [88.3–89.4%] | 88.97% | 0.9467 |
| Voting Classifier | 89.59% | 99.17% | 85.15% [84.5–85.8%] | 78.85% | 0.9558 |
| Logistic Regression | 88.54% | 98.82% | 83.85% [83.2–84.5%] | 79.94% | 0.9400 |
| LinearSVC | 88.48% | 98.85% | 83.78% [83.1–84.5%] | 80.12% | 0.9392 |
| AutoModel-ML (HistGradBoost) | 86.95% | 94.32% | 83.88% [83.2–84.5%] | 92.37% | 0.8984 |
| GaussianNB | 85.63% | 96.64% | 80.68% [79.9–81.4%] | 72.77% | 0.8197 |
| Multinomial NB | 83.32% | 97.68% | 76.61% [75.8–77.4%] | 60.87% | 0.9408 |
| Random Forest | 85.38% | 96.21% | 80.61% [79.9–81.3%] | 86.53% | 0.9132 |
| TextCNN | 90.49% | 96.15% | 87.83% [87.2–88.4%] | 90.56% | 0.9364 |
| AutoModel-DL (BiGRU + Attn.) | 89.65% | 97.45% | 86.13% [85.5–86.8%] | 95.40% | 0.9234 |
| BiLSTM | 88.77% | 96.32% | 85.46% [84.8–86.1%] | 96.57% | 0.9202 |
| 1D-CNN | 87.63% | 95.98% | 83.67% [83.0–84.3%] | 84.16% | 0.9331 |
| CNN-LSTM | 87.37% | 94.54% | 84.11% [83.4–84.8%] | 88.91% | 0.9100 |
| Hybrid (CNN + Dense) | 84.83% | 93.04% | 81.10% [80.4–81.8%] | 82.74% | 0.8929 |
| GRU | 79.97% | 88.79% | 76.31% [75.5–77.1%] | 81.58% | 0.7982 |
| Stacked LSTM (Non-converged)* | 48.67% | 56.06% | 45.21% [44.3–46.1%] | 0.66% | 0.5110 |
| LSTM (Non-converged)* | 47.92% | 56.45% | 43.93% [43.0–44.9%] | 0.97% | 0.4864 |
*Note: Dataset 2 contains zero empty rows; all holdout samples possess non-empty textual content.*

---

## Comprehensive Cross-Dataset Discussion & Analysis

The bidirectional transfer evaluation between Dataset 1 and Dataset 2 uncovers several empirical dynamics governing cross-corpus generalization in email phishing detection:

### 1. Generalization Under High Lexical Overlap (Near-Duplicate Subset, cos >= 0.80)
When tested on email samples that share high lexical overlap with the training distribution (cosine similarity >= 0.80), both deep learning and machine learning models demonstrate robust classification accuracy. Specifically:
* In the **Dataset 2 -> Dataset 1** direction (testing on 2,582 near-duplicate samples, 69.3% of Dataset 1 test set), near-duplicate accuracy spans from **84.82%** (STACKED_LSTM) to **99.03%** (LinearSVC). Classical ML models achieve between 92.99% and 99.03%.
* In the **Dataset 1 -> Dataset 2** direction (testing on 3,534 near-duplicate samples, 21.4% of Dataset 2 test set), near-duplicate accuracy among architectures that converged in-domain spans from **88.79%** (GRU) to **99.17%** (VotingClassifier).
* **Note on Non-Converged Baselines:** LSTM and Stacked LSTM models trained on Dataset 1 attained near-duplicate accuracies of only 56.45% and 56.06%. Because these architectures failed to optimize during in-domain training on Dataset 1 (achieving in-domain ROC-AUC of only ~0.52–0.53), their low cross-dataset performance represents an in-domain underfitting failure rather than a domain-shift transfer collapse.

### 2. Model-Specific Transfer Performance and the Impact of Uninformative Email Bodies on Truly Novel Emails (cos < 0.50)
Evaluating models strictly on the partition of emails exhibiting genuine lexical novelty (cosine similarity < 0.50) isolates authentic out-of-distribution transfer resilience from vocabulary memorization:
* **Transfer Direction Asymmetry in Accuracy Drops:** When moving from the aggregate test set to the truly novel partition, accuracy drops differ markedly across transfer directions:
  - In the **Dataset 2 -> Dataset 1** direction (novel partition N = 588), accuracy drops across models range from **8.41 pp** (STACKED_LSTM) to **21.11 pp** (TEXT_CNN). For instance, Logistic Regression drops by 11.32 pp (from 95.33% to 84.01%).
  - In the **Dataset 1 -> Dataset 2** direction (novel partition N = 11,191), accuracy drops are substantially more moderate across linear architectures, dropping only **2.98 pp** for SGDClassifier (from 91.85% to 88.87%), **4.69 pp** for Logistic Regression, and **4.70 pp** for LinearSVC.
* **Model-Specific Recall Variation (D2 -> D1):** On the full Dataset 2 -> Dataset 1 novel partition (N = 588), phishing recall exhibits pronounced model-specific variation across different architectural paradigms, rather than a universal family-wide collapse:
  - Several models suffer substantial recall degradation at the default 0.5 decision threshold, missing approximately half of novel phishing emails: Voting Classifier (55.84%), GaussianNB (49.75%), Multinomial NB (46.19%), AutoModel-DL (48.73%), BiLSTM (55.33%), TextCNN (59.39%), GRU (51.27%), and Hybrid (57.36%).
  - In sharp contrast, non-linear tree-based ensembles and specific deep architectures retain high novel phishing recall without collapsing: Random Forest (97.97%), AutoModel-ML (98.48%), LSTM (87.82%), and 1D-CNN (84.77%), alongside regularized linear models (Logistic Regression: 95.94%, LinearSVC: 95.43%, SGDClassifier: 94.92%).
* **The 'Empty' Email Body Artifact:** Forensic inspection of the 588 novel test emails revealed that **114 samples (19.39% of the novel partition)** consist solely of the uninformative literal string `'empty'`. Of these empty rows, **74 are labeled phishing** (accounting for **37.56% of all 197 novel phishing emails**) and 40 are labeled legitimate.
  - An empirical prediction audit across all models on these 114 empty rows (recorded in JSON metadata) revealed a binary divide: models such as Logistic Regression, LinearSVC, SGDClassifier, and Random Forest predicted class 1 (phishing) on all 114 empty rows, thereby capturing all 74 empty phishing emails by default.
  - In contrast, BiLSTM, TextCNN, AutoModel-DL, GRU, Hybrid, Voting Classifier, GaussianNB, and Multinomial NB predicted class 0 (legitimate) on all 114 empty rows, thus failing to detect any of the 74 empty phishing emails (0% recall on empty bodies). This single artifact explains exactly 74 missed phishing emails for these models.
* **Clean Novel Subset Evaluation (N = 474) and Ranking Shifts:** When these 114 uninformative empty email rows are excluded, model rankings and recall values shift substantially between the full and clean novel partitions, reflecting how each architecture handles empty sequences:
  - Models that defaulted to legitimate (0) on empty strings experience substantial recall gains on clean novel text: **BiLSTM** clean novel recall rises from 55.33% to **88.62%** (with 92.19% overall accuracy and a ROC-AUC of **0.9692**), **TextCNN** clean novel recall rises from 59.39% to **95.12%** (with 77.64% overall accuracy and a ROC-AUC of **0.9393**), and **Voting Classifier** rises from 55.84% to **89.43%** (with 91.14% overall accuracy and a ROC-AUC of **0.9704**).
  - However, the recall deficit does not universally disappear across sequence models: **AutoModel-DL** reaches only **78.05%** and **GRU** reaches only **82.11%** on clean novel emails. Even after removing empty inputs, these sequence architectures exhibit reduced sensitivity to novel phishing phrasing, indicating that vocabulary divergence beyond empty bodies continues to constrain their zero-shot detection.
  - Conversely, models that defaulted to phishing (1) on empty strings experience a recall decline when evaluated on clean novel text because the 74 empty phishing samples are no longer credited as true positives: **1D-CNN** recall drops from 84.77% to **75.61%** (-9.16 pp), **LSTM** drops from 87.82% to **80.49%** (-7.33 pp), **CNN-LSTM** drops from 81.73% to **70.73%** (-11.00 pp), and **Logistic Regression** drops from 95.94% to **93.50%** (-2.44 pp).
  Consequently, full novel recall is heavily confounded by model behavior on empty text, which fundamentally reshuffles model rankings between the full and clean novel partitions.
* **Discriminative Ranking Capacity Across Architectures:** Threshold-independent ranking capacity varies considerably across model families: while BiLSTM achieves a high novel ROC-AUC of **0.9195** (0.9692 clean) and LinearSVC achieves **0.9188** (0.9673 clean), other architectures such as TextCNN (0.7849 full / 0.9393 clean), LSTM (0.8384), and AutoModel-ML (0.8177) exhibit lower novel ranking scores, reflecting varying degrees of feature space degradation under domain transfer.
* **Confidence Interval Overlap and Statistical Equivalence:** Across the clean novel subset (N = 474), 95% Wilson confidence intervals span ±3 to 4 pp (BiLSTM: [89.4–94.3%], Voting Classifier: [88.2–93.4%], Logistic Regression: [85.4–91.2%], SGDClassifier: [81.3–87.8%]). Consequently, these top classifiers display overlapping uncertainty intervals rather than a statistically strict hierarchical superiority.
* **Robust Large-Scale Transfer (D1 -> D2):** On the Dataset 1 -> Dataset 2 novel partition (N = 11,191), where no empty rows exist, SGDClassifier achieves **88.87%** accuracy [95% CI: 88.3–89.4%] and **88.97%** recall (ROC-AUC 0.9467). Among deep architectures, TextCNN maintains **87.83%** accuracy [95% CI: 87.2–88.4%] and **90.56%** recall (ROC-AUC 0.9364), and AutoModel-DL achieves **86.13%** accuracy and **95.40%** recall (ROC-AUC 0.9234).

### 3. Representation Space Ablation: Lexical vs. Engineered Structural Features
The Optuna-tuned feature space ablation (Table 6e) provides conclusive evidence on the relative transferability of lexical versus structural representations:
* **Lexical TF-IDF Dominance:** Across both transfer directions, TF-IDF representations alone drive nearly all discriminative capacity. In D2 -> D1, TF-IDF achieves 95.41% accuracy and 0.9934 ROC-AUC for Logistic Regression. In D1 -> D2, TF-IDF achieves 88.46% accuracy and 0.9697 ROC-AUC.
* **Structural Feature Inefficacy:** Structural features alone (URL count, IP presence, urgency score, HTML flag) collapse toward majority-class baselines, achieving only 57.18% accuracy (ROC-AUC 0.6278) in D2 -> D1, and 48.00% accuracy (ROC-AUC 0.5174, with 0.00% phishing recall) in D1 -> D2.
* **Ranking Metric Ablation:** In the D1 -> D2 transfer direction, TF-IDF alone achieved slightly higher ROC-AUC than the Hybrid configuration for both Logistic Regression (0.9697 vs. 0.9674) and LinearSVC (0.9691 vs. 0.9672), confirming that incorporating engineered structural features does not improve cross-dataset ranking discriminability.

### 4. Intra-Dataset Train-Test Leakage and the 'Empty' Email Body Audit
A rigorous forensic audit of intra-dataset duplicate rows was conducted to assess whether duplicate leakage between training and test sets artificially inflated in-domain benchmarks:
* **Dataset 1 Conflicting Labels:** A global scan revealed 1 unique text string across 533 rows with conflicting binary labels (335 labeled phishing and 198 labeled legitimate). Specifically, all 533 conflicting rows contain the identical uninformative string `'empty'`. In the 20% stratified test split, 279 duplicate rows were present from the training set, of which **114 (40.86%)** were these identical empty email bodies with contradictory labels. Rather than representing semantic label ambiguity across distinct textual narratives, this finding reflects label noise arising from uninformative blank email bodies.
* **Impact of Test Set Deduplication:** Evaluating native Dataset 1 models on a strictly deduplicated clean test set (N = 3,448) revealed that in-domain accuracy actually **increased** by +0.81 to +0.97 pp across all classifiers (Logistic Regression: 97.10% -> 98.03%; LinearSVC: 97.05% -> 97.97%; SGDClassifier: 96.83% -> 97.80%). This result is consistent with the hypothesis that duplicate leakage did not artificially inflate in-domain accuracy; rather, contradictory labels on duplicate rows had imposed an artificial error ceiling. Note that standard in-domain test benchmarks for Dataset 1 still retain the 114 empty test rows, whereas the deduplicated evaluation isolates model performance from this noise.
* **Dataset 2 Integrity:** The same audit conducted on Dataset 2 confirmed 0 empty text rows and 0 conflicting label rows across all 82,485 samples, confirming full label consistency.

## Limitations and Threats to Validity
Several methodological boundaries should be considered when interpreting these findings:
1. **Token Pre-Processing Discrepancies:** An empirical quantification revealed substantial token normalization disparities between the benchmarks: Dataset 1 preserves common English stop words (29.63% of words) and punctuation characters (6.52% of characters), whereas Dataset 2 stripped both (6.61% stop words and 0.67% punctuation characters), resulting in a 4.5x stop-word and 9.7x punctuation density disparity. While target emails were evaluated through the source models' tokenizers/vectorizers, this pre-processing divergence is empirically identified as a likely contributing factor to the near-zero raw exact-string overlap (only 8 global matches across corpora, and 2 between train and test splits), alongside differences in corpus origin and content.
2. **In-Domain Optimization Failure of Specific Architectures:** Deep learning LSTM and Stacked LSTM models failed to converge during in-domain training on Dataset 1 (native ROC-AUC ≈ 0.52–0.53). Consequently, their low cross-dataset scores reflect in-domain optimization failure rather than domain shift transfer vulnerability.
3. **Zero-Shot Decision Thresholds:** All transfer evaluations were conducted strictly zero-shot at the default 0.5 decision threshold without post-hoc threshold adjustment or domain adaptation. As demonstrated by the clean novel evaluation, classification performance on out-of-distribution samples is highly sensitive to model behavior on uninformative inputs, with novel ROC-AUC scores spanning from 0.7849 to 0.9692 across deep models and 0.8796 to 0.9704 across classical ML models.
4. **Training Set Empty Rows in Dataset 1:** Dataset 1's training split contains 419 uninformative rows consisting of the literal string `'empty'` (on which models evaluated in the D1 -> D2 direction were trained). While Dataset 2's test partition contains no empty rows, the presence of these empty training rows in Dataset 1 reflects upstream dataset quality issues that may introduce noise into learned feature representations.
