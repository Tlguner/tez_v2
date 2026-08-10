Write a Python script that uses the training results of the models and the test dataset information to plot the ROC curve, Confusion Matrix, hardware resource consumption, and F1/Accuracy/Precision/Recall comparison graphs.

- Pipeline Mode: "${mode}"
- Trained Models: ${models_trained}
- Test Features Path: "${test_features_path}"
- Test Labels Path: "${test_labels_path}"
- Models Directory: "${models_dir}"
- ROC Data JSON Path: "${roc_json_path}"
- Models Training Metrics: ${models_metrics}
- Resource Plot Path: "${resource_plot_path}"

TEST DATA LOADING RULES:
- When loading Test Features ("${test_features_path}"):
  - If Mode is 'ML': This file is a scipy sparse matrix (in NPZ format). Load it using `scipy.sparse.load_npz("${test_features_path}")`. FOR RESOURCE AND MEMORY EFFICIENCY, DO NOT CONVERT IT TO A DENSE NUMPY ARRAY WITH `.toarray()`. Scikit-learn models natively support sparse matrices as input and generate predictions directly.
  - If Mode is 'DL': This file is a dense numpy array (in NPY format). Load it using `np.load("${test_features_path}")`.
- When loading Test Labels ("${test_labels_path}"):
  - For both modes, this file is a numpy array (in NPY format). Load it using `np.load("${test_labels_path}")`.
  - Flatten the labels to form a 1D integer array: `y_test = np.ravel(y_test).astype(int)`.

- MODEL LOADING AND PREDICTION RULES:
- ALWAYS use the defined "Models Directory" ("${models_dir}") path. DO NOT attempt to load models from other directories (e.g., the directory where the test features file is located).
- If Mode is 'ML':
  - Models are loaded from "${models_dir}/[model_name].pkl". Use `joblib` for loading (`import joblib` and `joblib.load()`).
  - CRITICAL: Both `GaussianNB` and `auto_model` (HistGradientBoostingClassifier) do NOT support sparse matrices. In your evaluation loop, you MUST check if the model name is `"GaussianNB"` or `"auto_model"` and convert `X_test` to dense using `.toarray()` before passing it to `.predict()` or `.predict_proba()`.
    Example structure for ML prediction loop:
    ```python
    for model_name in models_trained:
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        model = joblib.load(model_path)
        
        # Check if model requires dense input
        if model_name in ["GaussianNB", "auto_model"]:
            X_eval = X_test.toarray()
        else:
            X_eval = X_test
            
        preds = model.predict(X_eval)
        # and similarly for predict_proba if needed...
    ```
- If Mode is 'DL':
  - Models are trained and saved using **Keras/TensorFlow** at `${models_dir}/[model_name].h5`.
  - CRITICAL: Add this block at the TOP of your script (before any other imports):
    ```python
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow import keras
    ```
  - CRITICAL: The `auto_model` may use custom Keras layers. To handle unknown layers gracefully, wrap load_model with custom_objects and a fallback:
    ```python
    def load_dl_model(model_path):
        try:
            return keras.models.load_model(model_path)
        except Exception:
            # Fallback: try with compile=False for models with custom metrics
            return keras.models.load_model(model_path, compile=False)
    ```
  - Load each model: `model = load_dl_model(model_path)` then `model.trainable = False`.
  - For predictions, process in batches of 512:
    ```python
    proba = model.predict(X_test, batch_size=512, verbose=0).ravel()
    preds = (proba >= 0.5).astype(int)
    ```
  - Use `proba` directly as positive-class score for ROC AUC computation: `roc_curve(y_true, proba)`.
  - CRITICAL WINDOWS: Do NOT use `multiprocessing` or `threading` to parallelize predictions. Run sequentially.
- ERROR HANDLING:
  - If an error occurs during model loading or prediction, DO NOT silently generate mock predictions! Raise the exception or exit the script with a non-zero exit code so that the self-healing mechanism can detect and fix the issue.


GRAPHICAL AND JSON REQUIREMENTS:
- IMPORTANT: All text, titles, labels, legends, axis markings, and ticks in the generated plots MUST be written in English (suitable for international academic journals).
- For composite/ensemble models (like `VotingClassifier`, `hybrid`) and AI-designed models (like `auto_model`), dynamically format their names in the plots and JSON outputs (under both keys and `model_name` fields) to include their underlying sub-models or layers in parentheses (e.g., `VotingClassifier (LR + RF + SVC)` or `auto_model (CNN + BiGRU)`).

TASKS:
1. For each model (including ML-Hybrid Ensemble and ML-AutoModel), calculate ROC (tpr, fpr) values on the test dataset.
2. Save the outputs into two separate JSON files:
   - **Metrics Summary JSON** ("${results_json_path}"): A clean, lightweight JSON that saves only the general performance metrics of the models (`model_name`, `accuracy`, `precision`, `recall`, `f1_score`, `auc`, `time_s`, `avg_cpu_percent`, `max_cpu_percent`, `avg_ram_mb`, `max_ram_mb`, and `description` containing the structural descriptions of the models). The `model_name` used as keys or values in this JSON must be the formatted detailed names (like `VotingClassifier (LR + RF + SVC)` or `auto_model (CNN + BiGRU)`), and the `description` field must explicitly state their underlying architectures/constituent classifiers. This file MUST NOT contain the large `fpr` and `tpr` lists! Read the models' `time_s` and hardware utilization data from `${models_metrics}`.
   - **ROC Curve Raw Data JSON** ("${roc_json_path}"): A JSON that associates and saves the large `fpr` and `tpr` float lists with model names, which are needed to plot the ROC curve. The model name keys in this JSON must also use the formatted detailed names.
3. Plotting Graphs:
   - Save the ROC curves of all models in a single plot as "${roc_plot_path}".
    - Confusion Matrix Plots:
      - Plot a single large comparative graph containing subplots of the Confusion Matrices for all models (including the best-performing model) and save it to the "${cm_plot_path}" path.
      - In addition, plot individual Confusion Matrix graphs for each model and save them as separate files. 
         **CRITICAL FILENAME SANITIZATION RULE**: To prevent Windows MAX_PATH (260-character) path length limits or invalid filename character errors, the model name used in the individual confusion matrix filename MUST be sanitized. Strip out any parenthetical details, layer specifications, spaces, parentheses, arrows, brackets, and keep ONLY a short alphanumeric identifier (e.g., use `confusion_matrix_auto_model.png`, `confusion_matrix_TextCNN.png`, or `confusion_matrix_HybridCNN.png`). Save them as f"{os.path.dirname(cm_plot_path)}/confusion_matrix_{clean_model_name}.png".
      - **CRITICAL CONFUSION MATRIX STYLING RULES (Academic Standard)**:
        1. **Turn off Gridlines**: You MUST call `ax.grid(False)` or `plt.grid(False)` on each confusion matrix subplot/figure. Failing to do this creates an ugly "grid-inside-grid" effect over the matrix cells.
        2. **Normalization & Color Bar Consistency**: Always plot the heatmap using the **normalized** confusion matrix (percentages between `0.0` and `1.0` calculated row-wise: `cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]`). Set `vmin=0.0` and `vmax=1.0` on the heatmap. This ensures that the colorbar scale is always between 0% and 100% (0.0 to 1.0) and looks identical across all models, rather than showing raw counts stretching to 2000.
        3. **Cell Annotations (Prevent Overlaps)**: Display **both** the raw count and the percentage inside each cell, formatted as: `f"{count}\n({pct:.1%})"` (e.g., `1850\n(98.5%)` or `35\n(1.5%)`).
           - **CRITICAL**: To prevent text overlaps, DO NOT pass `annot=True` to `sns.heatmap` and DO NOT write manual loops (like nested `for` loops with `ax.text` or `plt.text`) to draw numbers. Instead, create a 2D list of formatted strings:
             `annot_labels = np.array([[f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})" for j in range(2)] for i in range(2)])`
             And pass it directly as the `annot` parameter: `sns.heatmap(cm_normalized, annot=annot_labels, fmt="", ...)`
        4. **Labels**: Use clear class labels on the axes: `["Safe", "Phishing"]`.
   - Plot a grouped bar chart comparing Accuracy, F1-Score, Precision, and Recall values of all models side-by-side and save it as "${performance_plot_path}".
   - Plot a bar chart comparing models' training times (`time_s`) in seconds and save it as "${time_plot_path}" (read time data from `${models_metrics}`).
   - Plot a dual-subplot bar chart comparing models' CPU and RAM resource consumptions and save it as "${resource_plot_path}" (read hardware usage data from `${models_metrics}`). The left subplot must compare CPU (Average and Maximum %), and the right subplot must compare RAM (Average and Maximum MB).
4. Print only the following to stdout:
<<<JSON_START>>>
{
  "roc_plot_path": "${roc_plot_path}",
  "cm_plot_path": "${cm_plot_path}",
  "performance_plot_path": "${performance_plot_path}",
  "time_plot_path": "${time_plot_path}",
  "resource_plot_path": "${resource_plot_path}",
  "results_json_path": "${results_json_path}",
  "roc_json_path": "${roc_json_path}"
}
<<<JSON_END>>>
Do not print anything else.
