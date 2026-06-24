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

MODEL LOADING AND PREDICTION RULES:
- ALWAYS use the defined "Models Directory" ("${models_dir}") path. DO NOT attempt to load models from other directories (e.g., the directory where the test features file is located).
- If Mode is 'ML':
  - Models are loaded from "${models_dir}/[model_name].pkl". Use `joblib` for loading (`import joblib` and `joblib.load()`).
- If Mode is 'DL':
  - Models are loaded from "${models_dir}/[model_name].h5". Use `tensorflow.keras.models.load_model()`.
  - CRITICAL (loading auto_model): Since a custom `BahdanauAttention` layer is used in the `auto_model.h5` model, you must define this layer class in the evaluation script and pass `custom_objects={'BahdanauAttention': BahdanauAttention}` when loading the model.
    Add the following BahdanauAttention class definition to the beginning of the script:
    ```python
    import tensorflow as tf
    class BahdanauAttention(tf.keras.layers.Layer):
        def __init__(self, units, **kwargs):
            super(BahdanauAttention, self).__init__(**kwargs)
            self.units = units
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            query_with_time_axis = tf.expand_dims(query, 1)
            score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector

        def get_config(self):
            config = super(BahdanauAttention, self).get_config()
            config.update({"units": self.units})
            return config
    ```
  - The model output size for binary classification can be either 2 columns (Softmax) or 1 column (Sigmoid). If the output has 2 columns, take the 1st column (phishing probability, e.g., `preds[:, 1]`). If it has 1 column, use it directly.
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
     - Plot a single large comparative graph containing subplots of the Confusion Matrices for all models (including the best-performing model) and save it to the "${cm_plot_path}" path (e.g., in a 2x3 grid layout).
     - In addition, plot individual Confusion Matrix graphs for each model and save them as separate files named f"{os.path.dirname(cm_plot_path)}/confusion_matrix_{model_name}.png" (e.g., `confusion_matrix_lstm.png`).
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
