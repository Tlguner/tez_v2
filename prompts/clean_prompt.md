You are a senior data engineer.
The raw dataset is located at: '${dataset_path}'
Analyze the following dataset metadata:
Total rows: ${total_rows}
EDA Summary: ${eda_summary}

TASK:
- You MUST load and clean the ENTIRE actual dataset. DO NOT generate any fake, miniature, or artificial datasets in the script.
- Automatically identify the text column (email content) and the label column (phishing vs. safe).
- Write a dynamically tailored Python script that cleans the dataset.
- Ignore all other columns in the dataset, but extract 4 metadata features directly from the raw text column BEFORE text cleaning/normalization:
  - 'url_count': Total count of URLs/links (e.g., matching 'http', 'https', 'www', or domain extensions like '.com', '.org', '.net') in the email raw text.
  - 'has_ip': 1 if the email text contains an IP address pattern (e.g., matching a standard IPv4 regex like '\\b\\d{1,3}\\.\\\d{1,3}\\.\\\d{1,3}\\.\\\d{1,3}\\b'), otherwise 0.
  - 'urgency_score': Total frequency count of threat/urgency keywords (case-insensitive) like: 'verify', 'suspend', 'update', 'password', 'urgent', 'immediate', 'security', 'warning', 'action required', 'alert'.
  - 'has_html': 1 if the email text contains HTML markers or tags (e.g., '<', '>', 'href', 'html', 'div', 'img'), otherwise 0.
- Ensure the label column is mapped to binary integer values (1 for phishing/spam variants, 0 for safe/ham variants).
- **TYPE SAFETY RULE**: When performing string checks, filtering, or using the `in` operator on columns, first ensure the values are cast to string using `.astype(str)` or check if the value is NaN to prevent `TypeError: 'in <string>' requires string as left operand, not int`.
- Drop unmappable rows.
- Save the clean DataFrame (containing the columns: 'text', 'label', 'url_count', 'has_ip', 'urgency_score', 'has_html') strictly to: '${cleaned_dataset_path}'
- Write a try-except block.
- At the very end of successful execution, output EXACTLY this JSON format between <<<JSON_START>>> and <<<JSON_END>>>:
{
  "ok": true,
  "original_shape": [rows, cols],
  "cleaned_shape": [rows, 6],
  "cleaned_dataset_path": "${cleaned_dataset_path}",
  "class_balance": {"0": count0, "1": count1}
}
- Do NOT write markdown fences other than ```python. Output only python code.
