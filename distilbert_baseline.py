# -*- coding: utf-8 -*-
"""
DistilBERT External Baseline Benchmark Script
Fine-tunes HuggingFace DistilBERT (distilbert-base-uncased) on RTX 5070 GPU for phishing detection
as an external pre-trained Transformer baseline for thesis comparison.
"""

import os
import sys
import json
import io
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Force UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] DistilBERT Baseline PyTorch Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

RUN_DIR = "thesis_pipeline/DL/runs/run_2026-08-05_18-37-42"
CLEANED_DATA_PATH = os.path.join(RUN_DIR, "cleaned_dataset.csv")
OUTPUT_JSON = "thesis_pipeline/plots/distilbert_baseline_results.json"
OUTPUT_PLOT = "thesis_pipeline/plots/external_baseline_comparison.png"
os.makedirs("thesis_pipeline/plots", exist_ok=True)

class PhishingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def main():
    print("\n--- Starting DistilBERT External Baseline Benchmark ---")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[ERROR] HuggingFace 'transformers' is not installed. Installing via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "accelerate"])
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("\nLoading cleaned dataset...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded dataset: {len(df)} samples.")
    
    text_col = 'text' if 'text' in df.columns else ('text_combined' if 'text_combined' in df.columns else df.columns[0])
    label_col = 'label' if 'label' in df.columns else ('Email Type' if 'Email Type' in df.columns else df.columns[1])
    
    # Stratified 80/20 train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df[text_col].astype(str).tolist(),
        df[label_col].astype(int).tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )
    
    model_name = "distilbert-base-uncased"
    print(f"\nInitializing AutoTokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Tokenizing train and test sets (max_length=256)...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)
    
    train_dataset = PhishingDataset(train_encodings, train_labels)
    test_dataset = PhishingDataset(test_encodings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"\nInitializing Model ({model_name})...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print("\nTraining DistilBERT on RTX 5070 GPU (2 epochs with Mixed Precision AMP)...")
    start_time = time.time()
    
    model.train()
    for epoch in range(2):
        total_loss = 0
        t_ep = time.time()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                print(f"  [Epoch {epoch+1}/2] Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        print(f"  [Epoch {epoch+1}/2] Complete | Avg Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t_ep:.2f}s")
        
    training_time_s = round(time.time() - start_time, 2)
    print(f"Training completed in {training_time_s} seconds.")
    
    # GPU Memory Tracking
    vram_used_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2) if device.type == 'cuda' else 0.0
    
    # Evaluation
    print("\nEvaluating DistilBERT on test set...")
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            
    acc = float(accuracy_score(test_labels, all_preds))
    prec = float(precision_score(test_labels, all_preds, zero_division=0))
    rec = float(recall_score(test_labels, all_preds, zero_division=0))
    f1 = float(f1_score(test_labels, all_preds, zero_division=0))
    auc = float(roc_auc_score(test_labels, all_probs))
    
    results = {
        "model_name": "DistilBERT (distilbert-base-uncased)",
        "type": "External Baseline (HuggingFace Pre-trained Transformer)",
        "parameters": "66,364,418 (66M)",
        "model_size_mb": 260,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc": round(auc, 4),
        "training_time_seconds": training_time_s,
        "max_vram_used_mb": vram_used_mb
    }
    
    print("\n--- DISTILBERT BASELINE RESULTS ---")
    print(f"  Accuracy:       {acc*100:.2f}%")
    print(f"  Precision:      {prec*100:.2f}%")
    print(f"  Recall:         {rec*100:.2f}%")
    print(f"  F1-Score:       {f1*100:.2f}%")
    print(f"  ROC-AUC:        {auc:.4f}")
    print(f"  Training Time:  {training_time_s}s")
    print(f"  Max VRAM Used:  {vram_used_mb} MB")
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] DistilBERT baseline completed! Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
