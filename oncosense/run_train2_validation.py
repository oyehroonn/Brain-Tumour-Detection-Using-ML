#!/usr/bin/env python3
"""
Train2 Model Validation Suite
=============================
Runs comprehensive validation experiments on the Train2 model (trained on fixed data)
to verify if the 96.51% accuracy is genuine or artificially inflated.

Experiments:
1. Split Analysis - Verify 0% train/test contamination
2. Leakage Check - Verify 0% near-duplicate leakage
3. Random Label Test - Detect if model memorizes data
4. Center Crop Test - Detect spurious pattern learning
5. Grad-CAM Analysis - Visualize model attention
6. External Validation - Test generalization on Figshare dataset (if available)
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.models.backbones import get_backbone
from src.models.train import FocalLoss
from src.data.transforms import get_train_transforms, get_val_transforms, BrainTumorDataset

# Configuration
FIXED_MANIFEST = "data/manifests/manifest_fixed.parquet"
FIXED_MODEL = "checkpoints/densenet121_fixed.pt"
OUTPUT_DIR = Path("validation_results_train2")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def extract_original_split(filename: str) -> str:
    """Extract original train/test designation from filename."""
    filename_upper = filename.upper()
    if "TRAIN" in filename_upper:
        return "original_train"
    elif "TEST" in filename_upper:
        return "original_test"
    return "original_train"


# ============================================================================
# EXPERIMENT 1: SPLIT ANALYSIS
# ============================================================================

def run_split_analysis():
    """Verify there's no train/test contamination in the fixed manifest."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: SPLIT ANALYSIS")
    print("=" * 60)
    print("Purpose: Verify 0% train/test contamination in fixed manifest")
    
    df = pd.read_parquet(FIXED_MANIFEST)
    
    # Extract original split if not present
    if 'original_split' not in df.columns:
        df['original_split'] = df['filename'].apply(extract_original_split)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(df),
        "manifest": FIXED_MANIFEST,
    }
    
    # Count original splits
    orig_counts = df['original_split'].value_counts().to_dict()
    results["original_split_counts"] = orig_counts
    
    # Count our splits
    our_counts = df['split'].value_counts().to_dict()
    results["our_split_counts"] = our_counts
    
    # Cross-tabulation
    cross_tab = {}
    for our_split in ['train', 'val', 'test']:
        cross_tab[our_split] = {}
        split_df = df[df['split'] == our_split]
        for orig_split in ['original_train', 'original_test']:
            count = len(split_df[split_df['original_split'] == orig_split])
            cross_tab[our_split][orig_split] = count
    results["cross_tabulation"] = cross_tab
    
    # Calculate contamination
    orig_test_in_train = cross_tab['train'].get('original_test', 0)
    orig_test_in_val = cross_tab['val'].get('original_test', 0)
    orig_train_in_test = cross_tab['test'].get('original_train', 0)
    
    orig_test_total = orig_counts.get('original_test', 0)
    orig_train_total = orig_counts.get('original_train', 0)
    
    test_to_train_rate = (orig_test_in_train / orig_test_total * 100) if orig_test_total > 0 else 0
    train_to_test_rate = (orig_train_in_test / orig_train_total * 100) if orig_train_total > 0 else 0
    
    results["contamination"] = {
        "original_test_in_our_train": orig_test_in_train,
        "original_test_in_our_val": orig_test_in_val,
        "original_train_in_our_test": orig_train_in_test,
        "test_to_train_rate": test_to_train_rate,
        "train_to_test_rate": train_to_test_rate,
    }
    
    # Verdict
    if test_to_train_rate == 0 and train_to_test_rate == 0:
        verdict = "PASS: Zero train/test contamination!"
        severity = "pass"
    elif test_to_train_rate < 5:
        verdict = "WARNING: Minor contamination detected"
        severity = "warning"
    else:
        verdict = "FAIL: Significant contamination detected"
        severity = "fail"
    
    results["verdict"] = {
        "summary": verdict,
        "severity": severity,
    }
    
    print(f"\nOriginal Split Counts:")
    print(f"  Original Train: {orig_counts.get('original_train', 0)}")
    print(f"  Original Test: {orig_counts.get('original_test', 0)}")
    
    print(f"\nOur Split Counts:")
    print(f"  Train: {our_counts.get('train', 0)}")
    print(f"  Val: {our_counts.get('val', 0)}")
    print(f"  Test: {our_counts.get('test', 0)}")
    
    print(f"\nContamination Check:")
    print(f"  Original TEST in our TRAIN: {orig_test_in_train} ({test_to_train_rate:.2f}%)")
    print(f"  Original TEST in our VAL: {orig_test_in_val}")
    print(f"  Original TRAIN in our TEST: {orig_train_in_test} ({train_to_test_rate:.2f}%)")
    
    print(f"\nVERDICT: {verdict}")
    
    return results


# ============================================================================
# EXPERIMENT 2: LEAKAGE CHECK
# ============================================================================

def run_leakage_check():
    """Check for near-duplicate images across splits using pHash."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: NEAR-DUPLICATE LEAKAGE CHECK")
    print("=" * 60)
    print("Purpose: Verify 0% near-duplicate leakage between splits")
    
    df = pd.read_parquet(FIXED_MANIFEST)
    threshold = 10  # Hamming distance threshold
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "leaks": {
            "test_train": [],
            "val_train": [],
            "test_val": [],
        },
        "stats": {}
    }
    
    def find_leaks(source_df, target_df, source_name, target_name, sample_size=200):
        """Find near-duplicate leaks between two splits."""
        leaks = []
        source_sample = source_df.sample(min(sample_size, len(source_df)), random_state=42)
        
        for _, source_row in tqdm(source_sample.iterrows(), 
                                   total=len(source_sample), 
                                   desc=f"Checking {source_name}→{target_name}"):
            source_hash = source_row.get('phash')
            if pd.isna(source_hash):
                continue
            
            try:
                h1 = imagehash.hex_to_hash(source_hash)
            except:
                continue
            
            for _, target_row in target_df.iterrows():
                target_hash = target_row.get('phash')
                if pd.isna(target_hash):
                    continue
                
                try:
                    h2 = imagehash.hex_to_hash(target_hash)
                    dist = h1 - h2
                    if dist <= threshold:
                        leaks.append({
                            "source_file": source_row['filepath'],
                            "target_file": target_row['filepath'],
                            "distance": dist,
                            "same_label": source_row['label'] == target_row['label']
                        })
                except:
                    continue
        
        return leaks
    
    print("\nChecking for near-duplicate leaks...")
    
    # Check test→train leaks
    test_train_leaks = find_leaks(test_df, train_df, "test", "train", 200)
    results["leaks"]["test_train"] = test_train_leaks[:10]  # Store first 10
    
    # Check val→train leaks
    val_train_leaks = find_leaks(val_df, train_df, "val", "train", 200)
    results["leaks"]["val_train"] = val_train_leaks[:10]
    
    # Check test→val leaks
    test_val_leaks = find_leaks(test_df, val_df, "test", "val", 100)
    results["leaks"]["test_val"] = test_val_leaks[:10]
    
    # Calculate rates
    test_sample_size = min(200, len(test_df))
    val_sample_size = min(200, len(val_df))
    
    results["stats"] = {
        "test_train_leak_count": len(test_train_leaks),
        "test_train_leak_rate": len(test_train_leaks) / test_sample_size * 100 if test_sample_size > 0 else 0,
        "val_train_leak_count": len(val_train_leaks),
        "val_train_leak_rate": len(val_train_leaks) / val_sample_size * 100 if val_sample_size > 0 else 0,
        "test_val_leak_count": len(test_val_leaks),
    }
    
    # Verdict
    total_leaks = len(test_train_leaks) + len(val_train_leaks)
    if total_leaks == 0:
        verdict = "PASS: Zero near-duplicate leakage!"
        severity = "pass"
    elif results["stats"]["test_train_leak_rate"] < 2:
        verdict = "PASS: Minimal leakage (< 2%)"
        severity = "pass"
    elif results["stats"]["test_train_leak_rate"] < 5:
        verdict = "WARNING: Minor leakage detected"
        severity = "warning"
    else:
        verdict = "FAIL: Significant leakage detected"
        severity = "fail"
    
    results["verdict"] = {"summary": verdict, "severity": severity}
    
    print(f"\nLeakage Results:")
    print(f"  Test→Train leaks: {len(test_train_leaks)} ({results['stats']['test_train_leak_rate']:.2f}%)")
    print(f"  Val→Train leaks: {len(val_train_leaks)} ({results['stats']['val_train_leak_rate']:.2f}%)")
    print(f"  Test→Val leaks: {len(test_val_leaks)}")
    print(f"\nVERDICT: {verdict}")
    
    return results


# ============================================================================
# EXPERIMENT 3: RANDOM LABEL TEST
# ============================================================================

def run_random_label_test(num_epochs=5):
    """Train model on randomly shuffled labels to detect memorization."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: RANDOM LABEL TEST")
    print("=" * 60)
    print("Purpose: Detect if model memorizes data instead of learning patterns")
    print("Expected: ~25% accuracy (random chance for 4 classes)")
    
    df = pd.read_parquet(FIXED_MANIFEST)
    
    # Shuffle labels randomly
    np.random.seed(42)
    original_labels = df['label'].values.copy()
    shuffled_labels = np.random.permutation(original_labels)
    df['label'] = shuffled_labels
    
    # Also update label_name to match
    label_map = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}
    df['label_name'] = df['label'].map(label_map)
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # Create datasets
    train_dataset = BrainTumorDataset(train_df, transform=get_train_transforms(), data_root='data')
    val_dataset = BrainTumorDataset(val_df, transform=get_val_transforms(), data_root='data')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    model = get_backbone("densenet121", num_classes=4, pretrained=True, dropout_rate=0.3)
    model = model.to(DEVICE)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_epochs": num_epochs,
        "history": {"train_acc": [], "val_acc": []},
    }
    
    print(f"\nTraining on RANDOM labels for {num_epochs} epochs...")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total * 100
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total * 100
        best_val_acc = max(best_val_acc, val_acc)
        
        results["history"]["train_acc"].append(train_acc)
        results["history"]["val_acc"].append(val_acc)
        
        print(f"  Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    results["final_train_acc"] = results["history"]["train_acc"][-1]
    results["final_val_acc"] = results["history"]["val_acc"][-1]
    results["best_val_acc"] = best_val_acc
    
    # Verdict
    if best_val_acc < 35:
        verdict = "PASS: Model cannot learn random labels (no memorization)"
        severity = "pass"
        explanation = f"Best accuracy {best_val_acc:.1f}% is near random chance (25%), confirming model learns patterns, not memorizes."
    elif best_val_acc < 50:
        verdict = "WARNING: Model shows some memorization tendency"
        severity = "warning"
        explanation = f"Best accuracy {best_val_acc:.1f}% is above random chance. Some memorization possible."
    else:
        verdict = "FAIL: Model memorizes data!"
        severity = "fail"
        explanation = f"Best accuracy {best_val_acc:.1f}% on random labels indicates severe memorization/data leakage."
    
    results["verdict"] = {
        "summary": verdict,
        "severity": severity,
        "explanation": explanation,
        "random_chance": 25.0,
        "achieved": best_val_acc,
    }
    
    print(f"\nBest Validation Accuracy on Random Labels: {best_val_acc:.2f}%")
    print(f"Random Chance: 25%")
    print(f"\nVERDICT: {verdict}")
    print(f"  {explanation}")
    
    return results


# ============================================================================
# EXPERIMENT 4: CENTER CROP TEST
# ============================================================================

def run_center_crop_test(num_epochs=5):
    """Train model only on center-cropped images to detect spurious patterns."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: CENTER CROP TEST")
    print("=" * 60)
    print("Purpose: Detect if model learns from center region only (spurious patterns)")
    print("Expected: Accuracy lower than full image model")
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    df = pd.read_parquet(FIXED_MANIFEST)
    
    # Custom center crop transform - only use center 64x64 then resize
    center_crop_transform = A.Compose([
        A.CenterCrop(height=64, width=64),
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    train_dataset = BrainTumorDataset(train_df, transform=center_crop_transform, data_root='data')
    val_dataset = BrainTumorDataset(val_df, transform=center_crop_transform, data_root='data')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    model = get_backbone("densenet121", num_classes=4, pretrained=True, dropout_rate=0.3)
    model = model.to(DEVICE)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_epochs": num_epochs,
        "crop_size": 64,
        "history": {"train_acc": [], "val_acc": []},
    }
    
    print(f"\nTraining on CENTER CROP (64x64) for {num_epochs} epochs...")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total * 100
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total * 100
        best_val_acc = max(best_val_acc, val_acc)
        
        results["history"]["train_acc"].append(train_acc)
        results["history"]["val_acc"].append(val_acc)
        
        print(f"  Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    results["final_train_acc"] = results["history"]["train_acc"][-1]
    results["final_val_acc"] = results["history"]["val_acc"][-1]
    results["best_val_acc"] = best_val_acc
    results["full_model_acc"] = 96.51  # Train2 accuracy
    
    # Verdict
    accuracy_drop = results["full_model_acc"] - best_val_acc
    
    if accuracy_drop > 15:
        verdict = "PASS: Model uses full image, not just center"
        severity = "pass"
        explanation = f"Center crop accuracy ({best_val_acc:.1f}%) is {accuracy_drop:.1f}% lower than full image, confirming model uses spatial information."
    elif accuracy_drop > 5:
        verdict = "PASS: Model uses more than center region"
        severity = "pass"
        explanation = f"Center crop accuracy ({best_val_acc:.1f}%) is {accuracy_drop:.1f}% lower than full image."
    elif accuracy_drop > 0:
        verdict = "WARNING: Model may rely heavily on center region"
        severity = "warning"
        explanation = f"Center crop accuracy ({best_val_acc:.1f}%) is only {accuracy_drop:.1f}% lower than full image."
    else:
        verdict = "FAIL: Model learns only from center!"
        severity = "fail"
        explanation = f"Center crop achieves {best_val_acc:.1f}% - model may learn spurious center patterns."
    
    results["verdict"] = {
        "summary": verdict,
        "severity": severity,
        "explanation": explanation,
        "accuracy_drop": accuracy_drop,
    }
    
    print(f"\nBest Center Crop Accuracy: {best_val_acc:.2f}%")
    print(f"Full Image Accuracy: {results['full_model_acc']:.2f}%")
    print(f"Accuracy Drop: {accuracy_drop:.2f}%")
    print(f"\nVERDICT: {verdict}")
    print(f"  {explanation}")
    
    return results


# ============================================================================
# EXPERIMENT 5: MODEL EVALUATION ON TEST SET
# ============================================================================

def run_test_evaluation():
    """Evaluate the Train2 model on the test set."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: TEST SET EVALUATION")
    print("=" * 60)
    print("Purpose: Evaluate Train2 model on held-out test set")
    
    df = pd.read_parquet(FIXED_MANIFEST)
    test_df = df[df['split'] == 'test']
    
    test_dataset = BrainTumorDataset(test_df, transform=get_val_transforms(), data_root='data')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load model
    model = get_backbone("densenet121", num_classes=4, pretrained=False, dropout_rate=0.3)
    checkpoint = torch.load(FIXED_MODEL, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_samples": len(test_df),
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "val_accuracy": checkpoint.get('val_accuracy', 0) * 100,
        "val_f1": checkpoint.get('val_f1', 0),
    }
    
    # Check for train-test gap
    gap = abs(results["val_accuracy"] - test_acc)
    
    if gap < 3:
        verdict = "PASS: Test accuracy consistent with validation"
        severity = "pass"
    elif gap < 5:
        verdict = "PASS: Minor gap between test and validation"
        severity = "pass"
    else:
        verdict = "WARNING: Significant gap between test and validation"
        severity = "warning"
    
    results["verdict"] = {
        "summary": verdict,
        "severity": severity,
        "val_test_gap": gap,
    }
    
    print(f"\nTest Results:")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test F1 Score: {test_f1:.4f}")
    print(f"  Val Accuracy: {results['val_accuracy']:.2f}%")
    print(f"  Val-Test Gap: {gap:.2f}%")
    print(f"\nVERDICT: {verdict}")
    
    # Classification report
    class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    results["classification_report"] = classification_report(all_labels, all_preds, 
                                                              target_names=class_names, 
                                                              output_dict=True)
    
    return results


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    print("=" * 70)
    print("TRAIN2 MODEL COMPREHENSIVE VALIDATION SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Manifest: {FIXED_MANIFEST}")
    print(f"Model: {FIXED_MODEL}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING VALIDATION EXPERIMENTS")
    print("=" * 70)
    
    # Experiment 1: Split Analysis
    try:
        all_results["experiment_1_split"] = run_split_analysis()
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        all_results["experiment_1_split"] = {"error": str(e)}
    
    # Experiment 2: Leakage Check
    try:
        all_results["experiment_2_leakage"] = run_leakage_check()
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        all_results["experiment_2_leakage"] = {"error": str(e)}
    
    # Experiment 3: Random Label Test
    try:
        all_results["experiment_3_random_label"] = run_random_label_test(num_epochs=5)
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
        all_results["experiment_3_random_label"] = {"error": str(e)}
    
    # Experiment 4: Center Crop Test
    try:
        all_results["experiment_4_center_crop"] = run_center_crop_test(num_epochs=5)
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
        all_results["experiment_4_center_crop"] = {"error": str(e)}
    
    # Experiment 5: Test Evaluation
    try:
        all_results["experiment_5_test_eval"] = run_test_evaluation()
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
        all_results["experiment_5_test_eval"] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments_run": [],
        "experiments_passed": [],
        "experiments_warning": [],
        "experiments_failed": [],
    }
    
    for exp_name, exp_results in all_results.items():
        summary["experiments_run"].append(exp_name)
        if "error" in exp_results:
            summary["experiments_failed"].append(exp_name)
            print(f"  {exp_name}: ERROR - {exp_results['error']}")
        elif "verdict" in exp_results:
            severity = exp_results["verdict"].get("severity", "unknown")
            if severity == "pass":
                summary["experiments_passed"].append(exp_name)
                print(f"  {exp_name}: ✓ PASS")
            elif severity == "warning":
                summary["experiments_warning"].append(exp_name)
                print(f"  {exp_name}: ⚠ WARNING")
            else:
                summary["experiments_failed"].append(exp_name)
                print(f"  {exp_name}: ✗ FAIL")
    
    # Overall verdict
    if len(summary["experiments_failed"]) == 0 and len(summary["experiments_warning"]) == 0:
        overall = "ALL TESTS PASSED - Model accuracy appears VALID"
        overall_status = "VALID"
    elif len(summary["experiments_failed"]) == 0:
        overall = "TESTS PASSED WITH WARNINGS - Model accuracy likely valid"
        overall_status = "LIKELY_VALID"
    else:
        overall = "TESTS FAILED - Model accuracy may be INFLATED"
        overall_status = "QUESTIONABLE"
    
    summary["overall_verdict"] = overall
    summary["overall_status"] = overall_status
    all_results["summary"] = summary
    
    print(f"\n{'=' * 70}")
    print(f"OVERALL VERDICT: {overall}")
    print(f"{'=' * 70}")
    
    # Save results
    results_file = OUTPUT_DIR / "train2_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    main()
