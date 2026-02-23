#!/usr/bin/env python3
"""
Experiment 7: Original Train/Test Split Analysis

Analyzes whether our random split violated the original train/test split 
encoded in the Kaggle dataset filenames.

Filename patterns:
    - "BT-MRI GL Train (808).jpg"  → Original: Train
    - "BT-MRI Test GL (288).jpg"   → Original: Test

If original "Test" images ended up in our train split, this could explain
artificially high accuracy (train/test contamination).
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR


def extract_original_split(filename: str) -> str:
    """
    Extract the original train/test designation from filename.
    
    Patterns observed:
    - "BT-MRI GL Train (808).jpg" -> "train"
    - "BT-MRI Test GL (288).jpg" -> "test"
    - "BT-MRI  ME Train (985).jpg" -> "train"
    - "BT-MRI PI Test (1).jpg" -> "test"
    """
    filename_lower = filename.lower()
    
    if ' train ' in filename_lower or ' train(' in filename_lower or 'train ' in filename_lower:
        return 'original_train'
    elif ' test ' in filename_lower or ' test(' in filename_lower or 'test ' in filename_lower:
        return 'original_test'
    else:
        return 'unknown'


def analyze_split_contamination(manifest_path: str = "data/manifests/manifest.parquet") -> dict:
    """
    Analyze how original train/test split maps to our split.
    
    Returns:
        Dictionary with analysis results.
    """
    print("=" * 60)
    print("EXPERIMENT 7: Original Train/Test Split Analysis")
    print("=" * 60)
    
    df = pd.read_parquet(manifest_path)
    
    # Extract original split from filename
    df['original_split'] = df['filename'].apply(extract_original_split)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(df),
        'original_split_counts': {},
        'our_split_counts': {},
        'cross_tabulation': {},
        'contamination_analysis': {},
        'verdict': None
    }
    
    # Count original splits
    original_counts = df['original_split'].value_counts().to_dict()
    results['original_split_counts'] = original_counts
    print(f"\nOriginal split distribution (from filenames):")
    for split, count in original_counts.items():
        print(f"  {split}: {count} ({count/len(df)*100:.1f}%)")
    
    # Count our splits
    our_counts = df['split'].value_counts().to_dict()
    results['our_split_counts'] = our_counts
    print(f"\nOur split distribution:")
    for split, count in our_counts.items():
        print(f"  {split}: {count} ({count/len(df)*100:.1f}%)")
    
    # Cross-tabulation
    cross_tab = pd.crosstab(df['original_split'], df['split'])
    results['cross_tabulation'] = cross_tab.to_dict()
    
    print(f"\nCross-tabulation (Original vs Our Split):")
    print(cross_tab.to_string())
    
    # Contamination analysis
    print("\n" + "-" * 60)
    print("CONTAMINATION ANALYSIS")
    print("-" * 60)
    
    # Original test images that ended up in our train
    orig_test_in_our_train = len(df[(df['original_split'] == 'original_test') & 
                                     (df['split'] == 'train')])
    orig_test_total = len(df[df['original_split'] == 'original_test'])
    
    # Original train images that ended up in our test
    orig_train_in_our_test = len(df[(df['original_split'] == 'original_train') & 
                                     (df['split'] == 'test')])
    orig_train_total = len(df[df['original_split'] == 'original_train'])
    
    # Calculate contamination rates
    if orig_test_total > 0:
        test_to_train_rate = orig_test_in_our_train / orig_test_total * 100
    else:
        test_to_train_rate = 0
        
    if orig_train_total > 0:
        train_to_test_rate = orig_train_in_our_test / orig_train_total * 100
    else:
        train_to_test_rate = 0
    
    results['contamination_analysis'] = {
        'original_test_in_our_train': orig_test_in_our_train,
        'original_test_total': orig_test_total,
        'test_to_train_contamination_rate': test_to_train_rate,
        'original_train_in_our_test': orig_train_in_our_test,
        'original_train_total': orig_train_total,
        'train_to_test_contamination_rate': train_to_test_rate,
    }
    
    print(f"\nOriginal TEST images in our TRAIN split:")
    print(f"  {orig_test_in_our_train} / {orig_test_total} = {test_to_train_rate:.1f}%")
    
    print(f"\nOriginal TRAIN images in our TEST split:")
    print(f"  {orig_train_in_our_test} / {orig_train_total} = {train_to_test_rate:.1f}%")
    
    # Per-class analysis
    print("\n" + "-" * 60)
    print("PER-CLASS CONTAMINATION")
    print("-" * 60)
    
    per_class_contamination = {}
    for class_name in df['label_name'].unique():
        class_df = df[df['label_name'] == class_name]
        class_orig_test = class_df[class_df['original_split'] == 'original_test']
        class_in_train = len(class_orig_test[class_orig_test['split'] == 'train'])
        total_orig_test = len(class_orig_test)
        
        if total_orig_test > 0:
            rate = class_in_train / total_orig_test * 100
        else:
            rate = 0
            
        per_class_contamination[class_name] = {
            'original_test_in_our_train': class_in_train,
            'original_test_total': total_orig_test,
            'contamination_rate': rate
        }
        print(f"  {class_name}: {class_in_train}/{total_orig_test} = {rate:.1f}%")
    
    results['per_class_contamination'] = per_class_contamination
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if test_to_train_rate > 50:
        verdict = "CRITICAL: Severe train/test contamination detected!"
        verdict_severity = "critical"
        explanation = (
            f"Over {test_to_train_rate:.0f}% of original test images ended up in our "
            f"training set. This is a major source of data leakage and likely explains "
            f"the artificially high accuracy."
        )
    elif test_to_train_rate > 20:
        verdict = "WARNING: Significant train/test contamination detected"
        verdict_severity = "warning"
        explanation = (
            f"About {test_to_train_rate:.0f}% of original test images ended up in our "
            f"training set. This contamination contributes to inflated accuracy."
        )
    elif test_to_train_rate > 5:
        verdict = "NOTICE: Minor train/test contamination detected"
        verdict_severity = "notice"
        explanation = (
            f"About {test_to_train_rate:.0f}% of original test images ended up in our "
            f"training set. This may slightly inflate accuracy."
        )
    else:
        verdict = "PASS: Minimal train/test contamination"
        verdict_severity = "pass"
        explanation = (
            f"Only {test_to_train_rate:.1f}% contamination detected. "
            f"This is within acceptable limits."
        )
    
    results['verdict'] = {
        'summary': verdict,
        'severity': verdict_severity,
        'explanation': explanation,
        'test_to_train_rate': test_to_train_rate,
        'train_to_test_rate': train_to_test_rate
    }
    
    print(f"\n{verdict}")
    print(f"\n{explanation}")
    
    return results, df


def create_visualizations(df: pd.DataFrame, results: dict, output_dir: Path):
    """Create visualizations for the split analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cross-tabulation heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    cross_tab = pd.crosstab(df['original_split'], df['split'])
    
    im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(cross_tab.columns)))
    ax.set_yticks(range(len(cross_tab.index)))
    ax.set_xticklabels(cross_tab.columns, fontsize=12)
    ax.set_yticklabels(cross_tab.index, fontsize=12)
    
    ax.set_xlabel('Our Split', fontsize=14, fontweight='bold')
    ax.set_ylabel('Original Split (from filename)', fontsize=14, fontweight='bold')
    ax.set_title('Train/Test Split Contamination Matrix', fontsize=16, fontweight='bold')
    
    # Add value annotations
    for i in range(len(cross_tab.index)):
        for j in range(len(cross_tab.columns)):
            val = cross_tab.values[i, j]
            text_color = 'white' if val > cross_tab.values.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=text_color)
    
    plt.colorbar(im, ax=ax, label='Image Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'split_contamination_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class contamination bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(results['per_class_contamination'].keys())
    rates = [results['per_class_contamination'][c]['contamination_rate'] for c in classes]
    
    colors = ['#e74c3c' if r > 50 else '#f39c12' if r > 20 else '#27ae60' for r in rates]
    bars = ax.bar(classes, rates, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Tumor Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Contamination Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class: Original Test Images in Our Train Split', fontsize=14, fontweight='bold')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Critical threshold (50%)')
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Warning threshold (20%)')
    ax.legend()
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_contamination.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Summary pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original split breakdown of our train set
    our_train = df[df['split'] == 'train']
    train_orig_counts = our_train['original_split'].value_counts()
    
    axes[0].pie(train_orig_counts.values, labels=train_orig_counts.index, 
               autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#95a5a6'],
               explode=[0.02] * len(train_orig_counts))
    axes[0].set_title('Our TRAIN Set:\nOriginal Source Breakdown', fontsize=12, fontweight='bold')
    
    # Original split breakdown of our test set
    our_test = df[df['split'] == 'test']
    test_orig_counts = our_test['original_split'].value_counts()
    
    axes[1].pie(test_orig_counts.values, labels=test_orig_counts.index,
               autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#95a5a6'],
               explode=[0.02] * len(test_orig_counts))
    axes[1].set_title('Our TEST Set:\nOriginal Source Breakdown', fontsize=12, fontweight='bold')
    
    plt.suptitle('Split Composition Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'split_composition.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def run_experiment(manifest_path: str = None) -> dict:
    """Run the complete split analysis experiment."""
    os.chdir(Path(__file__).parent.parent.parent)
    
    if manifest_path is None:
        manifest_path = "data/manifests/manifest.parquet"
    
    output_dir = VALIDATION_RESULTS_DIR / "experiment_7_split"
    
    # Run analysis
    results, df = analyze_split_contamination(manifest_path)
    
    # Create visualizations
    create_visualizations(df, results, output_dir)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_experiment()
