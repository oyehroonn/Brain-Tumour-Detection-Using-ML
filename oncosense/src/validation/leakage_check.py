#!/usr/bin/env python3
"""
Experiment 6: Near-Duplicate Leakage Check

Quantifies how many perceptually similar images exist across train/val/test splits.
Uses perceptual hashing (pHash) to find near-duplicates that shouldn't be in 
different splits.

If a test image has a near-duplicate in the train set, the model may have 
effectively "seen" the answer during training.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR


def hex_to_hash(hex_str: str) -> imagehash.ImageHash:
    """Convert hex string to ImageHash object."""
    return imagehash.hex_to_hash(hex_str)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hash strings."""
    if hash1 is None or hash2 is None:
        return float('inf')
    try:
        h1 = hex_to_hash(hash1)
        h2 = hex_to_hash(hash2)
        return h1 - h2
    except:
        return float('inf')


def find_near_duplicates_across_splits(
    df: pd.DataFrame, 
    threshold: int = 10,
    sample_size: int = None
) -> Dict:
    """
    Find near-duplicate images that exist across different splits.
    
    Args:
        df: DataFrame with 'phash' and 'split' columns
        threshold: Hamming distance threshold for "near-duplicate"
        sample_size: If set, sample this many test images (for speed)
        
    Returns:
        Dictionary with leakage analysis results
    """
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    if sample_size and len(test_df) > sample_size:
        test_df = test_df.sample(n=sample_size, random_state=42)
    
    print(f"\nAnalyzing {len(test_df)} test images against {len(train_df)} train images...")
    
    results = {
        'test_train_leaks': [],
        'test_val_leaks': [],
        'val_train_leaks': [],
        'stats': {}
    }
    
    # Test vs Train
    print("\n[1/3] Checking TEST vs TRAIN leakage...")
    test_train_leaks = []
    for idx, test_row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test vs Train"):
        test_hash = test_row['phash']
        min_dist = float('inf')
        closest_train = None
        
        for _, train_row in train_df.iterrows():
            dist = hamming_distance(test_hash, train_row['phash'])
            if dist < min_dist:
                min_dist = dist
                closest_train = train_row
        
        if min_dist <= threshold:
            test_train_leaks.append({
                'test_filepath': test_row['filepath'],
                'test_label': test_row['label_name'],
                'train_filepath': closest_train['filepath'],
                'train_label': closest_train['label_name'],
                'hamming_distance': int(min_dist),
                'same_label': test_row['label_name'] == closest_train['label_name']
            })
    
    results['test_train_leaks'] = test_train_leaks
    
    # Test vs Val
    print("\n[2/3] Checking TEST vs VAL leakage...")
    test_val_leaks = []
    for idx, test_row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test vs Val"):
        test_hash = test_row['phash']
        
        for _, val_row in val_df.iterrows():
            dist = hamming_distance(test_hash, val_row['phash'])
            if dist <= threshold:
                test_val_leaks.append({
                    'test_filepath': test_row['filepath'],
                    'val_filepath': val_row['filepath'],
                    'hamming_distance': int(dist),
                })
                break
    
    results['test_val_leaks'] = test_val_leaks
    
    # Val vs Train
    print("\n[3/3] Checking VAL vs TRAIN leakage...")
    val_train_leaks = []
    for idx, val_row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val vs Train"):
        val_hash = val_row['phash']
        
        for _, train_row in train_df.iterrows():
            dist = hamming_distance(val_hash, train_row['phash'])
            if dist <= threshold:
                val_train_leaks.append({
                    'val_filepath': val_row['filepath'],
                    'train_filepath': train_row['filepath'],
                    'hamming_distance': int(dist),
                })
                break
    
    results['val_train_leaks'] = val_train_leaks
    
    # Compute statistics
    total_test = len(test_df)
    total_val = len(val_df)
    
    results['stats'] = {
        'threshold_used': threshold,
        'test_images_analyzed': total_test,
        'val_images_analyzed': total_val,
        'train_images': len(train_df),
        'test_train_leak_count': len(test_train_leaks),
        'test_train_leak_rate': len(test_train_leaks) / total_test * 100 if total_test > 0 else 0,
        'test_val_leak_count': len(test_val_leaks),
        'test_val_leak_rate': len(test_val_leaks) / total_test * 100 if total_test > 0 else 0,
        'val_train_leak_count': len(val_train_leaks),
        'val_train_leak_rate': len(val_train_leaks) / total_val * 100 if total_val > 0 else 0,
    }
    
    # Label consistency check
    same_label_leaks = sum(1 for l in test_train_leaks if l['same_label'])
    diff_label_leaks = len(test_train_leaks) - same_label_leaks
    results['stats']['same_label_leaks'] = same_label_leaks
    results['stats']['different_label_leaks'] = diff_label_leaks
    
    return results


def analyze_leakage_severity(leaks: List[Dict]) -> Dict:
    """Analyze the severity distribution of leaks."""
    if not leaks:
        return {'exact': 0, 'very_similar': 0, 'similar': 0}
    
    severity = {'exact': 0, 'very_similar': 0, 'similar': 0}
    
    for leak in leaks:
        dist = leak['hamming_distance']
        if dist == 0:
            severity['exact'] += 1
        elif dist <= 5:
            severity['very_similar'] += 1
        else:
            severity['similar'] += 1
    
    return severity


def create_visualizations(results: Dict, df: pd.DataFrame, output_dir: Path):
    """Create visualizations for the leakage analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = results['stats']
    
    # 1. Leakage rates bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Test→Train', 'Test→Val', 'Val→Train']
    rates = [
        stats['test_train_leak_rate'],
        stats['test_val_leak_rate'],
        stats['val_train_leak_rate']
    ]
    counts = [
        stats['test_train_leak_count'],
        stats['test_val_leak_count'],
        stats['val_train_leak_count']
    ]
    
    colors = ['#e74c3c' if r > 15 else '#f39c12' if r > 5 else '#27ae60' for r in rates]
    bars = ax.bar(categories, rates, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Leakage Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Near-Duplicate Leakage Across Splits', fontsize=16, fontweight='bold')
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Critical (15%)')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Warning (5%)')
    ax.legend()
    
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{rate:.1f}%\n({count})', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'leakage_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Severity distribution
    test_train_severity = analyze_leakage_severity(results['test_train_leaks'])
    
    if sum(test_train_severity.values()) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        severity_labels = ['Exact (d=0)', 'Very Similar (d≤5)', 'Similar (d≤10)']
        severity_values = [test_train_severity['exact'], 
                         test_train_severity['very_similar'],
                         test_train_severity['similar']]
        
        colors = ['#c0392b', '#e74c3c', '#f39c12']
        ax.bar(severity_labels, severity_values, color=colors, edgecolor='black')
        
        ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax.set_title('Leakage Severity Distribution (Test→Train)', fontsize=14, fontweight='bold')
        
        for i, v in enumerate(severity_values):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'severity_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Hamming distance histogram
    if results['test_train_leaks']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        distances = [l['hamming_distance'] for l in results['test_train_leaks']]
        ax.hist(distances, bins=range(0, 12), color='#3498db', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Hamming Distance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Near-Duplicate Distances', fontsize=14, fontweight='bold')
        ax.axvline(x=5, color='red', linestyle='--', label='Very Similar threshold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distance_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Save example leak pairs (first 10)
    if results['test_train_leaks']:
        save_leak_examples(results['test_train_leaks'][:10], df, output_dir)
    
    print(f"\nVisualizations saved to: {output_dir}")


def save_leak_examples(leaks: List[Dict], df: pd.DataFrame, output_dir: Path):
    """Save visual examples of leaked image pairs."""
    examples_dir = output_dir / 'leak_examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(__file__).parent.parent.parent / 'data'
    
    for i, leak in enumerate(leaks[:10]):
        try:
            test_path = data_root / leak['test_filepath']
            train_path = data_root / leak['train_filepath']
            
            if test_path.exists() and train_path.exists():
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                test_img = Image.open(test_path)
                train_img = Image.open(train_path)
                
                axes[0].imshow(test_img)
                axes[0].set_title(f"TEST: {leak['test_label']}", fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(train_img)
                axes[1].set_title(f"TRAIN: {leak['train_label']}", fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                plt.suptitle(f"Leak Pair #{i+1} - Hamming Distance: {leak['hamming_distance']}", 
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(examples_dir / f'leak_pair_{i+1}.png', dpi=100, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Warning: Could not save leak example {i+1}: {e}")


def run_experiment(
    manifest_path: str = None,
    threshold: int = 10,
    sample_size: int = 500
) -> Dict:
    """
    Run the complete leakage check experiment.
    
    Args:
        manifest_path: Path to manifest parquet file
        threshold: Hamming distance threshold for near-duplicates
        sample_size: Number of test images to sample (for speed)
    """
    os.chdir(Path(__file__).parent.parent.parent)
    
    if manifest_path is None:
        manifest_path = "data/manifests/manifest.parquet"
    
    output_dir = VALIDATION_RESULTS_DIR / "experiment_6_leakage"
    
    print("=" * 60)
    print("EXPERIMENT 6: Near-Duplicate Leakage Check")
    print("=" * 60)
    print(f"Threshold: {threshold} (Hamming distance)")
    print(f"Sample size: {sample_size if sample_size else 'all'} test images")
    
    df = pd.read_parquet(manifest_path)
    
    # Run leakage analysis
    results = find_near_duplicates_across_splits(df, threshold, sample_size)
    results['timestamp'] = datetime.now().isoformat()
    
    # Print summary
    stats = results['stats']
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nTest→Train Leakage: {stats['test_train_leak_count']} images ({stats['test_train_leak_rate']:.1f}%)")
    print(f"  - Same label: {stats['same_label_leaks']}")
    print(f"  - Different label: {stats['different_label_leaks']}")
    
    print(f"\nTest→Val Leakage: {stats['test_val_leak_count']} images ({stats['test_val_leak_rate']:.1f}%)")
    print(f"Val→Train Leakage: {stats['val_train_leak_count']} images ({stats['val_train_leak_rate']:.1f}%)")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    test_train_rate = stats['test_train_leak_rate']
    if test_train_rate > 15:
        verdict = "CRITICAL: Severe near-duplicate leakage detected!"
        severity = "critical"
        explanation = (
            f"{test_train_rate:.1f}% of test images have near-duplicates in the training set. "
            f"This is a major source of inflated accuracy."
        )
    elif test_train_rate > 5:
        verdict = "WARNING: Significant near-duplicate leakage detected"
        severity = "warning"
        explanation = (
            f"{test_train_rate:.1f}% of test images have near-duplicates in training. "
            f"This contributes to accuracy inflation."
        )
    elif test_train_rate > 1:
        verdict = "NOTICE: Minor near-duplicate leakage detected"
        severity = "notice"
        explanation = (
            f"{test_train_rate:.1f}% near-duplicate leakage detected. "
            f"This may slightly affect accuracy."
        )
    else:
        verdict = "PASS: Minimal near-duplicate leakage"
        severity = "pass"
        explanation = (
            f"Only {test_train_rate:.1f}% leakage detected. "
            f"Dataset splitting appears sound."
        )
    
    results['verdict'] = {
        'summary': verdict,
        'severity': severity,
        'explanation': explanation
    }
    
    print(f"\n{verdict}")
    print(f"\n{explanation}")
    
    # Create visualizations
    create_visualizations(results, df, output_dir)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for near-duplicate leakage")
    parser.add_argument("--threshold", "-t", type=int, default=10,
                       help="Hamming distance threshold")
    parser.add_argument("--sample", "-s", type=int, default=500,
                       help="Number of test images to sample")
    
    args = parser.parse_args()
    run_experiment(threshold=args.threshold, sample_size=args.sample)
