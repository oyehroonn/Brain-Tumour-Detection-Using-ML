#!/usr/bin/env python3
"""
Experiment 4: Proper pHash Near-Duplicate Clustering

Re-runs the full deduplication pipeline that was skipped in quick_setup.py.
Uses AgglomerativeClustering on perceptual hashes to group near-duplicate images,
then ensures same-cluster images stay in the same split.

If accuracy drops significantly compared to original training:
- Near-duplicate leakage was inflating accuracy
- The quick_setup.py bypass created data leakage
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imagehash
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR
from src.models.backbones import get_backbone
from src.models.train import FocalLoss
from src.data.transforms import get_train_transforms, get_val_transforms, BrainTumorDataset


def hex_to_binary(hex_str: str) -> np.ndarray:
    """Convert hex hash string to binary array."""
    return np.array([int(b) for b in bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)])


def compute_pairwise_distances(phashes: List[str]) -> np.ndarray:
    """Compute pairwise Hamming distances between perceptual hashes."""
    n = len(phashes)
    distances = np.zeros((n, n), dtype=np.float32)
    
    print(f"Computing pairwise distances for {n} images...")
    
    for i in tqdm(range(n), desc="Computing distances"):
        h1 = imagehash.hex_to_hash(phashes[i])
        for j in range(i + 1, n):
            h2 = imagehash.hex_to_hash(phashes[j])
            dist = h1 - h2  # Hamming distance
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def cluster_near_duplicates(
    df: pd.DataFrame,
    distance_threshold: int = 10
) -> pd.DataFrame:
    """
    Cluster near-duplicate images using AgglomerativeClustering.
    
    Args:
        df: DataFrame with 'phash' column
        distance_threshold: Max Hamming distance to consider as same cluster
        
    Returns:
        DataFrame with 'cluster_id' column added
    """
    df = df.copy()
    
    # Get unique phashes and their indices
    unique_phashes = df['phash'].unique()
    phash_to_idx = {ph: i for i, ph in enumerate(unique_phashes)}
    
    print(f"\nUnique phashes: {len(unique_phashes)}")
    print(f"Total images: {len(df)}")
    
    if len(unique_phashes) > 5000:
        print("\nWARNING: Large dataset - using sampling for clustering")
        # For very large datasets, sample and cluster
        sample_size = min(3000, len(unique_phashes))
        sample_indices = np.random.choice(len(unique_phashes), sample_size, replace=False)
        sample_phashes = [unique_phashes[i] for i in sample_indices]
        
        distances = compute_pairwise_distances(sample_phashes)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distances)
        
        # Map sampled phashes to clusters
        sample_cluster_map = {sample_phashes[i]: cluster_labels[i] for i in range(len(sample_phashes))}
        
        # Assign remaining phashes to nearest cluster or new cluster
        next_cluster = max(cluster_labels) + 1
        phash_cluster_map = {}
        
        for ph in unique_phashes:
            if ph in sample_cluster_map:
                phash_cluster_map[ph] = sample_cluster_map[ph]
            else:
                # Find nearest sampled phash
                h = imagehash.hex_to_hash(ph)
                min_dist = float('inf')
                nearest_cluster = next_cluster
                
                for sampled_ph, cluster in sample_cluster_map.items():
                    dist = h - imagehash.hex_to_hash(sampled_ph)
                    if dist < min_dist:
                        min_dist = dist
                        if dist <= distance_threshold:
                            nearest_cluster = cluster
                
                if nearest_cluster == next_cluster:
                    next_cluster += 1
                phash_cluster_map[ph] = nearest_cluster
    else:
        # Full clustering for smaller datasets
        distances = compute_pairwise_distances(list(unique_phashes))
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distances)
        phash_cluster_map = {unique_phashes[i]: cluster_labels[i] for i in range(len(unique_phashes))}
    
    # Assign cluster IDs to all rows
    df['cluster_id'] = df['phash'].map(phash_cluster_map)
    
    # Statistics
    num_clusters = df['cluster_id'].nunique()
    cluster_sizes = df['cluster_id'].value_counts()
    multi_image_clusters = (cluster_sizes > 1).sum()
    
    print(f"\nClustering Results:")
    print(f"  Total clusters: {num_clusters}")
    print(f"  Multi-image clusters: {multi_image_clusters}")
    print(f"  Largest cluster: {cluster_sizes.max()} images")
    print(f"  Average cluster size: {cluster_sizes.mean():.2f}")
    
    return df


def stratified_cluster_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Split data by clusters while maintaining class stratification.
    Ensures images from the same cluster are never split across sets.
    """
    df = df.copy()
    np.random.seed(random_seed)
    
    # Get cluster info
    cluster_info = df.groupby('cluster_id').agg({
        'label': lambda x: x.mode()[0],
        'sample_id': 'count'
    }).reset_index()
    cluster_info.columns = ['cluster_id', 'primary_label', 'size']
    
    # Group clusters by class
    clusters_by_class = defaultdict(list)
    for _, row in cluster_info.iterrows():
        clusters_by_class[row['primary_label']].append(row['cluster_id'])
    
    train_clusters = []
    val_clusters = []
    test_clusters = []
    
    # Stratified split for each class
    for label, clusters in clusters_by_class.items():
        clusters = np.array(clusters)
        np.random.shuffle(clusters)
        
        n_clusters = len(clusters)
        n_train = int(n_clusters * train_ratio)
        n_val = int(n_clusters * val_ratio)
        
        train_clusters.extend(clusters[:n_train])
        val_clusters.extend(clusters[n_train:n_train + n_val])
        test_clusters.extend(clusters[n_train + n_val:])
    
    # Assign splits
    train_set = set(train_clusters)
    val_set = set(val_clusters)
    test_set = set(test_clusters)
    
    def assign_split(cluster_id):
        if cluster_id in train_set:
            return 'train'
        elif cluster_id in val_set:
            return 'val'
        elif cluster_id in test_set:
            return 'test'
        return 'train'
    
    df['split'] = df['cluster_id'].apply(assign_split)
    
    return df


def create_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple:
    """Create train/val/test dataloaders."""
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    train_dataset = BrainTumorDataset(train_df, transform=get_train_transforms(), data_root='data')
    val_dataset = BrainTumorDataset(val_df, transform=get_val_transforms(), data_root='data')
    test_dataset = BrainTumorDataset(test_df, transform=get_val_transforms(), data_root='data')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "macro_f1": f1_score(all_labels, all_preds, average="macro")
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "macro_f1": f1_score(all_labels, all_preds, average="macro")
    }


def run_experiment(
    manifest_path: str = None,
    distance_threshold: int = 10,
    num_epochs: int = 15,
    batch_size: int = 32,
    device: str = None,
    original_best_accuracy: float = 0.98
) -> Dict:
    """
    Run the proper clustering experiment.
    
    Args:
        manifest_path: Path to raw manifest (before quick_setup)
        distance_threshold: pHash distance threshold for clustering
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        original_best_accuracy: Original model's best accuracy for comparison
    """
    os.chdir(Path(__file__).parent.parent.parent)
    
    if manifest_path is None:
        manifest_path = "data/manifests/manifest_raw.parquet"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = VALIDATION_RESULTS_DIR / "experiment_4_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENT 4: Proper pHash Near-Duplicate Clustering")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Distance threshold: {distance_threshold}")
    print(f"Epochs: {num_epochs}")
    print(f"\nThis experiment re-runs proper near-duplicate clustering")
    print("that was skipped in quick_setup.py")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'distance_threshold': distance_threshold,
        'num_epochs': num_epochs,
        'original_accuracy': original_best_accuracy,
        'clustering_stats': {},
        'history': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
        'final_metrics': {},
        'verdict': {}
    }
    
    # Load raw manifest
    print("\n[1/6] Loading raw manifest...")
    if not Path(manifest_path).exists():
        # Fall back to processed manifest
        manifest_path = "data/manifests/manifest.parquet"
        print(f"  Raw manifest not found, using: {manifest_path}")
    
    df = pd.read_parquet(manifest_path)
    print(f"  Loaded {len(df)} images")
    
    # Remove exact duplicates
    print("\n[2/6] Removing exact duplicates...")
    before = len(df)
    df = df.drop_duplicates(subset=['sample_id'], keep='first')
    after = len(df)
    print(f"  Removed {before - after} exact duplicates, {after} remaining")
    
    # Perform proper clustering
    print("\n[3/6] Clustering near-duplicates (this may take a while)...")
    df = cluster_near_duplicates(df, distance_threshold)
    
    results['clustering_stats'] = {
        'total_images': len(df),
        'num_clusters': df['cluster_id'].nunique(),
        'largest_cluster': int(df['cluster_id'].value_counts().max()),
        'avg_cluster_size': float(df['cluster_id'].value_counts().mean())
    }
    
    # Create proper splits
    print("\n[4/6] Creating cluster-aware splits...")
    df = stratified_cluster_split(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    
    # Save clustered manifest
    clustered_manifest_path = output_dir / "manifest_properly_clustered.parquet"
    df.to_parquet(clustered_manifest_path, index=False)
    
    # Print split statistics
    print("\nSplit Statistics:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"  {split}: {len(split_df)} images, {split_df['cluster_id'].nunique()} clusters")
    
    # Verify no cluster leakage
    train_clusters = set(df[df['split'] == 'train']['cluster_id'])
    val_clusters = set(df[df['split'] == 'val']['cluster_id'])
    test_clusters = set(df[df['split'] == 'test']['cluster_id'])
    
    overlap = (train_clusters & val_clusters) | (train_clusters & test_clusters) | (val_clusters & test_clusters)
    print(f"\nCluster overlap between splits: {len(overlap)} (should be 0)")
    
    # Create dataloaders
    print("\n[5/6] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(df, batch_size)
    
    # Initialize model
    print("\n[6/6] Training model with proper clustering...")
    model = get_backbone("densenet121", num_classes=4, pretrained=True, dropout_rate=0.3)
    model = model.to(device)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        results['history']['train_loss'].append(train_metrics['loss'])
        results['history']['train_acc'].append(train_metrics['accuracy'])
        results['history']['val_loss'].append(val_metrics['loss'])
        results['history']['val_acc'].append(val_metrics['accuracy'])
        
        print(f"  Train - Acc: {train_metrics['accuracy']*100:.2f}%, F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val   - Acc: {val_metrics['accuracy']*100:.2f}%, F1: {val_metrics['macro_f1']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_accuracy': best_val_acc
            }, output_dir / "model_proper_clustering.pt")
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    results['final_metrics'] = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['macro_f1'],
        'original_accuracy': original_best_accuracy,
        'accuracy_drop': original_best_accuracy - best_val_acc
    }
    
    print(f"\nOriginal model accuracy: {original_best_accuracy*100:.2f}%")
    print(f"Properly clustered accuracy: {best_val_acc*100:.2f}%")
    print(f"Test accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Accuracy drop: {(original_best_accuracy - best_val_acc)*100:.2f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    accuracy_drop = original_best_accuracy - best_val_acc
    
    if accuracy_drop > 0.15:
        verdict = "CRITICAL: Major accuracy drop - near-duplicate leakage confirmed!"
        severity = "critical"
        explanation = (
            f"Accuracy dropped by {accuracy_drop*100:.1f}% when using proper clustering. "
            f"This confirms that near-duplicate images across splits were inflating accuracy."
        )
    elif accuracy_drop > 0.08:
        verdict = "WARNING: Significant accuracy drop - likely near-duplicate leakage"
        severity = "warning"
        explanation = (
            f"Accuracy dropped by {accuracy_drop*100:.1f}% with proper clustering. "
            f"This suggests near-duplicate leakage was contributing to inflated accuracy."
        )
    elif accuracy_drop > 0.03:
        verdict = "NOTICE: Minor accuracy drop - possible minor leakage"
        severity = "notice"
        explanation = (
            f"Accuracy dropped by {accuracy_drop*100:.1f}% with proper clustering. "
            f"Some near-duplicate leakage may have been present."
        )
    else:
        verdict = "PASS: Minimal accuracy change - clustering doesn't impact results"
        severity = "pass"
        explanation = (
            f"Accuracy only changed by {accuracy_drop*100:.1f}% with proper clustering. "
            f"The quick_setup.py bypass did not significantly impact accuracy."
        )
    
    results['verdict'] = {
        'summary': verdict,
        'severity': severity,
        'explanation': explanation,
        'accuracy_drop': accuracy_drop
    }
    
    print(f"\n{verdict}")
    print(f"\n{explanation}")
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def create_visualizations(results: Dict, output_dir: Path):
    """Create visualizations for the clustering experiment."""
    
    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Original\n(Quick Setup)', 'Proper\nClustering', 'Difference']
    orig_acc = results['original_accuracy'] * 100
    new_acc = results['final_metrics']['best_val_accuracy'] * 100
    diff = orig_acc - new_acc
    
    values = [orig_acc, new_acc, diff]
    colors = ['#3498db', '#27ae60', '#e74c3c' if diff > 5 else '#f39c12' if diff > 2 else '#27ae60']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy / Drop (%)', fontsize=14, fontweight='bold')
    ax.set_title('Impact of Proper Near-Duplicate Clustering', fontsize=16, fontweight='bold')
    
    for bar, val in zip(bars, values):
        sign = '+' if val < 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{sign}{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    verdict = results['verdict']['severity'].upper()
    verdict_color = {'critical': 'red', 'warning': 'orange', 'notice': 'yellow', 'pass': 'green'}.get(results['verdict']['severity'], 'gray')
    ax.text(0.5, 0.95, f"VERDICT: {verdict}", transform=ax.transAxes, fontsize=14,
           fontweight='bold', ha='center', color=verdict_color,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=verdict_color))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Training curves
    if results['history']['train_acc']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(results['history']['train_acc']) + 1)
        
        axes[0].plot(epochs, [a*100 for a in results['history']['train_acc']], 'b-o', label='Train')
        axes[0].plot(epochs, [a*100 for a in results['history']['val_acc']], 'g-s', label='Val')
        axes[0].axhline(y=results['original_accuracy']*100, color='red', linestyle='--', 
                       label=f'Original ({results["original_accuracy"]*100:.0f}%)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Training Progress with Proper Clustering')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, results['history']['train_loss'], 'r-o', label='Train')
        axes[1].plot(epochs, results['history']['val_loss'], 'purple', marker='s', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss with Proper Clustering')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run proper clustering experiment")
    parser.add_argument("--threshold", "-t", type=int, default=10,
                       help="pHash distance threshold")
    parser.add_argument("--epochs", "-e", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--original-accuracy", "-a", type=float, default=0.98,
                       help="Original model's best accuracy")
    
    args = parser.parse_args()
    run_experiment(
        distance_threshold=args.threshold,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        original_best_accuracy=args.original_accuracy
    )
