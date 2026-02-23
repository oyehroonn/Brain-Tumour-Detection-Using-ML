#!/usr/bin/env python3
"""
Experiment 1: Random Label Shuffle Test

Trains a model with randomly shuffled labels to detect data leakage.
If the model achieves significantly above 25% accuracy (chance level for 4 classes),
we have confirmed data leakage.

This is the most definitive test for data leakage because:
- With random labels, there's no learnable signal from the images
- Any accuracy above chance MUST come from memorization or leakage
- Expected accuracy with random labels: ~25% (random chance)
"""

import os
import sys
import json
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR
from src.models.backbones import get_backbone
from src.models.train import FocalLoss
from src.data.transforms import get_train_transforms, get_val_transforms, BrainTumorDataset


def create_random_label_manifest(
    manifest_path: str,
    output_path: str,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a manifest with randomly shuffled labels.
    
    Args:
        manifest_path: Path to original manifest
        output_path: Path to save shuffled manifest
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shuffled labels
    """
    df = pd.read_parquet(manifest_path)
    
    # Store original labels for reference
    df['original_label'] = df['label'].copy()
    df['original_label_name'] = df['label_name'].copy()
    
    # Shuffle labels
    np.random.seed(random_seed)
    df['label'] = np.random.permutation(df['label'].values)
    
    # Update label names
    label_map = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}
    df['label_name'] = df['label'].map(label_map)
    
    # Save shuffled manifest
    df.to_parquet(output_path, index=False)
    
    # Verify shuffling
    match_rate = (df['label'] == df['original_label']).mean()
    print(f"Label match rate after shuffle: {match_rate:.1%} (expected ~25%)")
    
    return df


def create_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple:
    """Create train/val dataloaders from dataframe."""
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = BrainTumorDataset(train_df, transform=train_transform, data_root='data')
    val_dataset = BrainTumorDataset(val_df, transform=val_transform, data_root='data')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in pbar:
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
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
    
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "macro_f1": macro_f1
    }


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
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
    
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "macro_f1": macro_f1
    }


def run_experiment(
    manifest_path: str = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: str = None
) -> Dict:
    """
    Run the random label shuffle experiment.
    
    Args:
        manifest_path: Path to original manifest
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        
    Returns:
        Dictionary with experiment results
    """
    os.chdir(Path(__file__).parent.parent.parent)
    
    if manifest_path is None:
        manifest_path = "data/manifests/manifest.parquet"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = VALIDATION_RESULTS_DIR / "experiment_1_random_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENT 1: Random Label Shuffle Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"\nExpected accuracy with random labels: ~25%")
    print("If accuracy >> 25%, DATA LEAKAGE is confirmed!")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'device': device,
        'history': {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        },
        'final_metrics': {},
        'verdict': {}
    }
    
    # Create shuffled manifest
    print("\n[1/4] Creating randomly shuffled labels...")
    shuffled_manifest_path = output_dir / "manifest_shuffled.parquet"
    df = create_random_label_manifest(
        manifest_path, 
        str(shuffled_manifest_path)
    )
    
    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(df, batch_size)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n[3/4] Initializing model...")
    model = get_backbone(
        backbone_name="densenet121",
        num_classes=4,
        pretrained=True,
        dropout_rate=0.3
    )
    model = model.to(device)
    
    # Setup training
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training loop
    print("\n[4/4] Training with random labels...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Record history
        results['history']['train_loss'].append(train_metrics['loss'])
        results['history']['train_acc'].append(train_metrics['accuracy'])
        results['history']['val_loss'].append(val_metrics['loss'])
        results['history']['val_acc'].append(val_metrics['accuracy'])
        
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
    
    # Final metrics
    results['final_metrics'] = {
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': results['history']['train_acc'][-1],
        'final_val_accuracy': results['history']['val_acc'][-1],
        'expected_random_accuracy': 0.25
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nExpected accuracy (random chance): 25%")
    print(f"Best validation accuracy achieved: {best_val_acc*100:.2f}%")
    print(f"Final train accuracy: {results['history']['train_acc'][-1]*100:.2f}%")
    print(f"Final val accuracy: {results['history']['val_acc'][-1]*100:.2f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if best_val_acc > 0.50:
        verdict = "CRITICAL: Severe data leakage detected!"
        severity = "critical"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy on RANDOM labels. "
            f"This is {best_val_acc/0.25:.1f}x better than random chance. "
            f"The model is memorizing training data through leakage."
        )
    elif best_val_acc > 0.35:
        verdict = "WARNING: Significant data leakage detected"
        severity = "warning"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy on random labels. "
            f"This is significantly above the 25% random baseline. "
            f"Some form of data leakage is present."
        )
    elif best_val_acc > 0.30:
        verdict = "NOTICE: Possible minor data leakage"
        severity = "notice"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy on random labels. "
            f"This is slightly above the 25% baseline. "
            f"Minor leakage or statistical fluctuation possible."
        )
    else:
        verdict = "PASS: No significant data leakage detected"
        severity = "pass"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy on random labels. "
            f"This is close to the expected 25% random baseline. "
            f"No evidence of data leakage from this test."
        )
    
    results['verdict'] = {
        'summary': verdict,
        'severity': severity,
        'explanation': explanation,
        'accuracy_ratio': best_val_acc / 0.25
    }
    
    print(f"\n{verdict}")
    print(f"\n{explanation}")
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results


def create_visualizations(results: Dict, output_dir: Path):
    """Create visualizations for the random label experiment."""
    history = results['history']
    
    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs, [a*100 for a in history['train_acc']], 'b-o', label='Train', linewidth=2)
    axes[0].plot(epochs, [a*100 for a in history['val_acc']], 'g-s', label='Validation', linewidth=2)
    axes[0].axhline(y=25, color='red', linestyle='--', linewidth=2, label='Random Chance (25%)')
    axes[0].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Critical Threshold (50%)')
    
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy with Random Labels', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)
    
    # Loss plot
    axes[1].plot(epochs, history['train_loss'], 'r-o', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], 'purple', marker='s', label='Validation', linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Loss with Random Labels', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Final accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Random\nChance', 'Train\nAccuracy', 'Val\nAccuracy', 'Critical\nThreshold']
    values = [25, results['final_metrics']['final_train_accuracy']*100,
              results['final_metrics']['best_val_accuracy']*100, 50]
    
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Random Label Test: Final Results', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    # Add verdict text
    verdict = results['verdict']['severity'].upper()
    verdict_color = {'critical': 'red', 'warning': 'orange', 'notice': 'yellow', 'pass': 'green'}.get(results['verdict']['severity'], 'gray')
    ax.text(0.5, 0.95, f"VERDICT: {verdict}", transform=ax.transAxes, fontsize=14,
           fontweight='bold', ha='center', color=verdict_color,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=verdict_color))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run random label test")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", "-d", type=str, default=None,
                       help="Device to train on")
    
    args = parser.parse_args()
    run_experiment(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
