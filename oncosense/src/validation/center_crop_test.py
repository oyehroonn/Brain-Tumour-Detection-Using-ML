#!/usr/bin/env python3
"""
Experiment 2: Center Crop Only Test

Trains a model using only the center 64x64 pixels of each image,
where no tumor information should exist.

If the model achieves significantly above 25% accuracy:
- It's learning spurious features like brightness, contrast patterns
- The model exploits non-tumor artifacts
- Dataset has systematic biases unrelated to tumor content

Expected accuracy: ~25% (random chance)
"""

import os
import sys
import json
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
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR
from src.models.backbones import get_backbone
from src.models.train import FocalLoss
from src.data.transforms import load_image


class CenterCropDataset(torch.utils.data.Dataset):
    """
    Dataset that extracts only the center region of images.
    
    The center crop removes all tumor information, as tumors
    are typically located away from the exact center.
    """
    
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        center_size: int = 64,
        output_size: int = 224,
        data_root: str = "data",
        augment: bool = False
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.center_size = center_size
        self.output_size = output_size
        self.data_root = data_root
        
        # Transform for center crop images
        if augment:
            self.transform = A.Compose([
                A.CenterCrop(height=center_size, width=center_size),
                A.Resize(output_size, output_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(height=center_size, width=center_size),
                A.Resize(output_size, output_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        img_path = f"{self.data_root}/{row['filepath']}"
        image = load_image(img_path, convert_rgb=True)
        label = row["label"]
        
        transformed = self.transform(image=image)
        
        return {
            "image": transformed["image"],
            "label": label,
            "sample_id": row["sample_id"],
            "filepath": row["filepath"]
        }


def create_dataloaders(
    df: pd.DataFrame,
    center_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple:
    """Create train/val dataloaders with center crop only."""
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    train_dataset = CenterCropDataset(
        train_df, center_size=center_size, augment=True, data_root='data'
    )
    val_dataset = CenterCropDataset(
        val_df, center_size=center_size, augment=False, data_root='data'
    )
    
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


def visualize_center_crops(df: pd.DataFrame, output_dir: Path, num_samples: int = 12):
    """Visualize what the center crops look like."""
    sample_df = df.groupby('label').head(num_samples // 4)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if idx >= 12:
            break
            
        img_path = f"data/{row['filepath']}"
        try:
            image = load_image(img_path, convert_rgb=True)
            h, w = image.shape[:2]
            
            # Show original with center box
            ax = axes[idx]
            ax.imshow(image)
            
            # Draw center crop region
            center_size = 64
            x1 = (w - center_size) // 2
            y1 = (h - center_size) // 2
            
            rect = plt.Rectangle((x1, y1), center_size, center_size, 
                                fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect)
            
            ax.set_title(f"{class_names[row['label']]}", fontsize=10, fontweight='bold')
            ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    
    plt.suptitle('Original Images with Center Crop Region (64x64 red box)\n'
                'This small center region is all the model sees!', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'center_crop_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment(
    manifest_path: str = None,
    center_size: int = 64,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: str = None
) -> Dict:
    """
    Run the center crop test experiment.
    
    Args:
        manifest_path: Path to manifest
        center_size: Size of center crop in pixels
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
    
    output_dir = VALIDATION_RESULTS_DIR / "experiment_2_center_crop"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENT 2: Center Crop Only Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Center crop size: {center_size}x{center_size} pixels")
    print(f"Epochs: {num_epochs}")
    print(f"\nThe model will only see the center {center_size}x{center_size} region")
    print("where NO tumor information should exist.")
    print(f"\nExpected accuracy: ~25% (random chance)")
    print("If accuracy >> 25%, model learns spurious features!")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'center_size': center_size,
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
    
    # Load data
    print("\n[1/5] Loading data...")
    df = pd.read_parquet(manifest_path)
    
    # Visualize center crops
    print("\n[2/5] Creating center crop visualization...")
    visualize_center_crops(df, output_dir)
    
    # Create dataloaders
    print("\n[3/5] Creating dataloaders with center crops...")
    train_loader, val_loader = create_dataloaders(df, center_size, batch_size)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n[4/5] Initializing model...")
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
    print("\n[5/5] Training with center crops only...")
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
    print(f"\nCenter crop size: {center_size}x{center_size} pixels")
    print(f"Expected accuracy (random chance): 25%")
    print(f"Best validation accuracy achieved: {best_val_acc*100:.2f}%")
    print(f"Final train accuracy: {results['history']['train_acc'][-1]*100:.2f}%")
    print(f"Final val accuracy: {results['history']['val_acc'][-1]*100:.2f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if best_val_acc > 0.45:
        verdict = "CRITICAL: Strong spurious pattern learning detected!"
        severity = "critical"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy using ONLY the center {center_size}x{center_size} "
            f"pixels where NO tumor should be visible. The model is learning spurious features "
            f"like brightness patterns, contrast, or other non-tumor artifacts."
        )
    elif best_val_acc > 0.35:
        verdict = "WARNING: Moderate spurious pattern learning detected"
        severity = "warning"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy on center crops. "
            f"This suggests the dataset contains non-tumor patterns that correlate with labels."
        )
    elif best_val_acc > 0.30:
        verdict = "NOTICE: Possible minor spurious patterns"
        severity = "notice"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy on center crops. "
            f"This is slightly above random chance, suggesting minor spurious patterns."
        )
    else:
        verdict = "PASS: No significant spurious patterns detected"
        severity = "pass"
        explanation = (
            f"Model achieved {best_val_acc*100:.1f}% accuracy on center crops. "
            f"This is close to the expected 25% random baseline. "
            f"The model doesn't appear to exploit non-tumor features from this test."
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
    """Create visualizations for the center crop experiment."""
    history = results['history']
    
    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs, [a*100 for a in history['train_acc']], 'b-o', label='Train', linewidth=2)
    axes[0].plot(epochs, [a*100 for a in history['val_acc']], 'g-s', label='Validation', linewidth=2)
    axes[0].axhline(y=25, color='red', linestyle='--', linewidth=2, label='Random Chance (25%)')
    axes[0].axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold (40%)')
    
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Accuracy with Center Crop ({results["center_size"]}x{results["center_size"]})', 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)
    
    # Loss plot
    axes[1].plot(epochs, history['train_loss'], 'r-o', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], 'purple', marker='s', label='Validation', linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Loss with Center Crop Only', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Final comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Random\nChance', 'Train\nAccuracy', 'Val\nAccuracy', 'Warning\nThreshold']
    values = [25, results['final_metrics']['final_train_accuracy']*100,
              results['final_metrics']['best_val_accuracy']*100, 40]
    
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Center Crop Test: Final Results', fontsize=16, fontweight='bold')
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
    
    parser = argparse.ArgumentParser(description="Run center crop test")
    parser.add_argument("--center-size", "-s", type=int, default=64,
                       help="Size of center crop in pixels")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", "-d", type=str, default=None,
                       help="Device to train on")
    
    args = parser.parse_args()
    run_experiment(
        center_size=args.center_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
