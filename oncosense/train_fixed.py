#!/usr/bin/env python3
"""
Train DenseNet121 on Fixed (Non-Leaky) Data
============================================
This script trains a new model using the properly split data from manifest_fixed.parquet.
Expected accuracy: 75-85% (realistic for brain tumor classification)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.models.backbones import get_backbone
from src.models.train import FocalLoss
from src.data.transforms import get_train_transforms, get_val_transforms, BrainTumorDataset


def create_dataloaders(manifest_path: str, batch_size: int = 32, num_workers: int = 0):
    """Create train/val/test dataloaders from fixed manifest."""
    df = pd.read_parquet(manifest_path)
    
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    train_dataset = BrainTumorDataset(train_df, transform=get_train_transforms(), data_root='data')
    val_dataset = BrainTumorDataset(val_df, transform=get_val_transforms(), data_root='data')
    test_dataset = BrainTumorDataset(test_df, transform=get_val_transforms(), data_root='data')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
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
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "predictions": all_preds,
        "labels": all_labels
    }


def main():
    print("=" * 70)
    print("OncoSense Training on FIXED (Non-Leaky) Data")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    manifest_path = "data/manifests/manifest_fixed.parquet"
    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Manifest: {manifest_path}")
    
    # Create dataloaders
    print("\n[1/4] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(manifest_path, batch_size)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = get_backbone("densenet121", num_classes=4, pretrained=True, dropout_rate=0.3)
    model = model.to(device)
    print(f"  Model: DenseNet121 (pretrained)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # Training loop
    print("\n[3/4] Training...")
    print("-" * 70)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['macro_f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])
        
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%, F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, F1: {val_metrics['macro_f1']:.4f}")
        
        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'val_f1': best_val_f1,
                'history': history
            }, checkpoint_dir / "densenet121_fixed.pt")
            print(f"  >> New best model saved! (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Final test evaluation
    print("\n" + "=" * 70)
    print("[4/4] FINAL TEST EVALUATION")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / "densenet121_fixed.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"\nTest Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1: {test_metrics['macro_f1']:.4f}")
    
    # Classification report
    class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    print("\nClassification Report:")
    print(classification_report(test_metrics['labels'], test_metrics['predictions'], 
                               target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    print("Confusion Matrix:")
    print(cm)
    
    # Save training results
    results = {
        'timestamp': datetime.now().isoformat(),
        'manifest': manifest_path,
        'epochs_trained': len(history['train_loss']),
        'best_epoch': checkpoint['epoch'],
        'best_val_accuracy': best_val_acc,
        'best_val_f1': best_val_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['macro_f1'],
        'history': history,
        'config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': device
        }
    }
    
    with open(checkpoint_dir / "training_results_fixed.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {checkpoint_dir / 'densenet121_fixed.pt'}")
    print(f"Results saved to: {checkpoint_dir / 'training_results_fixed.json'}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON")
    print("=" * 70)
    print(f"  Previous (leaky data): 98.08% - INVALID")
    print(f"  Current (fixed data):  {test_metrics['accuracy']*100:.2f}% - VALID")
    print(f"\n  This accuracy is {'within' if 75 <= test_metrics['accuracy']*100 <= 90 else 'outside'} the expected 75-90% range for this task.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
