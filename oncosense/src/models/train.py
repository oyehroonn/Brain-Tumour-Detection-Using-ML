"""
Training module for OncoSense.
Implements model training with focal loss, mixed precision, and early stopping.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import yaml
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from .backbones import get_backbone, BrainTumorClassifier


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p) = -α(1-p)^γ * log(p)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 0.001,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[GradScaler],
    device: str,
    scheduler: Optional[Any] = None,
    gradient_clip: float = 1.0,
    use_amp: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device == "cuda":
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
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
        for batch in tqdm(dataloader, desc="Validating"):
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
    
    # Per-class metrics
    report = classification_report(
        all_labels, all_preds, 
        output_dict=True, 
        zero_division=0
    )
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "classification_report": report
    }


def train_model(
    model_name: str = "densenet121",
    data_config_path: str = "configs/data.yaml",
    train_config_path: str = "configs/train.yaml",
    model_config_path: str = "configs/models.yaml",
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model from scratch.
    
    Args:
        model_name: Name of the backbone to train.
        data_config_path: Path to data config.
        train_config_path: Path to training config.
        model_config_path: Path to model config.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to train on.
        
    Returns:
        Tuple of (trained_model, training_history).
    """
    # Load configs
    train_config = load_config(train_config_path)
    model_config = load_config(model_config_path)
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Get device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create model
    model = get_backbone(
        model_name,
        num_classes=4,
        pretrained=True,
        dropout_rate=model_config["backbones"][model_name]["dropout_rate"]
    )
    model = model.to(device)
    
    # Create dataloaders
    from ..data.transforms import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config_path=data_config_path,
        train_config_path=train_config_path
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Loss function
    loss_config = train_config["loss"]
    if loss_config["name"] == "focal":
        class_weights = torch.tensor(loss_config["class_weights"]).to(device)
        criterion = FocalLoss(
            gamma=loss_config["gamma"],
            alpha=class_weights
        )
    else:
        class_weights = torch.tensor(loss_config["class_weights"]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    opt_config = train_config["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=opt_config["lr"],
        weight_decay=opt_config["weight_decay"],
        betas=tuple(opt_config["betas"])
    )
    
    # Scheduler
    sched_config = train_config["scheduler"]
    if sched_config["name"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=sched_config["T_max"],
            eta_min=sched_config["eta_min"]
        )
        step_scheduler_per_batch = False
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=opt_config["lr"],
            epochs=train_config["training"]["epochs"],
            steps_per_epoch=len(train_loader)
        )
        step_scheduler_per_batch = True
    
    # Mixed precision (only enabled for CUDA)
    use_mixed_precision = train_config["training"]["mixed_precision"] and device == "cuda"
    scaler = GradScaler("cuda", enabled=use_mixed_precision) if device == "cuda" else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_config["early_stopping"]["patience"],
        min_delta=train_config["early_stopping"]["min_delta"],
        mode=train_config["early_stopping"]["mode"]
    )
    
    # Training loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    
    use_amp = train_config["training"]["mixed_precision"] and device == "cuda"
    
    for epoch in range(train_config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{train_config['training']['epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            scheduler if step_scheduler_per_batch else None,
            train_config["training"]["gradient_clip_val"],
            use_amp=use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler (epoch-level)
        if not step_scheduler_per_batch:
            scheduler.step()
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["macro_f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["macro_f1"])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']*100:.2f}% | "
              f"F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"F1: {val_metrics['macro_f1']:.4f}")
        
        # Save best model
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics["macro_f1"],
                "val_acc": val_metrics["accuracy"],
                "config": {
                    "model_name": model_name,
                    "train_config": train_config
                }
            }
            
            save_path = f"{checkpoint_dir}/{model_name}_best.pt"
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")
        
        # Early stopping
        if early_stopping(val_metrics["macro_f1"]):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"Best Val F1: {best_f1:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(f"{checkpoint_dir}/{model_name}_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Acc: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1: {test_metrics['macro_f1']:.4f}")
    
    history["test_metrics"] = test_metrics
    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_f1
    
    # Save training history
    history_path = f"{checkpoint_dir}/{model_name}_history.json"
    with open(history_path, "w") as f:
        # Convert numpy types for JSON serialization
        history_serializable = {
            k: [float(v) if isinstance(v, (np.floating, float)) else v for v in vals]
            if isinstance(vals, list) else vals
            for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2, default=str)
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument("--model", "-m", type=str, default="densenet121",
                        choices=["densenet121", "xception", "efficientnet_b0"],
                        help="Model backbone to train")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="Device to train on")
    parser.add_argument("--checkpoint-dir", "-c", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
