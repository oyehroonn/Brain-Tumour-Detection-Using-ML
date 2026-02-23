#!/usr/bin/env python3
"""
Experiment 3: External Dataset Validation

Tests the trained model on a completely different dataset (Figshare/Jun Cheng)
to measure true generalization capability.

The Jun Cheng dataset:
- 3,064 T1-weighted contrast-enhanced MRI images
- 3 tumor classes: meningioma, glioma, pituitary
- Different source, different scanners, different preprocessing

If 98% accuracy is real: Should achieve 80-90% on external data
If 98% is an artifact: Will likely drop to 40-60%
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR, DATA_DIR
from src.models.backbones import get_backbone


# Figshare dataset info
FIGSHARE_URL = "https://ndownloader.figshare.com/files/3381290"
DATASET_NAME = "figshare_jun_cheng"

# Class mapping between datasets
# Original Kaggle: glioma(0), meningioma(1), pituitary(2), no_tumor(3)
# Figshare: meningioma(1), glioma(2), pituitary(3) - note different ordering!
FIGSHARE_TO_KAGGLE_MAPPING = {
    1: 1,  # meningioma -> meningioma
    2: 0,  # glioma -> glioma
    3: 2,  # pituitary -> pituitary
}

CLASS_NAMES = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}


class ExternalDataset(torch.utils.data.Dataset):
    """Dataset for external validation images."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': label,
            'filepath': img_path
        }


def get_inference_transforms():
    """Get transforms for inference."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def download_figshare_dataset(output_dir: Path) -> bool:
    """
    Download the Figshare brain tumor dataset.
    
    Note: This is a large file (~800MB). For manual download, go to:
    https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "brain_tumor_dataset.zip"
    
    if (output_dir / "brainTumorDataPublic_1-766").exists():
        print("Dataset already downloaded and extracted.")
        return True
    
    print(f"\nDownloading Figshare dataset...")
    print(f"URL: {FIGSHARE_URL}")
    print(f"This may take several minutes (~800MB)...")
    print("\nAlternatively, download manually from:")
    print("https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")
    print(f"and extract to: {output_dir}")
    
    try:
        # Create progress bar for download
        def download_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(FIGSHARE_URL, zip_path, download_progress)
        print("\nDownload complete!")
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up zip
        zip_path.unlink()
        print("Extraction complete!")
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nPlease download manually from:")
        print("https://figshare.com/articles/dataset/brain_tumor_dataset/1512427")
        return False


def load_figshare_dataset(dataset_dir: Path) -> tuple:
    """
    Load the Figshare dataset and prepare for evaluation.
    
    Returns:
        Tuple of (image_paths, labels, label_names)
    """
    # Find the data directories
    data_dirs = list(dataset_dir.glob("brainTumorDataPublic_*"))
    
    if not data_dirs:
        # Try alternative structure
        data_dirs = [dataset_dir]
    
    image_paths = []
    labels = []
    
    print(f"\nLoading Figshare dataset from {dataset_dir}...")
    
    for data_dir in data_dirs:
        # Load the cvind.mat or folder structure
        # The Figshare dataset uses .mat files with labels
        
        # Look for image folders
        for subdir in data_dir.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                # Folder name is the class label
                class_label = int(subdir.name)
                
                if class_label in FIGSHARE_TO_KAGGLE_MAPPING:
                    kaggle_label = FIGSHARE_TO_KAGGLE_MAPPING[class_label]
                    
                    for img_file in subdir.glob("*.jpg"):
                        image_paths.append(str(img_file))
                        labels.append(kaggle_label)
                    
                    for img_file in subdir.glob("*.png"):
                        image_paths.append(str(img_file))
                        labels.append(kaggle_label)
    
    # Also check for flat structure with numbered files
    if not image_paths:
        print("Checking for flat file structure...")
        for mat_file in dataset_dir.rglob("*.mat"):
            # This dataset uses .mat files - would need scipy to load
            print(f"Found .mat file: {mat_file}")
        
        # Look for any image files
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for img_file in dataset_dir.rglob(ext):
                # Try to infer label from path
                parent = img_file.parent.name
                if parent.isdigit():
                    class_label = int(parent)
                    if class_label in FIGSHARE_TO_KAGGLE_MAPPING:
                        kaggle_label = FIGSHARE_TO_KAGGLE_MAPPING[class_label]
                        image_paths.append(str(img_file))
                        labels.append(kaggle_label)
    
    print(f"Loaded {len(image_paths)} images")
    
    if image_paths:
        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\nClass distribution:")
        for u, c in zip(unique, counts):
            print(f"  {CLASS_NAMES[u]}: {c}")
    
    return image_paths, labels


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model = get_backbone(
        backbone_name="densenet121",
        num_classes=4,
        pretrained=False,
        dropout_rate=0.3
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_on_external(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> Dict:
    """Evaluate model on external dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label']
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics (only for classes present in external dataset)
    present_classes = np.unique(all_labels)
    per_class_acc = {}
    for cls in present_classes:
        mask = all_labels == cls
        per_class_acc[CLASS_NAMES[cls]] = (all_preds[mask] == all_labels[mask]).mean()
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }


def run_experiment(
    checkpoint_path: str = None,
    device: str = None,
    batch_size: int = 32,
    original_accuracy: float = 0.98
) -> Dict:
    """
    Run the external validation experiment.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device to use
        batch_size: Batch size for evaluation
        original_accuracy: Original model accuracy for comparison
    """
    os.chdir(Path(__file__).parent.parent.parent)
    
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/densenet121_best.pt"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = VALIDATION_RESULTS_DIR / "experiment_3_external"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    external_data_dir = DATA_DIR / "external" / DATASET_NAME
    
    print("=" * 60)
    print("EXPERIMENT 3: External Dataset Validation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"\nThis experiment tests generalization on an external dataset")
    print("(Figshare/Jun Cheng) with different source and preprocessing.")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'original_accuracy': original_accuracy,
        'external_metrics': {},
        'verdict': {}
    }
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        results['error'] = "Checkpoint not found"
        return results
    
    # Download dataset if needed
    print("\n[1/4] Checking external dataset...")
    if not external_data_dir.exists() or not list(external_data_dir.glob("*")):
        print("External dataset not found. Attempting download...")
        
        # Create instructions for manual download
        instructions = f"""
MANUAL DOWNLOAD REQUIRED:

The Figshare brain tumor dataset is large (~800MB) and may require
manual download due to network restrictions.

1. Go to: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
2. Download the dataset (brain_tumor_dataset.zip or similar)
3. Extract to: {external_data_dir}

The expected structure after extraction:
{external_data_dir}/
├── 1/  (meningioma images)
├── 2/  (glioma images)
└── 3/  (pituitary images)

After placing the data, re-run this experiment.
"""
        print(instructions)
        
        # Try automatic download
        success = download_figshare_dataset(external_data_dir)
        
        if not success:
            # Save instructions
            with open(output_dir / "DOWNLOAD_INSTRUCTIONS.txt", 'w') as f:
                f.write(instructions)
            
            results['error'] = "External dataset not available"
            results['instructions'] = instructions
            
            # Save partial results
            with open(output_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nInstructions saved to: {output_dir / 'DOWNLOAD_INSTRUCTIONS.txt'}")
            return results
    
    # Load external dataset
    print("\n[2/4] Loading external dataset...")
    image_paths, labels = load_figshare_dataset(external_data_dir)
    
    if not image_paths:
        print("\nNo images found in external dataset.")
        print("Please check the dataset structure and re-run.")
        results['error'] = "No images found in external dataset"
        return results
    
    # Create dataloader
    print("\n[3/4] Preparing evaluation...")
    transform = get_inference_transforms()
    dataset = ExternalDataset(image_paths, labels, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Load model
    print("\n[4/4] Evaluating model...")
    model = load_model(checkpoint_path, device)
    
    # Evaluate
    metrics = evaluate_on_external(model, dataloader, device)
    results['external_metrics'] = metrics
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nOriginal model accuracy (Kaggle): {original_accuracy*100:.2f}%")
    print(f"External dataset accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"External dataset F1: {metrics['macro_f1']:.4f}")
    print(f"Total external samples: {metrics['total_samples']}")
    
    print("\nPer-class accuracy on external data:")
    for cls, acc in metrics['per_class_accuracy'].items():
        print(f"  {cls}: {acc*100:.2f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    external_acc = metrics['accuracy']
    generalization_gap = original_accuracy - external_acc
    
    if external_acc >= 0.80:
        verdict = "EXCELLENT: Strong generalization to external data!"
        severity = "pass"
        explanation = (
            f"Model achieved {external_acc*100:.1f}% on external data. "
            f"This suggests the model learned genuine tumor features."
        )
    elif external_acc >= 0.65:
        verdict = "GOOD: Reasonable generalization to external data"
        severity = "pass"
        explanation = (
            f"Model achieved {external_acc*100:.1f}% on external data. "
            f"Some domain shift expected between datasets."
        )
    elif external_acc >= 0.50:
        verdict = "WARNING: Moderate generalization gap"
        severity = "warning"
        explanation = (
            f"Model achieved {external_acc*100:.1f}% on external data "
            f"(gap of {generalization_gap*100:.1f}% from original). "
            f"The model may be overfitting to dataset-specific features."
        )
    elif external_acc >= 0.35:
        verdict = "CRITICAL: Large generalization gap detected!"
        severity = "critical"
        explanation = (
            f"Model only achieved {external_acc*100:.1f}% on external data "
            f"(gap of {generalization_gap*100:.1f}%). "
            f"The high original accuracy is likely due to dataset artifacts."
        )
    else:
        verdict = "CRITICAL: Model fails to generalize!"
        severity = "critical"
        explanation = (
            f"Model only achieved {external_acc*100:.1f}% on external data. "
            f"This is close to random chance for 3 classes (~33%). "
            f"The original 98% accuracy is almost certainly an artifact."
        )
    
    results['verdict'] = {
        'summary': verdict,
        'severity': severity,
        'explanation': explanation,
        'generalization_gap': generalization_gap
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
    """Create visualizations for external validation."""
    if 'error' in results:
        return
    
    metrics = results['external_metrics']
    
    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Original\n(Kaggle)', 'External\n(Figshare)', 'Gap']
    orig = results['original_accuracy'] * 100
    ext = metrics['accuracy'] * 100
    gap = orig - ext
    
    colors = ['#3498db', '#27ae60' if ext >= 65 else '#f39c12' if ext >= 50 else '#e74c3c', 
              '#e74c3c' if gap > 30 else '#f39c12' if gap > 15 else '#27ae60']
    
    bars = ax.bar(categories, [orig, ext, gap], color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy / Gap (%)', fontsize=14, fontweight='bold')
    ax.set_title('Generalization to External Dataset', fontsize=16, fontweight='bold')
    
    for bar, val in zip(bars, [orig, ext, gap]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    verdict = results['verdict']['severity'].upper()
    verdict_color = {'critical': 'red', 'warning': 'orange', 'pass': 'green'}.get(
        results['verdict']['severity'], 'gray')
    ax.text(0.5, 0.95, f"VERDICT: {verdict}", transform=ax.transAxes, fontsize=14,
           fontweight='bold', ha='center', color=verdict_color,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=verdict_color))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrix
    if 'confusion_matrix' in metrics:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cm = np.array(metrics['confusion_matrix'])
        # Only show classes present in external data (0, 1, 2 - not no_tumor)
        cm_subset = cm[:3, :3]
        class_subset = ['glioma', 'meningioma', 'pituitary']
        
        im = ax.imshow(cm_subset, cmap='Blues')
        
        ax.set_xticks(range(len(class_subset)))
        ax.set_yticks(range(len(class_subset)))
        ax.set_xticklabels(class_subset, rotation=45, ha='right')
        ax.set_yticklabels(class_subset)
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix on External Dataset', fontsize=14, fontweight='bold')
        
        # Add value annotations
        for i in range(len(class_subset)):
            for j in range(len(class_subset)):
                val = cm_subset[i, j]
                text_color = 'white' if val > cm_subset.max() / 2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                       fontsize=12, fontweight='bold', color=text_color)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Per-class accuracy
    if metrics['per_class_accuracy']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(metrics['per_class_accuracy'].keys())
        accs = [metrics['per_class_accuracy'][c] * 100 for c in classes]
        
        colors = ['#27ae60' if a >= 70 else '#f39c12' if a >= 50 else '#e74c3c' for a in accs]
        bars = ax.bar(classes, accs, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Per-Class Accuracy on External Dataset', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.axhline(y=33.3, color='red', linestyle='--', alpha=0.5, label='Random Chance (3 classes)')
        ax.legend()
        
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run external validation")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--original-accuracy", "-a", type=float, default=0.98,
                       help="Original model accuracy")
    
    args = parser.parse_args()
    run_experiment(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        original_accuracy=args.original_accuracy
    )
