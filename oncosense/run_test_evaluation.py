#!/usr/bin/env python3
"""
Comprehensive Test Set Evaluation
Evaluates the trained model on the held-out test set with full metrics and visualizations.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# Setup paths
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from src.models.backbones import get_backbone
from src.data.transforms import get_val_transforms, BrainTumorDataset

# Configuration
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
CLASS_NAMES_LOWER = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
OUTPUT_DIR = Path("test_evaluation_results")

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#1A1A1A',
    'secondary': '#6B6B6B', 
    'accent': '#2D2D2D',
    'glioma': '#E74C3C',
    'meningioma': '#3498DB',
    'pituitary': '#27AE60',
    'no_tumor': '#9B59B6'
}


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = get_backbone('densenet121', num_classes=4, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def load_test_data(manifest_path, batch_size=32):
    """Load test dataset."""
    print(f"Loading test data from {manifest_path}...")
    
    df = pd.read_parquet(manifest_path)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"Test samples: {len(test_df)}")
    print(f"Class distribution:")
    for label in sorted(test_df['label'].unique()):
        count = len(test_df[test_df['label'] == label])
        name = CLASS_NAMES[label]
        print(f"  {name}: {count} ({count/len(test_df)*100:.1f}%)")
    
    transform = get_val_transforms()
    dataset = BrainTumorDataset(test_df, transform, data_root="data")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return dataloader, test_df


def run_evaluation(model, dataloader, device):
    """Run inference on test set and collect predictions."""
    print("\nRunning evaluation...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_filepaths = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            filepaths = batch['filepath']
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
            all_filepaths.extend(filepaths)
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'filepaths': all_filepaths
    }


def compute_metrics(results):
    """Compute comprehensive evaluation metrics."""
    print("\nComputing metrics...")
    
    y_true = results['labels']
    y_pred = results['predictions']
    y_probs = results['probabilities']
    
    # Overall metrics
    metrics = {
        'overall': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
            'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
            'macro_precision': float(precision_score(y_true, y_pred, average='macro')),
            'macro_recall': float(recall_score(y_true, y_pred, average='macro')),
            'total_samples': len(y_true),
            'correct_predictions': int((y_true == y_pred).sum()),
            'incorrect_predictions': int((y_true != y_pred).sum())
        }
    }
    
    # Per-class metrics
    metrics['per_class'] = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        class_preds = y_pred[mask]
        class_true = y_true[mask]
        class_probs = y_probs[mask, i]
        
        correct = (class_preds == class_true).sum()
        total = len(class_true)
        
        metrics['per_class'][name] = {
            'accuracy': float(correct / total) if total > 0 else 0.0,
            'precision': float(precision_score(y_true == i, y_pred == i, zero_division=0)),
            'recall': float(recall_score(y_true == i, y_pred == i, zero_division=0)),
            'f1_score': float(f1_score(y_true == i, y_pred == i, zero_division=0)),
            'support': int(total),
            'correct': int(correct),
            'avg_confidence': float(class_probs.mean()) if total > 0 else 0.0
        }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # ROC-AUC (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    metrics['roc_auc'] = {}
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc_score = auc(fpr, tpr)
        metrics['roc_auc'][name] = float(roc_auc_score)
    metrics['roc_auc']['macro_avg'] = float(np.mean(list(metrics['roc_auc'].values())))
    
    # Confidence analysis
    correct_mask = y_true == y_pred
    correct_probs = y_probs[np.arange(len(y_pred)), y_pred][correct_mask]
    incorrect_probs = y_probs[np.arange(len(y_pred)), y_pred][~correct_mask]
    
    metrics['confidence'] = {
        'avg_confidence_correct': float(correct_probs.mean()) if len(correct_probs) > 0 else 0.0,
        'avg_confidence_incorrect': float(incorrect_probs.mean()) if len(incorrect_probs) > 0 else 0.0,
        'overall_avg_confidence': float(y_probs.max(axis=1).mean()),
        'confidence_gap': float(correct_probs.mean() - incorrect_probs.mean()) if len(incorrect_probs) > 0 else 0.0
    }
    
    return metrics


def create_visualizations(results, metrics, output_dir):
    """Create comprehensive visualizations."""
    print("\nGenerating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = results['labels']
    y_pred = results['predictions']
    y_probs = results['probabilities']
    
    # 1. Confusion Matrix (Detailed)
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = np.array(metrics['confusion_matrix'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add accuracy annotations
    for i in range(4):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total * 100 if total > 0 else 0
        ax.text(4.3, i + 0.5, f'{acc:.1f}%', va='center', fontsize=10, color='#666')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    colors = [COLORS['glioma'], COLORS['meningioma'], COLORS['pituitary'], COLORS['no_tumor']]
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = metrics['roc_auc'][name]
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # 3. Per-Class Metrics Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['precision', 'recall', 'f1_score']
    titles = ['Precision', 'Recall', 'F1-Score']
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        values = [metrics['per_class'][name][metric] for name in CLASS_NAMES]
        colors_list = [COLORS['glioma'], COLORS['meningioma'], COLORS['pituitary'], COLORS['no_tumor']]
        
        bars = ax.bar(CLASS_NAMES, values, color=colors_list, edgecolor='white', linewidth=1)
        ax.set_ylim([0, 1.1])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=10)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.axhline(y=np.mean(values), color='#666', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('Per-Class Performance Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # 4. Confidence Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    correct_mask = y_true == y_pred
    correct_conf = y_probs[np.arange(len(y_pred)), y_pred][correct_mask]
    incorrect_conf = y_probs[np.arange(len(y_pred)), y_pred][~correct_mask]
    
    # Histogram
    ax = axes[0]
    ax.hist(correct_conf, bins=30, alpha=0.7, label=f'Correct (n={len(correct_conf)})', 
            color=COLORS['pituitary'], edgecolor='white')
    ax.hist(incorrect_conf, bins=30, alpha=0.7, label=f'Incorrect (n={len(incorrect_conf)})', 
            color=COLORS['glioma'], edgecolor='white')
    ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axvline(x=correct_conf.mean(), color=COLORS['pituitary'], linestyle='--', lw=2)
    ax.axvline(x=incorrect_conf.mean() if len(incorrect_conf) > 0 else 0, 
               color=COLORS['glioma'], linestyle='--', lw=2)
    
    # Box plot by class
    ax = axes[1]
    conf_by_class = [y_probs[y_true == i, i] for i in range(4)]
    bp = ax.boxplot(conf_by_class, labels=CLASS_NAMES, patch_artist=True)
    colors_list = [COLORS['glioma'], COLORS['meningioma'], COLORS['pituitary'], COLORS['no_tumor']]
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Confidence (True Class)', fontsize=11, fontweight='bold')
    ax.set_title('Confidence by True Class', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # 5. Summary Dashboard
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('TEST SET EVALUATION REPORT', fontsize=18, fontweight='bold', y=0.98)
    
    # Overall metrics panel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    overall = metrics['overall']
    text = f"""
    OVERALL METRICS
    ─────────────────
    Accuracy: {overall['accuracy']*100:.2f}%
    Macro F1: {overall['macro_f1']:.4f}
    Weighted F1: {overall['weighted_f1']:.4f}
    
    Precision: {overall['macro_precision']:.4f}
    Recall: {overall['macro_recall']:.4f}
    
    Samples: {overall['total_samples']}
    Correct: {overall['correct_predictions']}
    Errors: {overall['incorrect_predictions']}
    """
    ax1.text(0.1, 0.9, text, transform=ax1.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='#ddd'))
    
    # Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1:])
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 11, 'weight': 'bold'})
    ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=10)
    ax2.set_ylabel('True', fontsize=10)
    
    # ROC curves
    ax3 = fig.add_subplot(gs[1, :2])
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    colors_list = [COLORS['glioma'], COLORS['meningioma'], COLORS['pituitary'], COLORS['no_tumor']]
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors_list)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = metrics['roc_auc'][name]
        ax3.plot(fpr, tpr, color=color, lw=2, label=f'{name} ({roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax3.set_xlabel('FPR', fontsize=10, fontweight='bold')
    ax3.set_ylabel('TPR', fontsize=10, fontweight='bold')
    ax3.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Per-class accuracy
    ax4 = fig.add_subplot(gs[1, 2])
    accuracies = [metrics['per_class'][name]['accuracy'] for name in CLASS_NAMES]
    bars = ax4.barh(CLASS_NAMES, accuracies, color=colors_list, edgecolor='white')
    ax4.set_xlim([0, 1.1])
    ax4.set_xlabel('Accuracy', fontsize=10, fontweight='bold')
    ax4.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, accuracies):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=9, fontweight='bold')
    
    # Confidence analysis
    ax5 = fig.add_subplot(gs[2, :2])
    correct_conf = y_probs[np.arange(len(y_pred)), y_pred][correct_mask]
    incorrect_conf = y_probs[np.arange(len(y_pred)), y_pred][~correct_mask]
    ax5.hist(correct_conf, bins=25, alpha=0.7, label='Correct', color=COLORS['pituitary'])
    ax5.hist(incorrect_conf, bins=25, alpha=0.7, label='Incorrect', color=COLORS['glioma'])
    ax5.set_xlabel('Confidence', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax5.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    
    # AUC summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    auc_text = "ROC-AUC SCORES\n─────────────────\n"
    for name in CLASS_NAMES:
        auc_text += f"{name}: {metrics['roc_auc'][name]:.4f}\n"
    auc_text += f"\nMacro Avg: {metrics['roc_auc']['macro_avg']:.4f}"
    auc_text += f"\n\nCONFIDENCE\n─────────────────\n"
    auc_text += f"Correct: {metrics['confidence']['avg_confidence_correct']:.3f}\n"
    auc_text += f"Incorrect: {metrics['confidence']['avg_confidence_incorrect']:.3f}\n"
    auc_text += f"Gap: {metrics['confidence']['confidence_gap']:.3f}"
    ax6.text(0.1, 0.9, auc_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='#ddd'))
    
    plt.savefig(output_dir / 'test_evaluation_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    # 6. Metrics Summary Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['Metric', 'Glioma', 'Meningioma', 'Pituitary', 'No Tumor', 'Average']
    ]
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        row = [metric.replace('_', ' ').title()]
        values = []
        for name in CLASS_NAMES:
            val = metrics['per_class'][name][metric]
            row.append(f'{val:.4f}')
            values.append(val)
        row.append(f'{np.mean(values):.4f}')
        table_data.append(row)
    
    # Add support
    row = ['Support']
    total = 0
    for name in CLASS_NAMES:
        val = metrics['per_class'][name]['support']
        row.append(str(val))
        total += val
    row.append(str(total))
    table_data.append(row)
    
    # Add AUC
    row = ['ROC-AUC']
    values = []
    for name in CLASS_NAMES:
        val = metrics['roc_auc'][name]
        row.append(f'{val:.4f}')
        values.append(val)
    row.append(f'{np.mean(values):.4f}')
    table_data.append(row)
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.15] + [0.12]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(6):
        table[(0, j)].set_facecolor('#2D2D2D')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style first column
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 0)].set_text_props(fontweight='bold')
    
    ax.set_title('Test Set Evaluation Metrics', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'metrics_table.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  Saved visualizations to {output_dir}/")


def main():
    print("=" * 70)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    checkpoint_path = Path("checkpoints/densenet121_fixed.pt")
    manifest_path = Path("data/manifests/manifest_fixed.parquet")
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return
    
    # Load model
    model, checkpoint = load_model(checkpoint_path, device)
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load test data
    dataloader, test_df = load_test_data(manifest_path)
    
    # Run evaluation
    results = run_evaluation(model, dataloader, device)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {metrics['overall']['accuracy']*100:.2f}%")
    print(f"Macro F1-Score: {metrics['overall']['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['overall']['weighted_f1']:.4f}")
    
    print("\nPer-Class Performance:")
    print("-" * 60)
    print(f"{'Class':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name in CLASS_NAMES:
        m = metrics['per_class'][name]
        print(f"{name:<15} {m['accuracy']:>10.1%} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}")
    
    print("\nROC-AUC Scores:")
    for name in CLASS_NAMES:
        print(f"  {name}: {metrics['roc_auc'][name]:.4f}")
    print(f"  Macro Average: {metrics['roc_auc']['macro_avg']:.4f}")
    
    print("\nConfidence Analysis:")
    print(f"  Avg confidence (correct): {metrics['confidence']['avg_confidence_correct']:.4f}")
    print(f"  Avg confidence (incorrect): {metrics['confidence']['avg_confidence_incorrect']:.4f}")
    print(f"  Confidence gap: {metrics['confidence']['confidence_gap']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"{'':>15} " + " ".join([f"{n:>12}" for n in CLASS_NAMES]))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>15} " + " ".join([f"{cm[i,j]:>12}" for j in range(4)]))
    
    # Create visualizations
    create_visualizations(results, metrics, OUTPUT_DIR)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    metrics['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': str(checkpoint_path),
        'manifest': str(manifest_path),
        'device': device,
        'model': 'DenseNet-121',
        'num_classes': 4,
        'test_samples': len(test_df)
    }
    
    results_path = OUTPUT_DIR / 'test_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'filepath': results['filepaths'],
        'true_label': results['labels'],
        'true_class': [CLASS_NAMES[l] for l in results['labels']],
        'predicted_label': results['predictions'],
        'predicted_class': [CLASS_NAMES[p] for p in results['predictions']],
        'correct': results['labels'] == results['predictions'],
        'confidence': results['probabilities'].max(axis=1),
        'prob_glioma': results['probabilities'][:, 0],
        'prob_meningioma': results['probabilities'][:, 1],
        'prob_pituitary': results['probabilities'][:, 2],
        'prob_no_tumor': results['probabilities'][:, 3]
    })
    predictions_path = OUTPUT_DIR / 'test_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("Generated files:")
    for f in OUTPUT_DIR.glob('*'):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
