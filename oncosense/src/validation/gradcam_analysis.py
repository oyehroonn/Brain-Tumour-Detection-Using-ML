#!/usr/bin/env python3
"""
Experiment 5: Grad-CAM Analysis

Visualizes what regions the model actually uses for predictions.
If model focuses on edges, corners, or non-tumor regions, it's learning spurious features.

Metrics computed:
- Center Region Attention (should be high for tumor detection)
- Edge Attention Ratio (high = suspicious)
- Attention Consistency across classes
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR
from src.models.backbones import get_backbone, load_checkpoint
from src.xai.gradcam import GradCAMGenerator
from src.data.transforms import get_val_transforms, load_image


CLASS_NAMES = {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}


def compute_attention_metrics(heatmap: np.ndarray) -> Dict:
    """
    Compute attention distribution metrics from Grad-CAM heatmap.
    
    Args:
        heatmap: 2D numpy array (H, W) with values in [0, 1]
        
    Returns:
        Dictionary of attention metrics
    """
    h, w = heatmap.shape
    
    # Define regions
    center_size = 0.5  # Center 50% of each dimension
    edge_size = 0.15   # Outer 15% of each dimension
    
    ch, cw = int(h * center_size), int(w * center_size)
    eh, ew = int(h * edge_size), int(w * edge_size)
    
    # Center region
    center_start_h = (h - ch) // 2
    center_start_w = (w - cw) // 2
    center_region = heatmap[center_start_h:center_start_h+ch, 
                           center_start_w:center_start_w+cw]
    
    # Edge regions (top, bottom, left, right bands)
    top_edge = heatmap[:eh, :]
    bottom_edge = heatmap[-eh:, :]
    left_edge = heatmap[:, :ew]
    right_edge = heatmap[:, -ew:]
    
    # Compute metrics
    total_attention = heatmap.sum() + 1e-10
    
    center_attention = center_region.sum() / total_attention
    
    edge_attention = (top_edge.sum() + bottom_edge.sum() + 
                     left_edge.sum() + right_edge.sum()) / total_attention
    # Avoid double counting corners
    corners = (heatmap[:eh, :ew].sum() + heatmap[:eh, -ew:].sum() +
              heatmap[-eh:, :ew].sum() + heatmap[-eh:, -ew:].sum())
    edge_attention = (edge_attention * total_attention - corners) / total_attention
    
    # Peak location
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_y_norm = peak_idx[0] / h
    peak_x_norm = peak_idx[1] / w
    peak_in_center = (0.25 < peak_y_norm < 0.75) and (0.25 < peak_x_norm < 0.75)
    
    # Attention spread (entropy-like measure)
    heatmap_norm = heatmap / (heatmap.sum() + 1e-10)
    entropy = -np.sum(heatmap_norm * np.log(heatmap_norm + 1e-10))
    max_entropy = np.log(h * w)
    spread = entropy / max_entropy  # Normalized [0, 1]
    
    # High attention area (percentage above threshold)
    threshold = 0.5
    high_attention_area = (heatmap >= threshold).sum() / (h * w)
    
    return {
        'center_attention_ratio': float(center_attention),
        'edge_attention_ratio': float(edge_attention),
        'peak_in_center': bool(peak_in_center),
        'peak_location': (float(peak_y_norm), float(peak_x_norm)),
        'attention_spread': float(spread),
        'high_attention_area': float(high_attention_area),
        'max_activation': float(heatmap.max()),
        'mean_activation': float(heatmap.mean())
    }


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


def analyze_single_image(
    model: torch.nn.Module,
    cam_generator: GradCAMGenerator,
    image_path: str,
    transform,
    device: str,
    true_label: int
) -> Dict:
    """Analyze a single image with Grad-CAM."""
    # Load and preprocess
    original_img = load_image(image_path, convert_rgb=True)
    transformed = transform(image=original_img)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()
    
    # Generate Grad-CAM
    heatmap = cam_generator.generate(input_tensor, target_class=pred_class)
    
    # Compute metrics
    metrics = compute_attention_metrics(heatmap)
    metrics['true_label'] = true_label
    metrics['predicted_label'] = pred_class
    metrics['correct'] = pred_class == true_label
    metrics['confidence'] = float(confidence)
    metrics['image_path'] = image_path
    
    return metrics, heatmap, original_img


def run_experiment(
    checkpoint_path: str = None,
    manifest_path: str = None,
    num_samples: int = 100,
    device: str = None
) -> Dict:
    """
    Run the complete Grad-CAM analysis experiment.
    
    Args:
        checkpoint_path: Path to model checkpoint
        manifest_path: Path to manifest parquet
        num_samples: Number of images to analyze
        device: Device to use
    """
    os.chdir(Path(__file__).parent.parent.parent)
    
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/densenet121_best.pt"
    if manifest_path is None:
        manifest_path = "data/manifests/manifest.parquet"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = VALIDATION_RESULTS_DIR / "experiment_5_gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXPERIMENT 5: Grad-CAM Analysis")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {num_samples}")
    
    # Load model
    print("\nLoading model...")
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None
    
    model = load_model(checkpoint_path, device)
    
    # Initialize Grad-CAM generator
    print("Initializing Grad-CAM generator...")
    cam_generator = GradCAMGenerator(
        model=model,
        backbone_name="densenet121",
        device=device,
        use_cuda=(device == "cuda")
    )
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(manifest_path)
    val_df = df[df['split'] == 'val'].copy()
    
    # Sample images (stratified by class)
    samples_per_class = num_samples // 4
    sampled_dfs = []
    for label in range(4):
        class_df = val_df[val_df['label'] == label]
        n = min(samples_per_class, len(class_df))
        sampled_dfs.append(class_df.sample(n=n, random_state=42))
    
    sample_df = pd.concat(sampled_dfs).reset_index(drop=True)
    print(f"Analyzing {len(sample_df)} images...")
    
    # Get transforms
    transform = get_val_transforms()
    
    # Analyze images
    all_metrics = []
    sample_visualizations = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing"):
        image_path = f"data/{row['filepath']}"
        
        try:
            metrics, heatmap, original_img = analyze_single_image(
                model, cam_generator, image_path, transform, device, row['label']
            )
            all_metrics.append(metrics)
            
            # Save first 5 per class for visualization
            class_count = sum(1 for m in sample_visualizations 
                            if m['true_label'] == row['label'])
            if class_count < 5:
                sample_visualizations.append({
                    'metrics': metrics,
                    'heatmap': heatmap,
                    'original_img': original_img
                })
        except Exception as e:
            print(f"Warning: Failed to analyze {image_path}: {e}")
    
    # Aggregate statistics
    results = aggregate_results(all_metrics)
    results['timestamp'] = datetime.now().isoformat()
    results['num_samples'] = len(all_metrics)
    results['checkpoint'] = checkpoint_path
    
    # Print summary
    print_summary(results)
    
    # Create visualizations
    create_visualizations(results, sample_visualizations, output_dir)
    
    # Save results
    results_file = output_dir / "results.json"
    
    # Remove non-serializable items
    save_results = {k: v for k, v in results.items() if k != 'all_metrics'}
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results


def aggregate_results(all_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across all analyzed images."""
    if not all_metrics:
        return {}
    
    results = {
        'all_metrics': all_metrics,
        'global_stats': {},
        'per_class_stats': {},
        'verdict': {}
    }
    
    # Global statistics
    center_ratios = [m['center_attention_ratio'] for m in all_metrics]
    edge_ratios = [m['edge_attention_ratio'] for m in all_metrics]
    peaks_in_center = [m['peak_in_center'] for m in all_metrics]
    spreads = [m['attention_spread'] for m in all_metrics]
    correct = [m['correct'] for m in all_metrics]
    
    results['global_stats'] = {
        'mean_center_attention': float(np.mean(center_ratios)),
        'std_center_attention': float(np.std(center_ratios)),
        'mean_edge_attention': float(np.mean(edge_ratios)),
        'std_edge_attention': float(np.std(edge_ratios)),
        'peak_in_center_rate': float(np.mean(peaks_in_center)),
        'mean_attention_spread': float(np.mean(spreads)),
        'accuracy': float(np.mean(correct))
    }
    
    # Per-class statistics
    for label in range(4):
        class_metrics = [m for m in all_metrics if m['true_label'] == label]
        if class_metrics:
            results['per_class_stats'][CLASS_NAMES[label]] = {
                'count': len(class_metrics),
                'mean_center_attention': float(np.mean([m['center_attention_ratio'] for m in class_metrics])),
                'mean_edge_attention': float(np.mean([m['edge_attention_ratio'] for m in class_metrics])),
                'peak_in_center_rate': float(np.mean([m['peak_in_center'] for m in class_metrics])),
                'accuracy': float(np.mean([m['correct'] for m in class_metrics]))
            }
    
    return results


def print_summary(results: Dict):
    """Print summary of Grad-CAM analysis."""
    stats = results['global_stats']
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nGlobal Attention Metrics:")
    print(f"  Center Attention: {stats['mean_center_attention']:.2%} ± {stats['std_center_attention']:.2%}")
    print(f"  Edge Attention: {stats['mean_edge_attention']:.2%} ± {stats['std_edge_attention']:.2%}")
    print(f"  Peak in Center: {stats['peak_in_center_rate']:.1%} of images")
    print(f"  Attention Spread: {stats['mean_attention_spread']:.3f}")
    print(f"  Validation Accuracy: {stats['accuracy']:.2%}")
    
    print(f"\nPer-Class Analysis:")
    for class_name, class_stats in results['per_class_stats'].items():
        print(f"\n  {class_name.upper()}:")
        print(f"    Center Attention: {class_stats['mean_center_attention']:.2%}")
        print(f"    Edge Attention: {class_stats['mean_edge_attention']:.2%}")
        print(f"    Peak in Center: {class_stats['peak_in_center_rate']:.1%}")
        print(f"    Class Accuracy: {class_stats['accuracy']:.2%}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    center_att = stats['mean_center_attention']
    edge_att = stats['mean_edge_attention']
    peak_center = stats['peak_in_center_rate']
    
    issues = []
    
    if center_att < 0.3:
        issues.append(f"Low center attention ({center_att:.1%}) - model may not focus on tumor region")
    
    if edge_att > 0.3:
        issues.append(f"High edge attention ({edge_att:.1%}) - model may learn border artifacts")
    
    if peak_center < 0.5:
        issues.append(f"Peak rarely in center ({peak_center:.1%}) - suspicious attention pattern")
    
    # Check for class-specific issues
    for class_name, class_stats in results['per_class_stats'].items():
        if class_stats['mean_edge_attention'] > 0.4:
            issues.append(f"Class '{class_name}' has high edge attention ({class_stats['mean_edge_attention']:.1%})")
    
    if issues:
        if len(issues) >= 3:
            verdict = "CRITICAL: Multiple suspicious attention patterns detected!"
            severity = "critical"
        elif len(issues) >= 1:
            verdict = "WARNING: Suspicious attention patterns detected"
            severity = "warning"
        
        explanation = "Issues found:\n" + "\n".join(f"  - {issue}" for issue in issues)
    else:
        verdict = "PASS: Attention patterns appear reasonable"
        severity = "pass"
        explanation = (
            f"Model shows {center_att:.1%} center attention and {peak_center:.1%} "
            f"of peaks in center region. This suggests the model may be focusing "
            f"on relevant tumor regions."
        )
    
    results['verdict'] = {
        'summary': verdict,
        'severity': severity,
        'explanation': explanation,
        'issues': issues
    }
    
    print(f"\n{verdict}")
    print(f"\n{explanation}")


def create_visualizations(results: Dict, sample_vis: List[Dict], output_dir: Path):
    """Create visualizations for Grad-CAM analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Attention distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Center vs Edge attention by class
    classes = list(results['per_class_stats'].keys())
    center_atts = [results['per_class_stats'][c]['mean_center_attention'] for c in classes]
    edge_atts = [results['per_class_stats'][c]['mean_edge_attention'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, center_atts, width, label='Center Attention', color='#27ae60')
    bars2 = axes[0].bar(x + width/2, edge_atts, width, label='Edge Attention', color='#e74c3c')
    
    axes[0].set_ylabel('Attention Ratio', fontsize=12, fontweight='bold')
    axes[0].set_title('Attention Distribution by Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=15)
    axes[0].legend()
    axes[0].axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
    
    # Peak location scatter (if we have enough data)
    all_metrics = results.get('all_metrics', [])
    if all_metrics:
        peak_y = [m['peak_location'][0] for m in all_metrics]
        peak_x = [m['peak_location'][1] for m in all_metrics]
        colors = [m['true_label'] for m in all_metrics]
        
        scatter = axes[1].scatter(peak_x, peak_y, c=colors, cmap='tab10', alpha=0.6, s=50)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel('X Position (normalized)', fontsize=12)
        axes[1].set_ylabel('Y Position (normalized)', fontsize=12)
        axes[1].set_title('Peak Attention Locations', fontsize=14, fontweight='bold')
        
        # Add center region box
        rect = plt.Rectangle((0.25, 0.25), 0.5, 0.5, fill=False, 
                            edgecolor='green', linewidth=2, linestyle='--')
        axes[1].add_patch(rect)
        axes[1].text(0.5, 0.77, 'Center Region', ha='center', fontsize=10, color='green')
        
        plt.colorbar(scatter, ax=axes[1], label='Class')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Sample Grad-CAM visualizations (grid)
    if sample_vis:
        n_samples = min(16, len(sample_vis))
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(20, n_rows * 3))
        
        for i in range(n_samples):
            row = i // n_cols
            col = (i % n_cols) * 2
            
            vis = sample_vis[i]
            orig_img = vis['original_img']
            heatmap = vis['heatmap']
            metrics = vis['metrics']
            
            # Original image
            if n_rows > 1:
                ax_orig = axes[row, col]
                ax_heat = axes[row, col + 1]
            else:
                ax_orig = axes[col]
                ax_heat = axes[col + 1]
            
            ax_orig.imshow(orig_img)
            ax_orig.set_title(f"True: {CLASS_NAMES[metrics['true_label']]}", fontsize=9)
            ax_orig.axis('off')
            
            # Heatmap overlay
            if orig_img.max() > 1:
                orig_norm = orig_img.astype(np.float32) / 255.0
            else:
                orig_norm = orig_img
            
            # Resize heatmap
            import cv2
            heatmap_resized = cv2.resize(heatmap, (orig_norm.shape[1], orig_norm.shape[0]))
            
            # Create simple overlay
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.5 * orig_norm + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            
            ax_heat.imshow(overlay)
            pred_str = CLASS_NAMES[metrics['predicted_label']]
            correct_str = "✓" if metrics['correct'] else "✗"
            ax_heat.set_title(f"Pred: {pred_str} {correct_str}\nCenter: {metrics['center_attention_ratio']:.0%}", 
                            fontsize=9)
            ax_heat.axis('off')
        
        # Hide unused axes
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = (i % n_cols) * 2
            if n_rows > 1:
                axes[row, col].axis('off')
                axes[row, col + 1].axis('off')
            else:
                axes[col].axis('off')
                axes[col + 1].axis('off')
        
        plt.suptitle('Sample Grad-CAM Visualizations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_gradcams.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Summary statistics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Center\nAttention', 'Edge\nAttention', 'Peak in\nCenter', 'Accuracy']
    values = [
        results['global_stats']['mean_center_attention'],
        results['global_stats']['mean_edge_attention'],
        results['global_stats']['peak_in_center_rate'],
        results['global_stats']['accuracy']
    ]
    
    colors = ['#27ae60', '#e74c3c', '#3498db', '#9b59b6']
    bars = ax.bar(metrics_names, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Value (ratio)', fontsize=14, fontweight='bold')
    ax.set_title('Grad-CAM Analysis Summary', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    # Add threshold lines
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Attention threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Grad-CAM analysis")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--samples", "-n", type=int, default=100,
                       help="Number of images to analyze")
    parser.add_argument("--device", "-d", type=str, default=None,
                       help="Device to use")
    
    args = parser.parse_args()
    run_experiment(
        checkpoint_path=args.checkpoint,
        num_samples=args.samples,
        device=args.device
    )
