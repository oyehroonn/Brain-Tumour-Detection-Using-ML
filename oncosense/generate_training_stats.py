#!/usr/bin/env python3
"""
Generate Training Statistics Visualizations for OncoSense DenseNet121 Model.
Creates learning curves, KPI dashboards, and metrics tables.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import matplotlib.gridspec as gridspec

# Training data from completed training run (Epochs 1-6)
TRAINING_DATA = {
    "epochs": [1, 2, 3, 4, 5, 6],
    "train_loss": [0.2449, 0.0950, 0.0752, 0.0513, 0.0512, 0.0384],
    "train_acc": [80.10, 91.47, 93.38, 95.42, 95.46, 96.20],
    "train_f1": [0.7970, 0.9130, 0.9325, 0.9533, 0.9536, 0.9613],
    "val_loss": [0.0928, 0.0749, 0.0487, 0.0472, 0.0572, 0.0429],
    "val_acc": [92.01, 94.25, 95.53, 96.17, 96.33, 98.08],
    "val_f1": [0.9178, 0.9424, 0.9539, 0.9611, 0.9622, 0.9802],
}

DATASET_INFO = {
    "train_samples": 4612,
    "val_samples": 626,
    "test_samples": 1359,
    "total_samples": 6597,
    "classes": ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
    "model_name": "DenseNet121",
}

OUTPUT_DIR = Path("training_stats")


def create_output_directory():
    """Create the output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")


def plot_learning_curve_accuracy():
    """Generate learning curve for accuracy (train vs validation)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = TRAINING_DATA["epochs"]
    train_acc = TRAINING_DATA["train_acc"]
    val_acc = TRAINING_DATA["val_acc"]
    
    # Plot lines
    ax.plot(epochs, train_acc, 'b-o', linewidth=2, markersize=8, label='Training Accuracy')
    ax.plot(epochs, val_acc, 'g-s', linewidth=2, markersize=8, label='Validation Accuracy')
    
    # Fill area between curves
    ax.fill_between(epochs, train_acc, val_acc, alpha=0.2, color='gray')
    
    # Add value annotations
    for i, (t, v) in enumerate(zip(train_acc, val_acc)):
        ax.annotate(f'{t:.1f}%', (epochs[i], t), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9, color='blue')
        ax.annotate(f'{v:.1f}%', (epochs[i], v), textcoords="offset points", 
                   xytext=(0, -15), ha='center', fontsize=9, color='green')
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('DenseNet121 Learning Curve - Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(75, 100)
    ax.set_xticks(epochs)
    
    # Add best epoch marker
    best_idx = np.argmax(val_acc)
    ax.axvline(x=epochs[best_idx], color='red', linestyle='--', alpha=0.5, label='Best Epoch')
    ax.scatter([epochs[best_idx]], [val_acc[best_idx]], color='red', s=200, zorder=5, 
               marker='*', label=f'Best: {val_acc[best_idx]:.2f}%')
    
    # Add text box with final metrics
    textstr = f'Best Validation Accuracy: {max(val_acc):.2f}%\nFinal Training Accuracy: {train_acc[-1]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "learning_curve_accuracy.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curve_loss():
    """Generate learning curve for loss (train vs validation)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = TRAINING_DATA["epochs"]
    train_loss = TRAINING_DATA["train_loss"]
    val_loss = TRAINING_DATA["val_loss"]
    
    # Plot lines
    ax.plot(epochs, train_loss, 'r-o', linewidth=2, markersize=8, label='Training Loss')
    ax.plot(epochs, val_loss, 'purple', linestyle='-', marker='s', linewidth=2, 
            markersize=8, label='Validation Loss')
    
    # Fill area between curves
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.2, color='orange')
    
    # Add value annotations
    for i, (t, v) in enumerate(zip(train_loss, val_loss)):
        ax.annotate(f'{t:.4f}', (epochs[i], t), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9, color='red')
        ax.annotate(f'{v:.4f}', (epochs[i], v), textcoords="offset points", 
                   xytext=(0, -15), ha='center', fontsize=9, color='purple')
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('DenseNet121 Learning Curve - Loss', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(epochs)
    
    # Add best epoch marker
    best_idx = np.argmin(val_loss)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5)
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], color='green', s=200, zorder=5, 
               marker='*')
    
    # Add text box with final metrics
    textstr = f'Best Validation Loss: {min(val_loss):.4f}\nFinal Training Loss: {train_loss[-1]:.4f}'
    props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "learning_curve_loss.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_kpi_dashboard():
    """Generate a multi-panel KPI dashboard."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Final Accuracy Bar
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Train Acc', 'Val Acc']
    values = [TRAINING_DATA["train_acc"][-1], TRAINING_DATA["val_acc"][-1]]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Final Accuracy', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax1.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% Target')
    ax1.legend(loc='lower right')
    
    # Panel 2: F1 Score Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    f1_metrics = ['Train F1', 'Val F1']
    f1_values = [TRAINING_DATA["train_f1"][-1], TRAINING_DATA["val_f1"][-1]]
    colors_f1 = ['#e74c3c', '#9b59b6']
    bars2 = ax2.bar(f1_metrics, f1_values, color=colors_f1, edgecolor='black', linewidth=1.5)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('Final F1 Score', fontsize=12, fontweight='bold')
    for bar, val in zip(bars2, f1_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='0.95 Target')
    ax2.legend(loc='lower right')
    
    # Panel 3: Dataset Split Pie Chart
    ax3 = fig.add_subplot(gs[0, 2])
    sizes = [DATASET_INFO["train_samples"], DATASET_INFO["val_samples"], 
             DATASET_INFO["test_samples"]]
    labels = [f'Train\n{sizes[0]:,}', f'Val\n{sizes[1]:,}', f'Test\n{sizes[2]:,}']
    colors_pie = ['#3498db', '#2ecc71', '#f39c12']
    explode = (0.02, 0.02, 0.02)
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                        explode=explode, autopct='%1.1f%%',
                                        shadow=True, startangle=90)
    ax3.set_title('Dataset Split', fontsize=12, fontweight='bold')
    
    # Panel 4: Key Metrics Table
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Value', 'Status'],
        ['Best Validation Accuracy', f'{max(TRAINING_DATA["val_acc"]):.2f}%', 'Excellent'],
        ['Best Validation F1', f'{max(TRAINING_DATA["val_f1"]):.4f}', 'Excellent'],
        ['Best Validation Loss', f'{min(TRAINING_DATA["val_loss"]):.4f}', 'Good'],
        ['Epochs Completed', '6', '-'],
        ['Model Architecture', 'DenseNet121', '-'],
        ['Total Training Samples', f'{DATASET_INFO["train_samples"]:,}', '-'],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color the header row
    for j in range(3):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color status column
    status_colors = {'Excellent': '#2ecc71', 'Good': '#f39c12', '-': '#ecf0f1'}
    for i in range(1, len(table_data)):
        status = table_data[i][2]
        table[(i, 2)].set_facecolor(status_colors.get(status, '#ecf0f1'))
    
    ax4.set_title('Key Performance Indicators', fontsize=12, fontweight='bold', pad=20)
    
    # Panel 5: Model Info Box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    info_text = f"""
    MODEL SUMMARY
    ─────────────────
    Architecture: {DATASET_INFO['model_name']}
    
    Classes:
    • Glioma
    • Meningioma
    • Pituitary
    • No Tumor
    
    Dataset:
    • Total: {DATASET_INFO['total_samples']:,} images
    • Split: 70/10/20
    
    Training:
    • Epochs: 6 (stopped early)
    • Device: CPU (Mac)
    • Time: ~2.5 hours
    """
    
    ax5.text(0.1, 0.95, info_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    # Main title
    fig.suptitle('OncoSense DenseNet121 - Training KPI Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    save_path = OUTPUT_DIR / "model_kpi_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_metrics_table():
    """Generate a professional metrics table visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Table data
    headers = ['Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Val Loss', 'Val Acc', 'Val F1', 'Saved']
    
    data = []
    for i, epoch in enumerate(TRAINING_DATA["epochs"]):
        row = [
            str(epoch),
            f'{TRAINING_DATA["train_loss"][i]:.4f}',
            f'{TRAINING_DATA["train_acc"][i]:.2f}%',
            f'{TRAINING_DATA["train_f1"][i]:.4f}',
            f'{TRAINING_DATA["val_loss"][i]:.4f}',
            f'{TRAINING_DATA["val_acc"][i]:.2f}%',
            f'{TRAINING_DATA["val_f1"][i]:.4f}',
            '✓ Best' if i == len(TRAINING_DATA["epochs"]) - 1 else '✓'
        ]
        data.append(row)
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color code cells based on improvement
    for i in range(1, len(data) + 1):
        # Val Acc column (index 5) - highlight best
        val_acc = float(data[i-1][5].replace('%', ''))
        if val_acc >= 98:
            table[(i, 5)].set_facecolor('#27ae60')
            table[(i, 5)].set_text_props(color='white', fontweight='bold')
        elif val_acc >= 95:
            table[(i, 5)].set_facecolor('#a9dfbf')
        
        # Val F1 column (index 6) - highlight best
        val_f1 = float(data[i-1][6])
        if val_f1 >= 0.98:
            table[(i, 6)].set_facecolor('#27ae60')
            table[(i, 6)].set_text_props(color='white', fontweight='bold')
        elif val_f1 >= 0.95:
            table[(i, 6)].set_facecolor('#a9dfbf')
        
        # Saved column - highlight best model
        if 'Best' in data[i-1][7]:
            table[(i, 7)].set_facecolor('#f1c40f')
            table[(i, 7)].set_text_props(fontweight='bold')
        
        # Alternate row colors
        if i % 2 == 0:
            for j in range(len(headers)):
                if table[(i, j)].get_facecolor() == (1.0, 1.0, 1.0, 1.0):
                    table[(i, j)].set_facecolor('#f8f9fa')
    
    # Title and footer
    ax.set_title('DenseNet121 Training Progress - Epoch by Epoch Metrics', 
                 fontsize=14, fontweight='bold', pad=40, y=1.02)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#27ae60', label='Excellent (≥98% / ≥0.98)'),
        mpatches.Patch(facecolor='#a9dfbf', label='Good (≥95% / ≥0.95)'),
        mpatches.Patch(facecolor='#f1c40f', label='Best Model Checkpoint'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=10)
    
    # Add summary text
    summary_text = (
        f"Training Summary: Started at 80.10% accuracy (Epoch 1) → "
        f"Achieved 98.08% accuracy (Epoch 6) | "
        f"Improvement: +17.98 percentage points"
    )
    ax.text(0.5, -0.08, summary_text, transform=ax.transAxes, fontsize=10,
            ha='center', style='italic', color='#666666')
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "training_metrics_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def print_summary():
    """Print a summary of the training statistics."""
    print("\n" + "=" * 60)
    print("TRAINING STATISTICS SUMMARY")
    print("=" * 60)
    print(f"\nModel: {DATASET_INFO['model_name']}")
    print(f"Epochs Completed: {len(TRAINING_DATA['epochs'])}")
    print(f"\nBest Results (Epoch 6):")
    print(f"  Validation Accuracy: {max(TRAINING_DATA['val_acc']):.2f}%")
    print(f"  Validation F1 Score: {max(TRAINING_DATA['val_f1']):.4f}")
    print(f"  Validation Loss: {min(TRAINING_DATA['val_loss']):.4f}")
    print(f"\nDataset:")
    print(f"  Training: {DATASET_INFO['train_samples']:,} samples")
    print(f"  Validation: {DATASET_INFO['val_samples']:,} samples")
    print(f"  Test: {DATASET_INFO['test_samples']:,} samples")
    print(f"\nVisualizations saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


def main():
    """Main function to generate all visualizations."""
    print("Generating Training Statistics Visualizations...")
    print("-" * 50)
    
    # Create output directory
    create_output_directory()
    
    # Generate visualizations
    print("\n[1/4] Generating Learning Curve (Accuracy)...")
    plot_learning_curve_accuracy()
    
    print("[2/4] Generating Learning Curve (Loss)...")
    plot_learning_curve_loss()
    
    print("[3/4] Generating KPI Dashboard...")
    plot_kpi_dashboard()
    
    print("[4/4] Generating Training Metrics Table...")
    plot_training_metrics_table()
    
    # Print summary
    print_summary()
    
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
