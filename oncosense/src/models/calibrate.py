"""
Model calibration for OncoSense.
Implements temperature scaling and calibration metrics (ECE, Brier score).
"""

from typing import Tuple, Optional, Dict, Any
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


def load_config(config_path: str = "configs/fusion.yaml") -> dict:
    """Load fusion configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error.
    
    ECE = sum_b |B_b|/n * |acc(B_b) - conf(B_b)|
    
    Args:
        probs: Predicted probabilities (N, C).
        labels: True labels (N,).
        n_bins: Number of confidence bins.
        
    Returns:
        ECE value.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
    
    return ece


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).
    
    Args:
        probs: Predicted probabilities (N, C).
        labels: True labels (N,).
        
    Returns:
        Brier score.
    """
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute negative log-likelihood.
    
    Args:
        probs: Predicted probabilities (N, C).
        labels: True labels (N,).
        
    Returns:
        NLL value.
    """
    eps = 1e-8
    correct_probs = probs[np.arange(len(labels)), labels]
    return -np.mean(np.log(correct_probs + eps))


class TemperatureScaling:
    """
    Temperature scaling calibration.
    
    Learns a single scalar temperature T to calibrate probabilities:
    p_calibrated = softmax(logits / T)
    """
    
    def __init__(self, init_temperature: float = 1.5):
        self.temperature = init_temperature
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        optimize_metric: str = "nll"
    ) -> float:
        """
        Find optimal temperature using validation set.
        
        Args:
            logits: Model logits (N, C).
            labels: True labels (N,).
            optimize_metric: Metric to optimize ("nll", "ece", "brier").
            
        Returns:
            Optimal temperature value.
        """
        def objective(T):
            T = max(T[0], 0.01)  # Ensure positive
            scaled_logits = logits / T
            probs = self._softmax(scaled_logits)
            
            if optimize_metric == "nll":
                return compute_nll(probs, labels)
            elif optimize_metric == "ece":
                return compute_ece(probs, labels)
            elif optimize_metric == "brier":
                return compute_brier_score(probs, labels)
            else:
                raise ValueError(f"Unknown metric: {optimize_metric}")
        
        result = minimize(
            objective,
            [self.temperature],
            method="L-BFGS-B",
            bounds=[(0.01, 10.0)]
        )
        
        self.temperature = result.x[0]
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits (N, C).
            
        Returns:
            Calibrated probabilities.
        """
        scaled = logits / self.temperature
        return self._softmax(scaled)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)


def calibrate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    optimize_metric: str = "nll"
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate a model using temperature scaling.
    
    Args:
        model: PyTorch model.
        val_loader: Validation data loader.
        device: Device to run on.
        optimize_metric: Metric to optimize.
        
    Returns:
        Tuple of (optimal_temperature, metrics_dict).
    """
    model.eval()
    model = model.to(device)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting logits"):
            images = batch["image"].to(device)
            labels = batch["label"]
            
            logits = model(images)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    
    # Compute metrics before calibration
    probs_uncalibrated = TemperatureScaling._softmax(logits)
    metrics_before = {
        "ece_before": compute_ece(probs_uncalibrated, labels),
        "brier_before": compute_brier_score(probs_uncalibrated, labels),
        "nll_before": compute_nll(probs_uncalibrated, labels)
    }
    
    # Fit temperature scaling
    ts = TemperatureScaling()
    optimal_temp = ts.fit(logits, labels, optimize_metric)
    
    # Compute metrics after calibration
    probs_calibrated = ts.calibrate(logits)
    metrics_after = {
        "ece_after": compute_ece(probs_calibrated, labels),
        "brier_after": compute_brier_score(probs_calibrated, labels),
        "nll_after": compute_nll(probs_calibrated, labels),
        "temperature": optimal_temp
    }
    
    metrics = {**metrics_before, **metrics_after}
    
    # Print summary
    print("\n" + "=" * 50)
    print("Calibration Results")
    print("=" * 50)
    print(f"Optimal Temperature: {optimal_temp:.4f}")
    print(f"\nECE:   {metrics['ece_before']:.4f} → {metrics['ece_after']:.4f}")
    print(f"Brier: {metrics['brier_before']:.4f} → {metrics['brier_after']:.4f}")
    print(f"NLL:   {metrics['nll_before']:.4f} → {metrics['nll_after']:.4f}")
    
    return optimal_temp, metrics


def get_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get calibration curve data for plotting.
    
    Args:
        probs: Predicted probabilities (N, C).
        labels: True labels (N,).
        n_bins: Number of bins.
        
    Returns:
        Tuple of (bin_centers, accuracies, counts).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accs = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = in_bin.sum()
        
        if count > 0:
            bin_accs.append(accuracies[in_bin].mean())
        else:
            bin_accs.append(0)
        bin_counts.append(count)
    
    return bin_centers, np.array(bin_accs), np.array(bin_counts)


def plot_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None
):
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        probs: Predicted probabilities.
        labels: True labels.
        n_bins: Number of bins.
        title: Plot title.
        save_path: Path to save figure.
    """
    import matplotlib.pyplot as plt
    
    bin_centers, bin_accs, bin_counts = get_calibration_curve(probs, labels, n_bins)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.7, 
            edgecolor='black', label='Model')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Histogram of predictions
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.7, 
            edgecolor='black', color='gray')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration curve to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test calibration metrics
    np.random.seed(42)
    
    # Simulate uncalibrated model (overconfident)
    n_samples = 1000
    n_classes = 4
    
    # Generate pseudo-logits
    logits = np.random.randn(n_samples, n_classes) * 2
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Make some predictions correct
    for i in range(n_samples):
        if np.random.random() > 0.3:  # 70% accuracy
            logits[i, labels[i]] += 3
    
    probs = TemperatureScaling._softmax(logits)
    
    print("Before calibration:")
    print(f"  ECE: {compute_ece(probs, labels):.4f}")
    print(f"  Brier: {compute_brier_score(probs, labels):.4f}")
    print(f"  NLL: {compute_nll(probs, labels):.4f}")
    
    ts = TemperatureScaling()
    ts.fit(logits, labels)
    
    probs_cal = ts.calibrate(logits)
    
    print(f"\nAfter calibration (T={ts.temperature:.3f}):")
    print(f"  ECE: {compute_ece(probs_cal, labels):.4f}")
    print(f"  Brier: {compute_brier_score(probs_cal, labels):.4f}")
    print(f"  NLL: {compute_nll(probs_cal, labels):.4f}")
