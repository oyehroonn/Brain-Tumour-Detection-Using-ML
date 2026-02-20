"""
EGU-Fusion++ implementation for OncoSense.
Entropy-Guided Uncertainty-weighted fusion with abstention.
"""

from typing import Dict, List, Tuple, Optional
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .uncertainty import predictive_entropy, mc_dropout_inference


def load_config(config_path: str = "configs/fusion.yaml") -> dict:
    """Load fusion configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class EGUFusion(nn.Module):
    """
    Entropy-Guided Uncertainty Fusion (EGU-Fusion++).
    
    Fuses predictions from multiple models using uncertainty-based weighting.
    Models with lower uncertainty get higher weight.
    
    w_m(x) = exp(-α * u_m(x)) / Σ_k exp(-α * u_k(x))
    p(y|x) = Σ_m w_m(x) * p_m(y|x)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        uncertainty_metric: str = "entropy",
        max_prob_threshold: float = 0.6,
        entropy_threshold: float = 1.2,
        abstention_mode: str = "or"
    ):
        """
        Initialize EGU-Fusion.
        
        Args:
            alpha: Temperature for uncertainty weighting (higher = more aggressive).
            uncertainty_metric: Metric to use ("entropy" or "variance").
            max_prob_threshold: Abstain if max_prob below this.
            entropy_threshold: Abstain if entropy above this.
            abstention_mode: "or" (either condition) or "and" (both conditions).
        """
        super().__init__()
        self.alpha = alpha
        self.uncertainty_metric = uncertainty_metric
        self.max_prob_threshold = max_prob_threshold
        self.entropy_threshold = entropy_threshold
        self.abstention_mode = abstention_mode
    
    def compute_weights(
        self,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fusion weights from uncertainties.
        
        Args:
            uncertainties: Uncertainty values (B, M) for M models.
            
        Returns:
            Normalized weights (B, M).
        """
        # Softmax with negative uncertainty (lower uncertainty = higher weight)
        weights = F.softmax(-self.alpha * uncertainties, dim=1)
        return weights
    
    def fuse(
        self,
        probs_list: List[torch.Tensor],
        uncertainties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse predictions from multiple models.
        
        Args:
            probs_list: List of probability tensors (B, C) for each model.
            uncertainties: Uncertainty tensor (B, M).
            
        Returns:
            Tuple of (fused_probs (B, C), weights (B, M)).
        """
        # Stack probabilities: (B, M, C)
        stacked_probs = torch.stack(probs_list, dim=1)
        
        # Compute weights: (B, M)
        weights = self.compute_weights(uncertainties)
        
        # Weighted sum: (B, C)
        fused_probs = torch.einsum("bmc,bm->bc", stacked_probs, weights)
        
        return fused_probs, weights
    
    def should_abstain(
        self,
        fused_probs: torch.Tensor,
        fused_entropy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Determine whether to abstain from prediction.
        
        Args:
            fused_probs: Fused probability tensor (B, C).
            fused_entropy: Pre-computed entropy (optional).
            
        Returns:
            Boolean tensor (B,) indicating abstention.
        """
        max_prob = fused_probs.max(dim=1).values
        
        if fused_entropy is None:
            fused_entropy = predictive_entropy(fused_probs)
        
        low_confidence = max_prob < self.max_prob_threshold
        high_entropy = fused_entropy > self.entropy_threshold
        
        if self.abstention_mode == "or":
            return low_confidence | high_entropy
        else:
            return low_confidence & high_entropy
    
    def forward(
        self,
        probs_list: List[torch.Tensor],
        uncertainties: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with fusion and abstention.
        
        Args:
            probs_list: List of probability tensors from each model.
            uncertainties: Uncertainty tensor for each model.
            
        Returns:
            Dictionary with fused predictions and metadata.
        """
        fused_probs, weights = self.fuse(probs_list, uncertainties)
        fused_entropy = predictive_entropy(fused_probs)
        abstain = self.should_abstain(fused_probs, fused_entropy)
        
        return {
            "fused_probs": fused_probs,
            "predictions": fused_probs.argmax(dim=1),
            "max_prob": fused_probs.max(dim=1).values,
            "entropy": fused_entropy,
            "weights": weights,
            "abstain": abstain
        }


def ensemble_inference(
    models: Dict[str, nn.Module],
    x: torch.Tensor,
    num_mc_passes: int = 20,
    calibrated: bool = True,
    device: str = "cuda"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Run MC-dropout inference on ensemble of models.
    
    Args:
        models: Dictionary mapping model names to models.
        x: Input tensor (B, C, H, W).
        num_mc_passes: Number of MC-dropout passes per model.
        calibrated: Whether to use calibrated predictions.
        device: Device to run on.
        
    Returns:
        Tuple of (probs_dict, uncertainties_dict).
    """
    probs_dict = {}
    uncertainties_dict = {}
    
    x = x.to(device)
    
    for name, model in models.items():
        model = model.to(device)
        probs, _, unc = mc_dropout_inference(
            model, x, num_mc_passes, calibrated
        )
        probs_dict[name] = probs
        uncertainties_dict[name] = unc
    
    return probs_dict, uncertainties_dict


def fused_ensemble_inference(
    models: Dict[str, nn.Module],
    x: torch.Tensor,
    fusion: EGUFusion,
    num_mc_passes: int = 20,
    calibrated: bool = True,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Run full EGU-Fusion++ inference on input batch.
    
    Args:
        models: Dictionary of ensemble models.
        x: Input tensor.
        fusion: EGUFusion module.
        num_mc_passes: Number of MC passes.
        calibrated: Use calibrated predictions.
        device: Device to run on.
        
    Returns:
        Dictionary with fused predictions and per-model outputs.
    """
    # Get per-model predictions
    probs_dict, unc_dict = ensemble_inference(
        models, x, num_mc_passes, calibrated, device
    )
    
    # Extract probabilities and uncertainties
    model_names = list(models.keys())
    probs_list = [probs_dict[name] for name in model_names]
    
    # Get uncertainty metric for fusion
    if fusion.uncertainty_metric == "entropy":
        uncertainties = torch.stack([
            unc_dict[name]["entropy"] for name in model_names
        ], dim=1)
    else:
        uncertainties = torch.stack([
            unc_dict[name]["variance"] for name in model_names
        ], dim=1)
    
    # Run fusion
    result = fusion(probs_list, uncertainties)
    
    # Add per-model info
    result["per_model_probs"] = probs_dict
    result["per_model_uncertainties"] = unc_dict
    result["model_names"] = model_names
    
    return result


def compute_risk_coverage_curve(
    fused_probs: np.ndarray,
    labels: np.ndarray,
    abstain_scores: np.ndarray,
    coverage_points: Optional[List[float]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute risk-coverage curve for selective prediction.
    
    Args:
        fused_probs: Fused probabilities (N, C).
        labels: True labels (N,).
        abstain_scores: Scores for ranking (higher = more likely to abstain).
        coverage_points: Coverage levels to evaluate.
        
    Returns:
        Dictionary with coverage and risk arrays.
    """
    if coverage_points is None:
        coverage_points = np.linspace(0.1, 1.0, 10)
    
    predictions = fused_probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    
    # Sort by abstain score (descending - highest abstain first)
    sort_idx = np.argsort(-abstain_scores)
    correct_sorted = correct[sort_idx]
    
    n_samples = len(labels)
    risks = []
    coverages = []
    
    for cov in coverage_points:
        n_keep = int(n_samples * cov)
        if n_keep == 0:
            continue
        
        # Keep samples with lowest abstain scores
        kept = correct_sorted[-n_keep:]
        risk = 1 - kept.mean()  # Error rate
        
        risks.append(risk)
        coverages.append(cov)
    
    return {
        "coverage": np.array(coverages),
        "risk": np.array(risks),
        "auc": np.trapz(risks, coverages) if len(risks) > 1 else 0.0
    }


def evaluate_fusion(
    models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    fusion: EGUFusion,
    num_mc_passes: int = 20,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate EGU-Fusion on test set.
    
    Args:
        models: Dictionary of models.
        test_loader: Test data loader.
        fusion: EGUFusion module.
        num_mc_passes: Number of MC passes.
        device: Device to run on.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    all_probs = []
    all_labels = []
    all_abstain_scores = []
    all_abstain = []
    per_model_correct = {name: [] for name in models.keys()}
    
    for batch in tqdm(test_loader, desc="Evaluating fusion"):
        images = batch["image"].to(device)
        labels = batch["label"].numpy()
        
        result = fused_ensemble_inference(
            models, images, fusion, num_mc_passes, True, device
        )
        
        all_probs.append(result["fused_probs"].cpu().numpy())
        all_labels.append(labels)
        all_abstain_scores.append(result["entropy"].cpu().numpy())
        all_abstain.append(result["abstain"].cpu().numpy())
        
        # Per-model accuracy
        for name in models.keys():
            preds = result["per_model_probs"][name].argmax(dim=1).cpu().numpy()
            per_model_correct[name].extend((preds == labels).tolist())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_abstain_scores = np.concatenate(all_abstain_scores)
    all_abstain = np.concatenate(all_abstain)
    
    # Compute metrics
    predictions = all_probs.argmax(axis=1)
    accuracy = (predictions == all_labels).mean()
    
    # Accuracy on non-abstained samples
    non_abstain_mask = ~all_abstain
    if non_abstain_mask.sum() > 0:
        accuracy_non_abstain = (
            predictions[non_abstain_mask] == all_labels[non_abstain_mask]
        ).mean()
    else:
        accuracy_non_abstain = 0.0
    
    # Risk-coverage curve
    rc_curve = compute_risk_coverage_curve(
        all_probs, all_labels, all_abstain_scores
    )
    
    metrics = {
        "accuracy": accuracy,
        "accuracy_non_abstain": accuracy_non_abstain,
        "abstention_rate": all_abstain.mean(),
        "risk_coverage_auc": rc_curve["auc"]
    }
    
    # Per-model accuracy
    for name in models.keys():
        metrics[f"accuracy_{name}"] = np.mean(per_model_correct[name])
    
    return metrics


if __name__ == "__main__":
    # Test EGU-Fusion
    fusion = EGUFusion(alpha=1.0)
    
    # Simulate 3 models, batch of 4, 4 classes
    batch_size = 4
    num_classes = 4
    num_models = 3
    
    probs_list = [
        F.softmax(torch.randn(batch_size, num_classes), dim=1)
        for _ in range(num_models)
    ]
    uncertainties = torch.rand(batch_size, num_models)
    
    result = fusion(probs_list, uncertainties)
    
    print("EGU-Fusion Test:")
    print(f"  Fused probs shape: {result['fused_probs'].shape}")
    print(f"  Predictions: {result['predictions']}")
    print(f"  Max prob: {result['max_prob']}")
    print(f"  Entropy: {result['entropy']}")
    print(f"  Weights: {result['weights']}")
    print(f"  Abstain: {result['abstain']}")
