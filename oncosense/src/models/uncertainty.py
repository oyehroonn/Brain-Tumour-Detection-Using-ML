"""
Uncertainty estimation for OncoSense.
Implements MC-dropout based uncertainty quantification.
"""

from typing import Dict, Tuple, Optional
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def load_config(config_path: str = "configs/models.yaml") -> dict:
    """Load model configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy of probability distribution.
    
    H(p) = -sum(p * log(p))
    
    Args:
        probs: Probability tensor (B, C) or (B, T, C) for MC samples.
        
    Returns:
        Entropy tensor (B,).
    """
    if probs.dim() == 3:
        # Average over MC samples first
        probs = probs.mean(dim=1)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    
    return entropy


def mutual_information(mc_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute mutual information (epistemic uncertainty) from MC samples.
    
    MI = H(E[p]) - E[H(p)]
    
    Args:
        mc_probs: MC probability samples (B, T, C).
        
    Returns:
        Mutual information tensor (B,).
    """
    # Predictive entropy: H(E[p])
    mean_probs = mc_probs.mean(dim=1)
    h_mean = predictive_entropy(mean_probs)
    
    # Expected entropy: E[H(p)]
    eps = 1e-8
    entropies = -torch.sum(mc_probs * torch.log(mc_probs + eps), dim=-1)
    mean_entropy = entropies.mean(dim=1)
    
    return h_mean - mean_entropy


def prediction_variance(mc_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute variance of predicted class probability across MC samples.
    
    Args:
        mc_probs: MC probability samples (B, T, C).
        
    Returns:
        Variance tensor (B,) - variance of max class probability.
    """
    # Get predicted class for each sample
    mean_probs = mc_probs.mean(dim=1)
    predicted_class = mean_probs.argmax(dim=1)
    
    # Get probability of predicted class across all MC samples
    batch_size = mc_probs.size(0)
    pred_probs = mc_probs[
        torch.arange(batch_size).unsqueeze(1),
        :,
        predicted_class.unsqueeze(1)
    ].squeeze(-1)
    
    return pred_probs.var(dim=1)


def mc_dropout_inference(
    model: nn.Module,
    x: torch.Tensor,
    num_passes: int = 20,
    calibrated: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run MC-dropout inference and compute uncertainty metrics.
    
    Args:
        model: Model with MC-dropout enabled.
        x: Input tensor (B, C, H, W).
        num_passes: Number of stochastic forward passes.
        calibrated: Whether to use temperature-calibrated predictions.
        
    Returns:
        Tuple of:
            - Mean probabilities (B, num_classes)
            - Predicted classes (B,)
            - Dictionary of uncertainty metrics
    """
    model.train()  # Enable dropout
    
    mc_probs = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            if calibrated:
                logits = model.forward_with_temperature(x)
            else:
                logits = model(x)
            probs = F.softmax(logits, dim=1)
            mc_probs.append(probs)
    
    # Stack MC samples: (B, T, C)
    mc_probs = torch.stack(mc_probs, dim=1)
    
    # Compute mean prediction
    mean_probs = mc_probs.mean(dim=1)
    predicted = mean_probs.argmax(dim=1)
    
    # Compute uncertainty metrics
    uncertainties = {
        "entropy": predictive_entropy(mean_probs),
        "mutual_information": mutual_information(mc_probs),
        "variance": prediction_variance(mc_probs),
        "max_prob": mean_probs.max(dim=1).values,
        "mc_probs": mc_probs  # Store for later analysis
    }
    
    return mean_probs, predicted, uncertainties


def batch_mc_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_passes: int = 20,
    calibrated: bool = True,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Run MC-dropout inference on entire dataset.
    
    Args:
        model: Model with MC-dropout.
        dataloader: DataLoader for the dataset.
        num_passes: Number of MC passes.
        calibrated: Whether to use calibrated predictions.
        device: Device to run on.
        
    Returns:
        Dictionary with predictions and uncertainties.
    """
    model = model.to(device)
    
    all_probs = []
    all_labels = []
    all_predictions = []
    all_entropy = []
    all_mi = []
    all_variance = []
    all_sample_ids = []
    
    for batch in tqdm(dataloader, desc="MC-dropout inference"):
        images = batch["image"].to(device)
        labels = batch["label"]
        sample_ids = batch["sample_id"]
        
        probs, preds, uncertainties = mc_dropout_inference(
            model, images, num_passes, calibrated
        )
        
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_predictions.append(preds.cpu().numpy())
        all_entropy.append(uncertainties["entropy"].cpu().numpy())
        all_mi.append(uncertainties["mutual_information"].cpu().numpy())
        all_variance.append(uncertainties["variance"].cpu().numpy())
        all_sample_ids.extend(sample_ids)
    
    return {
        "probs": np.concatenate(all_probs),
        "labels": np.concatenate(all_labels),
        "predictions": np.concatenate(all_predictions),
        "entropy": np.concatenate(all_entropy),
        "mutual_information": np.concatenate(all_mi),
        "variance": np.concatenate(all_variance),
        "sample_ids": all_sample_ids
    }


def ensemble_uncertainty(
    ensemble_probs: Dict[str, torch.Tensor],
    ensemble_uncertainties: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute ensemble-level uncertainty metrics.
    
    Args:
        ensemble_probs: Dict mapping model names to probabilities (B, C).
        ensemble_uncertainties: Dict mapping model names to uncertainty dicts.
        
    Returns:
        Dictionary of ensemble uncertainty metrics.
    """
    probs_list = list(ensemble_probs.values())
    
    # Stack all model predictions
    stacked = torch.stack(probs_list, dim=1)  # (B, M, C)
    
    # Ensemble disagreement (variance of predictions)
    mean_probs = stacked.mean(dim=1)
    disagreement = stacked.var(dim=1).mean(dim=1)
    
    # Per-model entropy average
    entropies = torch.stack([
        ensemble_uncertainties[name]["entropy"] 
        for name in ensemble_probs.keys()
    ], dim=1)
    mean_entropy = entropies.mean(dim=1)
    
    return {
        "ensemble_entropy": predictive_entropy(mean_probs),
        "disagreement": disagreement,
        "mean_individual_entropy": mean_entropy,
        "ensemble_mi": mutual_information(stacked)
    }


class UncertaintyEstimator:
    """
    High-level uncertainty estimation for a single model or ensemble.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_passes: int = 20,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.num_passes = num_passes
        self.device = device
    
    def estimate(
        self,
        x: torch.Tensor,
        calibrated: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty for input batch.
        
        Args:
            x: Input tensor (B, C, H, W).
            calibrated: Whether to use calibrated model.
            
        Returns:
            Dictionary with predictions and uncertainties.
        """
        x = x.to(self.device)
        probs, preds, uncertainties = mc_dropout_inference(
            self.model, x, self.num_passes, calibrated
        )
        
        return {
            "probs": probs,
            "predictions": preds,
            "entropy": uncertainties["entropy"],
            "mutual_information": uncertainties["mutual_information"],
            "variance": uncertainties["variance"],
            "max_prob": uncertainties["max_prob"]
        }
    
    def should_abstain(
        self,
        uncertainties: Dict[str, torch.Tensor],
        max_prob_threshold: float = 0.6,
        entropy_threshold: float = 1.2
    ) -> torch.Tensor:
        """
        Determine whether to abstain from prediction.
        
        Args:
            uncertainties: Dictionary with uncertainty metrics.
            max_prob_threshold: Abstain if max_prob below this.
            entropy_threshold: Abstain if entropy above this.
            
        Returns:
            Boolean tensor indicating abstention.
        """
        low_confidence = uncertainties["max_prob"] < max_prob_threshold
        high_entropy = uncertainties["entropy"] > entropy_threshold
        
        return low_confidence | high_entropy


if __name__ == "__main__":
    # Test uncertainty estimation
    from .backbones import get_backbone
    
    model = get_backbone("densenet121", num_classes=4)
    
    # Test input
    x = torch.randn(4, 3, 224, 224)
    
    probs, preds, uncertainties = mc_dropout_inference(model, x, num_passes=10)
    
    print("MC-Dropout Inference Test:")
    print(f"  Mean probs shape: {probs.shape}")
    print(f"  Predictions: {preds}")
    print(f"  Entropy: {uncertainties['entropy']}")
    print(f"  Variance: {uncertainties['variance']}")
