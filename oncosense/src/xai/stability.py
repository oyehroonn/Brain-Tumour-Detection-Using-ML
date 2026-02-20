"""
Saliency Stability Scoring (SSG) for OncoSense.
Evaluates the stability of Grad-CAM explanations under small perturbations.
"""

from typing import Dict, List, Tuple, Optional
import yaml

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

from .gradcam import GradCAMGenerator, compute_heatmap_iou


def load_config(config_path: str = "configs/rag.yaml") -> dict:
    """Load RAG/stability configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PerturbationGenerator:
    """Generates small perturbations for stability testing."""
    
    @staticmethod
    def brightness_shift(
        image: torch.Tensor,
        shift_range: Tuple[float, float] = (-0.1, 0.1)
    ) -> torch.Tensor:
        """Apply random brightness shift."""
        shift = np.random.uniform(shift_range[0], shift_range[1])
        return torch.clamp(image + shift, 0, 1)
    
    @staticmethod
    def gaussian_noise(
        image: torch.Tensor,
        std: float = 0.02
    ) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)
    
    @staticmethod
    def crop_jitter(
        image: torch.Tensor,
        pixels: int = 5
    ) -> torch.Tensor:
        """Apply small random crop and resize back."""
        _, _, h, w = image.shape
        
        # Random crop offsets
        top = np.random.randint(0, pixels + 1)
        left = np.random.randint(0, pixels + 1)
        bottom = np.random.randint(0, pixels + 1)
        right = np.random.randint(0, pixels + 1)
        
        # Crop
        cropped = image[:, :, top:h-bottom, left:w-right]
        
        # Resize back to original size
        resized = torch.nn.functional.interpolate(
            cropped, size=(h, w), mode="bilinear", align_corners=False
        )
        
        return resized
    
    @staticmethod
    def contrast_shift(
        image: torch.Tensor,
        factor_range: Tuple[float, float] = (0.9, 1.1)
    ) -> torch.Tensor:
        """Apply random contrast adjustment."""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        mean = image.mean()
        return torch.clamp((image - mean) * factor + mean, 0, 1)


def generate_perturbations(
    image: torch.Tensor,
    num_perturbations: int = 5,
    config: Optional[Dict] = None
) -> List[torch.Tensor]:
    """
    Generate multiple perturbations of an input image.
    
    Args:
        image: Input tensor (1, C, H, W).
        num_perturbations: Number of perturbations to generate.
        config: Optional perturbation configuration.
        
    Returns:
        List of perturbed tensors.
    """
    perturbations = []
    
    if config is None:
        config = {
            "brightness_range": (-0.1, 0.1),
            "noise_std": 0.02,
            "crop_pixels": 5
        }
    
    for i in range(num_perturbations):
        # Cycle through perturbation types
        perturbation_type = i % 4
        
        if perturbation_type == 0:
            perturbed = PerturbationGenerator.brightness_shift(
                image, config.get("brightness_range", (-0.1, 0.1))
            )
        elif perturbation_type == 1:
            perturbed = PerturbationGenerator.gaussian_noise(
                image, config.get("noise_std", 0.02)
            )
        elif perturbation_type == 2:
            perturbed = PerturbationGenerator.crop_jitter(
                image, config.get("crop_pixels", 5)
            )
        else:
            perturbed = PerturbationGenerator.contrast_shift(
                image, config.get("contrast_range", (0.9, 1.1))
            )
        
        perturbations.append(perturbed)
    
    return perturbations


def compute_ssim(
    heatmap1: np.ndarray,
    heatmap2: np.ndarray
) -> float:
    """
    Compute SSIM between two heatmaps.
    
    Args:
        heatmap1: First heatmap (H, W).
        heatmap2: Second heatmap (H, W).
        
    Returns:
        SSIM value in [-1, 1], higher is more similar.
    """
    # Ensure same size
    if heatmap1.shape != heatmap2.shape:
        heatmap2 = cv2.resize(heatmap2, (heatmap1.shape[1], heatmap1.shape[0]))
    
    # Compute SSIM
    score = ssim(heatmap1, heatmap2, data_range=1.0)
    
    return score


def compute_correlation(
    heatmap1: np.ndarray,
    heatmap2: np.ndarray
) -> float:
    """
    Compute Pearson correlation between two heatmaps.
    
    Args:
        heatmap1: First heatmap.
        heatmap2: Second heatmap.
        
    Returns:
        Correlation coefficient in [-1, 1].
    """
    # Flatten
    h1_flat = heatmap1.flatten()
    h2_flat = heatmap2.flatten()
    
    # Compute correlation
    correlation = np.corrcoef(h1_flat, h2_flat)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0


class SaliencyStabilityScorer:
    """
    Computes stability scores for Grad-CAM explanations.
    
    Tests whether the explanation is stable under small input perturbations.
    Unstable explanations should not be trusted.
    """
    
    def __init__(
        self,
        gradcam_generator: GradCAMGenerator,
        num_perturbations: int = 5,
        similarity_metric: str = "ssim",
        perturbation_config: Optional[Dict] = None
    ):
        """
        Initialize stability scorer.
        
        Args:
            gradcam_generator: GradCAM generator instance.
            num_perturbations: Number of perturbations to test.
            similarity_metric: Metric to use ("ssim", "iou", "correlation").
            perturbation_config: Configuration for perturbations.
        """
        self.gradcam_generator = gradcam_generator
        self.num_perturbations = num_perturbations
        self.similarity_metric = similarity_metric
        self.perturbation_config = perturbation_config or {}
    
    def compute_stability_score(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict:
        """
        Compute stability score for a single input.
        
        Args:
            input_tensor: Input tensor (1, C, H, W).
            target_class: Optional target class for Grad-CAM.
            
        Returns:
            Dictionary with stability metrics.
        """
        # Generate original heatmap
        original_heatmap = self.gradcam_generator.generate(
            input_tensor, target_class
        )
        
        # Generate perturbations
        perturbations = generate_perturbations(
            input_tensor,
            self.num_perturbations,
            self.perturbation_config
        )
        
        # Generate heatmaps for perturbations
        perturbed_heatmaps = [
            self.gradcam_generator.generate(p, target_class)
            for p in perturbations
        ]
        
        # Compute similarities
        similarities = []
        for p_heatmap in perturbed_heatmaps:
            if self.similarity_metric == "ssim":
                sim = compute_ssim(original_heatmap, p_heatmap)
            elif self.similarity_metric == "iou":
                sim = compute_heatmap_iou(original_heatmap, p_heatmap)
            elif self.similarity_metric == "correlation":
                sim = compute_correlation(original_heatmap, p_heatmap)
            else:
                raise ValueError(f"Unknown metric: {self.similarity_metric}")
            similarities.append(sim)
        
        # Compute stability score (mean similarity)
        stability_score = np.mean(similarities)
        
        return {
            "stability_score": stability_score,
            "similarities": similarities,
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "std_similarity": np.std(similarities),
            "original_heatmap": original_heatmap,
            "perturbed_heatmaps": perturbed_heatmaps
        }
    
    def is_stable(
        self,
        input_tensor: torch.Tensor,
        threshold: float = 0.7,
        target_class: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Check if explanation is stable enough.
        
        Args:
            input_tensor: Input tensor.
            threshold: Stability threshold.
            target_class: Optional target class.
            
        Returns:
            Tuple of (is_stable, stability_score).
        """
        result = self.compute_stability_score(input_tensor, target_class)
        return result["stability_score"] >= threshold, result["stability_score"]


def compute_stability_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    backbone_name: str,
    num_perturbations: int = 5,
    similarity_metric: str = "ssim",
    device: str = "cuda"
) -> Dict:
    """
    Convenience function to compute stability score.
    
    Args:
        model: Classification model.
        input_tensor: Input tensor.
        backbone_name: Backbone architecture name.
        num_perturbations: Number of perturbations.
        similarity_metric: Similarity metric to use.
        device: Device to run on.
        
    Returns:
        Dictionary with stability metrics.
    """
    gradcam_gen = GradCAMGenerator(model, backbone_name, device)
    scorer = SaliencyStabilityScorer(
        gradcam_gen,
        num_perturbations,
        similarity_metric
    )
    
    return scorer.compute_stability_score(input_tensor)


class ExplanationGate:
    """
    Gating mechanism for explanations.
    
    Only allows explanations to be shown/used when they meet stability criteria.
    """
    
    def __init__(
        self,
        stability_threshold: float = 0.7,
        config_path: str = "configs/rag.yaml"
    ):
        """
        Initialize explanation gate.
        
        Args:
            stability_threshold: Minimum stability score to allow.
            config_path: Path to configuration file.
        """
        try:
            config = load_config(config_path)
            saliency_config = config.get("saliency_stability", {})
            self.stability_threshold = saliency_config.get(
                "min_stability_score", stability_threshold
            )
            self.num_perturbations = saliency_config.get("num_perturbations", 5)
        except FileNotFoundError:
            self.stability_threshold = stability_threshold
            self.num_perturbations = 5
    
    def should_show_explanation(
        self,
        stability_result: Dict
    ) -> Tuple[bool, str]:
        """
        Determine whether to show explanation.
        
        Args:
            stability_result: Result from compute_stability_score.
            
        Returns:
            Tuple of (should_show, reason).
        """
        score = stability_result["stability_score"]
        
        if score >= self.stability_threshold:
            return True, "stable"
        else:
            return False, f"unstable (score={score:.3f} < threshold={self.stability_threshold})"
    
    def gate(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        backbone_name: str,
        device: str = "cuda"
    ) -> Dict:
        """
        Full gating pipeline - compute stability and make decision.
        
        Args:
            model: Classification model.
            input_tensor: Input tensor.
            backbone_name: Backbone name.
            device: Device.
            
        Returns:
            Dictionary with gating decision and explanation (if allowed).
        """
        # Compute stability
        stability_result = compute_stability_score(
            model, input_tensor, backbone_name,
            self.num_perturbations, "ssim", device
        )
        
        # Make decision
        should_show, reason = self.should_show_explanation(stability_result)
        
        result = {
            "show_explanation": should_show,
            "reason": reason,
            "stability_score": stability_result["stability_score"]
        }
        
        if should_show:
            result["heatmap"] = stability_result["original_heatmap"]
        else:
            result["heatmap"] = None
            result["message"] = "Explanation suppressed due to instability"
        
        return result


if __name__ == "__main__":
    # Test stability scoring
    from ..models.backbones import get_backbone
    from .gradcam import GradCAMGenerator
    
    model = get_backbone("densenet121", num_classes=4, pretrained=True)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Create stability scorer
    gradcam_gen = GradCAMGenerator(model, "densenet121", device="cpu", use_cuda=False)
    scorer = SaliencyStabilityScorer(gradcam_gen, num_perturbations=3)
    
    # Compute stability
    result = scorer.compute_stability_score(x)
    
    print(f"Stability Score: {result['stability_score']:.4f}")
    print(f"Similarities: {result['similarities']}")
    print(f"Min: {result['min_similarity']:.4f}, Max: {result['max_similarity']:.4f}")
