"""
Grad-CAM implementation for OncoSense.
Generates class activation maps for model interpretability.
"""

from typing import Dict, Optional, Union, Tuple, List
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def load_config(config_path: str = "configs/models.yaml") -> dict:
    """Load model configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_target_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM based on backbone architecture.
    
    Args:
        model: The model instance.
        backbone_name: Name of the backbone.
        
    Returns:
        Target layer module.
    """
    if backbone_name == "densenet121":
        return model.backbone.features.denseblock4.denselayer16.conv2
    elif backbone_name == "xception":
        # Xception from timm
        return model.backbone.block12.rep[-1]
    elif backbone_name == "efficientnet_b0":
        return model.backbone.conv_head
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


class GradCAMGenerator:
    """
    High-level Grad-CAM generator for brain tumor classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        backbone_name: str,
        device: str = "cuda",
        use_cuda: bool = True
    ):
        """
        Initialize Grad-CAM generator.
        
        Args:
            model: The classification model.
            backbone_name: Name of the backbone architecture.
            device: Device to run on.
            use_cuda: Whether to use CUDA for Grad-CAM.
        """
        self.model = model.to(device)
        self.backbone_name = backbone_name
        self.device = device
        
        # Get target layer
        target_layer = get_target_layer(model, backbone_name)
        
        # Initialize Grad-CAM
        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda and torch.cuda.is_available()
        )
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        return_prediction: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, int, float]]:
        """
        Generate Grad-CAM heatmap for input.
        
        Args:
            input_tensor: Input tensor (1, C, H, W) or (C, H, W).
            target_class: Class to generate CAM for. If None, uses predicted class.
            return_prediction: Whether to also return prediction info.
            
        Returns:
            Heatmap array (H, W) with values in [0, 1], 
            optionally with (heatmap, pred_class, confidence).
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction if target_class not specified
        if target_class is None or return_prediction:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs.max(dim=1).values.item()
            
            if target_class is None:
                target_class = pred_class
        
        # Generate CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Get the CAM for the first (and only) image in batch
        heatmap = grayscale_cam[0]
        
        if return_prediction:
            return heatmap, pred_class, confidence
        return heatmap
    
    def generate_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate Grad-CAM overlay on original image.
        
        Args:
            input_tensor: Preprocessed input tensor.
            original_image: Original image as numpy array (H, W, 3), values in [0, 255].
            target_class: Class to visualize.
            colormap: OpenCV colormap for heatmap.
            alpha: Overlay alpha blending factor.
            
        Returns:
            Tuple of (overlay_image, heatmap, info_dict).
        """
        # Normalize original image to [0, 1]
        if original_image.max() > 1:
            original_image = original_image.astype(np.float32) / 255.0
        
        # Generate heatmap
        heatmap, pred_class, confidence = self.generate(
            input_tensor, target_class, return_prediction=True
        )
        
        # Resize heatmap to match original image
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Create overlay using pytorch-grad-cam utility
        overlay = show_cam_on_image(
            original_image,
            heatmap_resized,
            use_rgb=True,
            colormap=colormap
        )
        
        info = {
            "predicted_class": pred_class,
            "confidence": confidence,
            "target_class": target_class if target_class is not None else pred_class
        }
        
        return overlay, heatmap_resized, info
    
    def batch_generate(
        self,
        input_tensors: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmaps for batch of inputs.
        
        Args:
            input_tensors: Batch input tensor (B, C, H, W).
            target_classes: List of target classes for each image.
            
        Returns:
            Array of heatmaps (B, H, W).
        """
        input_tensors = input_tensors.to(self.device)
        batch_size = input_tensors.size(0)
        
        if target_classes is None:
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensors)
                target_classes = logits.argmax(dim=1).tolist()
        
        targets = [ClassifierOutputTarget(tc) for tc in target_classes]
        grayscale_cams = self.cam(input_tensor=input_tensors, targets=targets)
        
        return grayscale_cams


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    backbone_name: str,
    target_class: Optional[int] = None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Convenience function to generate Grad-CAM heatmap.
    
    Args:
        model: Classification model.
        input_tensor: Input tensor.
        backbone_name: Name of backbone architecture.
        target_class: Optional target class.
        device: Device to run on.
        
    Returns:
        Heatmap array (H, W).
    """
    generator = GradCAMGenerator(model, backbone_name, device)
    return generator.generate(input_tensor, target_class)


def visualize_gradcam(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Grad-CAM",
    show_colorbar: bool = True
) -> Optional[np.ndarray]:
    """
    Visualize Grad-CAM heatmap with matplotlib.
    
    Args:
        heatmap: Heatmap array (H, W).
        original_image: Original image.
        save_path: Optional path to save figure.
        title: Plot title.
        show_colorbar: Whether to show colorbar.
        
    Returns:
        Figure as numpy array if save_path provided.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    if show_colorbar:
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    if original_image.max() > 1:
        original_norm = original_image.astype(np.float32) / 255.0
    else:
        original_norm = original_image
    
    heatmap_resized = cv2.resize(heatmap, (original_norm.shape[1], original_norm.shape[0]))
    overlay = show_cam_on_image(original_norm, heatmap_resized, use_rgb=True)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def compute_heatmap_iou(
    heatmap1: np.ndarray,
    heatmap2: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute IoU between two binarized heatmaps.
    
    Args:
        heatmap1: First heatmap.
        heatmap2: Second heatmap.
        threshold: Binarization threshold.
        
    Returns:
        IoU score.
    """
    # Binarize
    binary1 = (heatmap1 >= threshold).astype(float)
    binary2 = (heatmap2 >= threshold).astype(float)
    
    # Compute IoU
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


if __name__ == "__main__":
    # Test Grad-CAM generation
    from ..models.backbones import get_backbone
    
    model = get_backbone("densenet121", num_classes=4, pretrained=True)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    
    # Generate Grad-CAM
    generator = GradCAMGenerator(model, "densenet121", device="cpu", use_cuda=False)
    heatmap, pred_class, confidence = generator.generate(x, return_prediction=True)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
