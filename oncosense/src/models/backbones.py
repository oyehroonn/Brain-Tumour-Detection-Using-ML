"""
Backbone models for OncoSense.
Provides DenseNet121, Xception, and EfficientNet-B0 with MC-dropout support.
"""

from typing import Optional, Dict, Any, List
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models


def load_config(config_path: str = "configs/models.yaml") -> dict:
    """Load model configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class MCDropout(nn.Module):
    """Dropout that stays active during inference for MC-dropout."""
    
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class BrainTumorClassifier(nn.Module):
    """
    Unified classifier wrapper for different backbones.
    
    Supports MC-dropout for uncertainty estimation.
    """
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        mc_dropout: bool = True
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.mc_dropout = mc_dropout
        
        # Create backbone
        if backbone_name == "densenet121":
            self.backbone = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            )
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == "xception":
            self.backbone = timm.create_model(
                "xception", 
                pretrained=pretrained,
                num_classes=0  # Remove classifier
            )
            num_features = self.backbone.num_features
            
        elif backbone_name == "efficientnet_b0":
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=0
            )
            num_features = self.backbone.num_features
            
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Custom classifier head with dropout
        if mc_dropout:
            self.dropout = MCDropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Temperature for calibration (learned separately)
        self.register_buffer("temperature", torch.ones(1))
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W).
            return_features: If True, also return features before classifier.
            
        Returns:
            Logits tensor (B, num_classes), optionally features.
        """
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def forward_with_temperature(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling applied."""
        logits = self.forward(x)
        return logits / self.temperature
    
    def predict_proba(self, x: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor.
            calibrated: Whether to apply temperature scaling.
            
        Returns:
            Probability tensor (B, num_classes).
        """
        if calibrated:
            logits = self.forward_with_temperature(x)
        else:
            logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def set_temperature(self, temperature: float):
        """Set the calibration temperature."""
        self.temperature.fill_(temperature)
    
    def get_gradcam_target_layer(self) -> nn.Module:
        """Get the target layer for Grad-CAM visualization."""
        if self.backbone_name == "densenet121":
            return self.backbone.features.denseblock4.denselayer16.conv2
        elif self.backbone_name == "xception":
            return self.backbone.block12.rep[-1]
        elif self.backbone_name == "efficientnet_b0":
            return self.backbone.conv_head
        else:
            raise ValueError(f"No Grad-CAM layer defined for {self.backbone_name}")


def get_backbone(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    mc_dropout: bool = True,
    config_path: Optional[str] = None
) -> BrainTumorClassifier:
    """
    Get a backbone model by name.
    
    Args:
        name: Backbone name (densenet121, xception, efficientnet_b0).
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        dropout_rate: Dropout rate for MC-dropout.
        mc_dropout: Whether to use MC-dropout (always on during inference).
        config_path: Optional path to model config.
        
    Returns:
        BrainTumorClassifier model.
    """
    if config_path:
        config = load_config(config_path)
        backbone_config = config["backbones"].get(name, {})
        pretrained = backbone_config.get("pretrained", pretrained)
        dropout_rate = backbone_config.get("dropout_rate", dropout_rate)
        num_classes = backbone_config.get("num_classes", num_classes)
    
    return BrainTumorClassifier(
        backbone_name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        mc_dropout=mc_dropout
    )


def load_checkpoint(
    model: BrainTumorClassifier,
    checkpoint_path: str,
    device: str = "cuda"
) -> BrainTumorClassifier:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        device: Device to load to.
        
    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Load temperature if available
    if "temperature" in checkpoint:
        model.set_temperature(checkpoint["temperature"])
    
    return model


def get_ensemble(
    model_names: List[str] = ["densenet121", "xception", "efficientnet_b0"],
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda",
    config_path: str = "configs/models.yaml"
) -> Dict[str, BrainTumorClassifier]:
    """
    Load an ensemble of models from checkpoints.
    
    Args:
        model_names: List of model names to load.
        checkpoint_dir: Directory containing checkpoints.
        device: Device to load models to.
        config_path: Path to model config.
        
    Returns:
        Dictionary mapping model names to loaded models.
    """
    config = load_config(config_path)
    ensemble = {}
    
    for name in model_names:
        model = get_backbone(name, config_path=config_path)
        
        checkpoint_path = config["backbones"][name].get(
            "checkpoint_path",
            f"{checkpoint_dir}/{name}_best.pt"
        )
        
        try:
            model = load_checkpoint(model, checkpoint_path, device)
            model = model.to(device)
            model.eval()
            ensemble[name] = model
            print(f"Loaded {name} from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint not found for {name}: {checkpoint_path}")
    
    return ensemble


if __name__ == "__main__":
    # Test model creation
    for name in ["densenet121", "xception", "efficientnet_b0"]:
        model = get_backbone(name, num_classes=4, pretrained=True)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        
        print(f"{name}:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
