"""
Inference module for OncoSense.
Handles single image and batch inference with full pipeline.
"""

from pathlib import Path
from typing import Dict, Optional, Union, List
import yaml

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .backbones import get_backbone, load_checkpoint, get_ensemble
from .uncertainty import mc_dropout_inference, UncertaintyEstimator
from .fusion import EGUFusion, fused_ensemble_inference
from ..data.transforms import get_inference_transforms, load_image


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class OncoSenseInference:
    """
    High-level inference class for OncoSense pipeline.
    
    Handles single model or ensemble inference with:
    - MC-dropout uncertainty
    - Temperature calibration
    - EGU-Fusion (for ensemble)
    - Abstention
    """
    
    CLASS_NAMES = {
        0: "glioma",
        1: "meningioma",
        2: "pituitary",
        3: "no_tumor"
    }
    
    def __init__(
        self,
        model_config_path: str = "configs/models.yaml",
        fusion_config_path: str = "configs/fusion.yaml",
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
        use_ensemble: bool = True,
        num_mc_passes: int = 20
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_config_path: Path to model config.
            fusion_config_path: Path to fusion config.
            checkpoint_dir: Directory with model checkpoints.
            device: Device to run inference on.
            use_ensemble: Whether to use ensemble (vs single model).
            num_mc_passes: Number of MC-dropout passes.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_mc_passes = num_mc_passes
        self.use_ensemble = use_ensemble
        
        # Load configs
        self.model_config = load_config(model_config_path)
        self.fusion_config = load_config(fusion_config_path)
        
        # Load models
        if use_ensemble:
            self.models = get_ensemble(
                model_names=self.model_config["active_models"],
                checkpoint_dir=checkpoint_dir,
                device=self.device,
                config_path=model_config_path
            )
            
            # Initialize fusion
            self.fusion = EGUFusion(
                alpha=self.fusion_config["egu_fusion"]["alpha"],
                uncertainty_metric=self.fusion_config["egu_fusion"]["uncertainty_metric"],
                max_prob_threshold=self.fusion_config["abstention"]["max_prob_threshold"],
                entropy_threshold=self.fusion_config["abstention"]["entropy_threshold"],
                abstention_mode=self.fusion_config["abstention"]["mode"]
            )
        else:
            # Single model mode
            model_name = self.model_config["active_models"][0]
            self.model = get_backbone(model_name, config_path=model_config_path)
            checkpoint_path = f"{checkpoint_dir}/{model_name}_best.pt"
            self.model = load_checkpoint(self.model, checkpoint_path, self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.uncertainty_estimator = UncertaintyEstimator(
                self.model, num_mc_passes, self.device
            )
        
        # Transforms
        self.transform = get_inference_transforms(
            input_size=self.model_config["input"]["size"],
            mean=tuple(self.model_config["input"]["normalize_mean"]),
            std=tuple(self.model_config["input"]["normalize_std"])
        )
    
    def preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Image path, numpy array, or PIL Image.
            
        Returns:
            Preprocessed tensor ready for model.
        """
        if isinstance(image, (str, Path)):
            image = load_image(str(image), convert_rgb=True)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        transformed = self.transform(image=image)
        tensor = transformed["image"].unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_all: bool = False
    ) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image: Input image.
            return_all: Whether to return detailed per-model outputs.
            
        Returns:
            Dictionary with prediction results.
        """
        # Preprocess
        tensor = self.preprocess(image)
        tensor = tensor.to(self.device)
        
        if self.use_ensemble:
            return self._ensemble_predict(tensor, return_all)
        else:
            return self._single_predict(tensor)
    
    def _single_predict(self, tensor: torch.Tensor) -> Dict:
        """Single model prediction with uncertainty."""
        result = self.uncertainty_estimator.estimate(tensor, calibrated=True)
        
        pred_idx = result["predictions"].item()
        
        return {
            "predicted_class": pred_idx,
            "predicted_label": self.CLASS_NAMES[pred_idx],
            "confidence": result["max_prob"].item(),
            "probabilities": result["probs"][0].cpu().numpy().tolist(),
            "uncertainty": {
                "entropy": result["entropy"].item(),
                "mutual_information": result["mutual_information"].item(),
                "variance": result["variance"].item()
            },
            "abstain": self.uncertainty_estimator.should_abstain(
                result,
                self.fusion_config["abstention"]["max_prob_threshold"],
                self.fusion_config["abstention"]["entropy_threshold"]
            ).item()
        }
    
    def _ensemble_predict(self, tensor: torch.Tensor, return_all: bool) -> Dict:
        """Ensemble prediction with EGU-Fusion."""
        result = fused_ensemble_inference(
            self.models,
            tensor,
            self.fusion,
            self.num_mc_passes,
            calibrated=True,
            device=self.device
        )
        
        pred_idx = result["predictions"].item()
        
        output = {
            "predicted_class": pred_idx,
            "predicted_label": self.CLASS_NAMES[pred_idx],
            "confidence": result["max_prob"].item(),
            "probabilities": result["fused_probs"][0].cpu().numpy().tolist(),
            "uncertainty": {
                "entropy": result["entropy"].item()
            },
            "fusion_weights": {
                name: result["weights"][0, i].item()
                for i, name in enumerate(result["model_names"])
            },
            "abstain": result["abstain"].item()
        }
        
        if return_all:
            output["per_model"] = {}
            for name in result["model_names"]:
                probs = result["per_model_probs"][name][0].cpu().numpy()
                output["per_model"][name] = {
                    "predicted_class": int(probs.argmax()),
                    "confidence": float(probs.max()),
                    "probabilities": probs.tolist(),
                    "entropy": result["per_model_uncertainties"][name]["entropy"].item()
                }
        
        return output
    
    def batch_predict(
        self,
        images: List[Union[str, Path, np.ndarray]],
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of images.
            batch_size: Batch size for inference.
            
        Returns:
            List of prediction dictionaries.
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            tensors = [self.preprocess(img) for img in batch_images]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            
            # Run inference
            if self.use_ensemble:
                batch_result = fused_ensemble_inference(
                    self.models,
                    batch_tensor,
                    self.fusion,
                    self.num_mc_passes,
                    calibrated=True,
                    device=self.device
                )
                
                for j in range(len(batch_images)):
                    pred_idx = batch_result["predictions"][j].item()
                    results.append({
                        "predicted_class": pred_idx,
                        "predicted_label": self.CLASS_NAMES[pred_idx],
                        "confidence": batch_result["max_prob"][j].item(),
                        "entropy": batch_result["entropy"][j].item(),
                        "abstain": batch_result["abstain"][j].item()
                    })
            else:
                batch_result = self.uncertainty_estimator.estimate(
                    batch_tensor, calibrated=True
                )
                
                for j in range(len(batch_images)):
                    pred_idx = batch_result["predictions"][j].item()
                    results.append({
                        "predicted_class": pred_idx,
                        "predicted_label": self.CLASS_NAMES[pred_idx],
                        "confidence": batch_result["max_prob"][j].item(),
                        "entropy": batch_result["entropy"][j].item(),
                        "abstain": self.uncertainty_estimator.should_abstain(
                            {k: v[j:j+1] for k, v in batch_result.items()},
                            self.fusion_config["abstention"]["max_prob_threshold"],
                            self.fusion_config["abstention"]["entropy_threshold"]
                        ).item()
                    })
        
        return results


def run_inference(
    image_path: str,
    model_config_path: str = "configs/models.yaml",
    fusion_config_path: str = "configs/fusion.yaml",
    checkpoint_dir: str = "checkpoints",
    use_ensemble: bool = True,
    device: str = "cuda"
) -> Dict:
    """
    Convenience function for single image inference.
    
    Args:
        image_path: Path to image.
        model_config_path: Path to model config.
        fusion_config_path: Path to fusion config.
        checkpoint_dir: Checkpoint directory.
        use_ensemble: Whether to use ensemble.
        device: Device to run on.
        
    Returns:
        Prediction dictionary.
    """
    inferencer = OncoSenseInference(
        model_config_path=model_config_path,
        fusion_config_path=fusion_config_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        use_ensemble=use_ensemble
    )
    
    return inferencer.predict(image_path, return_all=True)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run OncoSense inference")
    parser.add_argument("image", type=str, help="Path to MRI image")
    parser.add_argument("--checkpoint-dir", "-c", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--single-model", "-s", action="store_true",
                        help="Use single model instead of ensemble")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path")
    
    args = parser.parse_args()
    
    result = run_inference(
        args.image,
        checkpoint_dir=args.checkpoint_dir,
        use_ensemble=not args.single_model,
        device=args.device
    )
    
    print(json.dumps(result, indent=2))
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
