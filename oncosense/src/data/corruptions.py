"""
Corruption suite for robustness evaluation (LoRRA-style).
Implements various image corruptions at different severity levels.
"""

from typing import Callable, Dict, List, Tuple, Optional
import yaml

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Corruption implementations

def gaussian_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Add Gaussian noise to image."""
    c = [0.04, 0.08, 0.12, 0.16, 0.20][severity - 1]
    image = np.array(image) / 255.0
    noise = np.random.normal(0, c, image.shape)
    noisy = np.clip(image + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def shot_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Add shot (Poisson) noise to image."""
    c = [500, 250, 100, 50, 25][severity - 1]
    image = np.array(image) / 255.0
    noisy = np.clip(np.random.poisson(image * c) / c, 0, 1)
    return (noisy * 255).astype(np.uint8)


def impulse_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Add salt and pepper noise."""
    c = [0.01, 0.02, 0.05, 0.08, 0.12][severity - 1]
    image = np.array(image).copy()
    
    # Salt
    salt = np.random.random(image.shape[:2]) < c / 2
    image[salt] = 255
    
    # Pepper
    pepper = np.random.random(image.shape[:2]) < c / 2
    image[pepper] = 0
    
    return image


def speckle_noise(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Add speckle (multiplicative) noise."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    image = np.array(image) / 255.0
    noise = np.random.randn(*image.shape) * c
    noisy = np.clip(image + image * noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Apply Gaussian blur."""
    c = [0.5, 1.0, 1.5, 2.0, 2.5][severity - 1]
    image = np.array(image)
    blurred = gaussian_filter(image, sigma=c)
    return blurred.astype(np.uint8)


def motion_blur(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Apply motion blur."""
    c = [3, 5, 7, 9, 11][severity - 1]
    
    # Create motion blur kernel
    kernel = np.zeros((c, c))
    kernel[c // 2, :] = np.ones(c) / c
    
    image = np.array(image)
    if len(image.shape) == 3:
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred


def brightness(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Adjust brightness (increase or decrease)."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    
    image = np.array(image) / 255.0
    
    # Randomly increase or decrease
    if np.random.random() > 0.5:
        bright = np.clip(image + c, 0, 1)
    else:
        bright = np.clip(image - c, 0, 1)
    
    return (bright * 255).astype(np.uint8)


def contrast(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Adjust contrast."""
    c = [0.75, 0.6, 0.45, 0.3, 0.15][severity - 1]
    
    image = np.array(image) / 255.0
    mean = np.mean(image)
    adjusted = np.clip((image - mean) * c + mean, 0, 1)
    
    return (adjusted * 255).astype(np.uint8)


def jpeg_compression(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Apply JPEG compression artifacts."""
    c = [80, 65, 50, 35, 20][severity - 1]
    
    # Encode and decode with JPEG
    pil_img = Image.fromarray(image)
    from io import BytesIO
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=c)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer))
    
    return compressed


def pixelate(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Pixelate image (downsample then upsample)."""
    c = [0.9, 0.8, 0.6, 0.4, 0.25][severity - 1]
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * c), int(w * c)
    
    pil_img = Image.fromarray(image)
    small = pil_img.resize((new_w, new_h), Image.NEAREST)
    pixelated = small.resize((w, h), Image.NEAREST)
    
    return np.array(pixelated)


def downsample(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """Downsample and upsample (simulates low resolution)."""
    c = [0.8, 0.6, 0.4, 0.3, 0.2][severity - 1]
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * c), int(w * c)
    
    pil_img = Image.fromarray(image)
    small = pil_img.resize((new_w, new_h), Image.BILINEAR)
    upsampled = small.resize((w, h), Image.BILINEAR)
    
    return np.array(upsampled)


def bias_field(image: np.ndarray, severity: int = 1) -> np.ndarray:
    """
    Simulate MRI bias field artifact.
    Creates a smooth multiplicative intensity variation across the image.
    """
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    
    h, w = image.shape[:2]
    image = np.array(image) / 255.0
    
    # Create smooth bias field using polynomial surface
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    
    # Random polynomial coefficients
    bias = 1 + c * (
        np.random.randn() * X + 
        np.random.randn() * Y + 
        np.random.randn() * X * Y +
        np.random.randn() * X**2 +
        np.random.randn() * Y**2
    )
    
    # Normalize bias field
    bias = bias / bias.mean()
    
    # Apply bias field
    if len(image.shape) == 3:
        bias = bias[:, :, np.newaxis]
    
    biased = np.clip(image * bias, 0, 1)
    
    return (biased * 255).astype(np.uint8)


# Corruption registry
CORRUPTIONS: Dict[str, Callable] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "speckle_noise": speckle_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "contrast": contrast,
    "jpeg_compression": jpeg_compression,
    "pixelate": pixelate,
    "downsample": downsample,
    "bias_field": bias_field,
}


def apply_corruption(
    image: np.ndarray,
    corruption_name: str,
    severity: int = 1
) -> np.ndarray:
    """
    Apply a corruption to an image.
    
    Args:
        image: Input image as numpy array.
        corruption_name: Name of the corruption to apply.
        severity: Severity level (1-5).
        
    Returns:
        Corrupted image.
    """
    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be 1-5, got {severity}")
    
    return CORRUPTIONS[corruption_name](image, severity)


def get_all_corruptions(
    image: np.ndarray,
    severity: int = 3
) -> Dict[str, np.ndarray]:
    """
    Apply all corruptions to an image.
    
    Args:
        image: Input image.
        severity: Severity level for all corruptions.
        
    Returns:
        Dictionary mapping corruption names to corrupted images.
    """
    results = {}
    for name, func in CORRUPTIONS.items():
        try:
            results[name] = func(image.copy(), severity)
        except Exception as e:
            print(f"Warning: Failed to apply {name}: {e}")
    return results


class CorruptedDataset:
    """
    Wrapper that applies corruptions to an existing dataset.
    
    Args:
        base_dataset: Base PyTorch dataset.
        corruption_name: Name of corruption to apply.
        severity: Severity level (1-5).
    """
    
    def __init__(
        self,
        base_dataset,
        corruption_name: str,
        severity: int = 3
    ):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Get image (may be tensor)
        image = item["image"]
        
        # Convert tensor to numpy if needed
        if hasattr(image, "numpy"):
            # Denormalize first
            from .transforms import denormalize_image
            image = denormalize_image(image)
        
        # Apply corruption
        corrupted = apply_corruption(image, self.corruption_name, self.severity)
        
        # Re-apply the base dataset's transform if it exists
        if hasattr(self.base_dataset, "transform") and self.base_dataset.transform:
            transformed = self.base_dataset.transform(image=corrupted)
            corrupted = transformed["image"]
        
        item["image"] = corrupted
        item["corruption"] = self.corruption_name
        item["severity"] = self.severity
        
        return item


def evaluate_robustness(
    model,
    test_dataset,
    corruptions: Optional[List[str]] = None,
    severities: List[int] = [1, 3, 5],
    device: str = "cuda"
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate model robustness across corruptions and severities.
    
    Args:
        model: PyTorch model.
        test_dataset: Base test dataset.
        corruptions: List of corruption names (default: all).
        severities: List of severity levels to test.
        device: Device to run inference on.
        
    Returns:
        Dictionary mapping corruption names to severity-accuracy dicts.
    """
    import torch
    from tqdm import tqdm
    
    if corruptions is None:
        corruptions = list(CORRUPTIONS.keys())
    
    model.eval()
    results = {}
    
    for corruption_name in corruptions:
        results[corruption_name] = {}
        
        for severity in severities:
            corrupted_dataset = CorruptedDataset(
                test_dataset, corruption_name, severity
            )
            
            loader = torch.utils.data.DataLoader(
                corrupted_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4
            )
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(
                    loader, 
                    desc=f"{corruption_name} (s={severity})",
                    leave=False
                ):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            accuracy = 100 * correct / total
            results[corruption_name][severity] = accuracy
    
    return results


if __name__ == "__main__":
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(description="Test corruption functions")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--output", "-o", type=str, default="corrupted_samples",
                        help="Output directory for corrupted images")
    parser.add_argument("--severity", "-s", type=int, default=3,
                        help="Severity level (1-5)")
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output, exist_ok=True)
    
    # Load image
    img = np.array(Image.open(args.image).convert("RGB"))
    
    # Apply all corruptions
    for name in CORRUPTIONS:
        try:
            corrupted = apply_corruption(img, name, args.severity)
            output_path = os.path.join(args.output, f"{name}_s{args.severity}.png")
            Image.fromarray(corrupted).save(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed {name}: {e}")
