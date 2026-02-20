"""
Data transforms and augmentation for OncoSense.
Provides training and inference transforms using albumentations.
"""

from typing import Dict, Any, Optional, Tuple
import yaml

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_train_transforms(config_path: str = "configs/data.yaml") -> A.Compose:
    """
    Get training data transforms with augmentation.
    
    Args:
        config_path: Path to data configuration file.
        
    Returns:
        Albumentations Compose transform.
    """
    config = load_config(config_path)
    prep = config["preprocessing"]
    aug = config["augmentation"]["train"]
    
    return A.Compose([
        # Resize to target size
        A.Resize(prep["input_size"], prep["input_size"]),
        
        # Augmentation
        A.HorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
        A.VerticalFlip(p=aug.get("vertical_flip", 0.0)),
        A.Rotate(limit=aug.get("rotation_limit", 15), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug.get("brightness_limit", 0.2),
            contrast_limit=aug.get("contrast_limit", 0.2),
            p=0.5
        ),
        A.GaussNoise(
            var_limit=aug.get("gaussian_noise_var", (0.01, 0.05)),
            p=0.3
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.3
        ),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=prep["normalize_mean"],
            std=prep["normalize_std"]
        ),
        ToTensorV2()
    ])


def get_val_transforms(config_path: str = "configs/data.yaml") -> A.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        config_path: Path to data configuration file.
        
    Returns:
        Albumentations Compose transform.
    """
    config = load_config(config_path)
    prep = config["preprocessing"]
    
    return A.Compose([
        A.Resize(prep["input_size"], prep["input_size"]),
        A.Normalize(
            mean=prep["normalize_mean"],
            std=prep["normalize_std"]
        ),
        ToTensorV2()
    ])


def get_inference_transforms(
    input_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Get inference transforms with explicit parameters.
    
    Args:
        input_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize a tensor image back to displayable format.
    
    Args:
        tensor: Normalized image tensor (C, H, W).
        mean: Normalization mean used.
        std: Normalization std used.
        
    Returns:
        Numpy array in (H, W, C) format, values in [0, 255].
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # If batch dimension exists, take first
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # Denormalize
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    
    img = tensor * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Convert to HWC
    img = np.transpose(img, (1, 2, 0))
    
    return img


def load_image(
    filepath: str,
    convert_rgb: bool = True
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        filepath: Path to image file.
        convert_rgb: Whether to convert grayscale to RGB.
        
    Returns:
        Numpy array in (H, W, C) format.
    """
    img = Image.open(filepath)
    
    if convert_rgb and img.mode != "RGB":
        img = img.convert("RGB")
    
    return np.array(img)


class BrainTumorDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for brain tumor MRI images.
    
    Args:
        manifest_df: DataFrame with image metadata.
        transform: Albumentations transform to apply.
        data_root: Root directory for image files.
    """
    
    def __init__(
        self,
        manifest_df,
        transform: Optional[A.Compose] = None,
        data_root: str = "data"
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.data_root = data_root
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Load image
        img_path = f"{self.data_root}/{row['filepath']}"
        if not img_path.startswith(self.data_root):
            img_path = row['filepath']
        
        image = load_image(img_path, convert_rgb=True)
        label = row["label"]
        
        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return {
            "image": image,
            "label": label,
            "sample_id": row["sample_id"],
            "filepath": row["filepath"],
            "label_name": row["label_name"]
        }


def create_dataloaders(
    manifest_path: str = "data/manifests/manifest.parquet",
    config_path: str = "configs/data.yaml",
    train_config_path: str = "configs/train.yaml",
    data_root: str = "data"
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        manifest_path: Path to manifest parquet file.
        config_path: Path to data config.
        train_config_path: Path to training config.
        data_root: Root directory for data.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    import pandas as pd
    
    # Load configs
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)
    
    # Load manifest
    df = pd.read_parquet(manifest_path)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    # Create transforms
    train_transform = get_train_transforms(config_path)
    val_transform = get_val_transforms(config_path)
    
    # Create datasets
    train_dataset = BrainTumorDataset(train_df, train_transform, data_root)
    val_dataset = BrainTumorDataset(val_df, val_transform, data_root)
    test_dataset = BrainTumorDataset(test_df, val_transform, data_root)
    
    # Create dataloaders
    batch_size = train_config["training"]["batch_size"]
    num_workers = train_config["training"]["num_workers"]
    pin_memory = train_config["training"]["pin_memory"]
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
