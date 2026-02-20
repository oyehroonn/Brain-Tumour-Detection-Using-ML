"""
Kaggle dataset downloader for OncoSense.
Downloads the 4-class brain tumor MRI dataset and organizes it.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional
import yaml

import kagglehub
from tqdm import tqdm
from PIL import Image


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_image(filepath: Path) -> bool:
    """Check if a file is a valid image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_dataset(
    output_dir: Optional[str] = None,
    config_path: str = "configs/data.yaml"
) -> Path:
    """
    Download the brain tumor MRI dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset. If None, uses config default.
        config_path: Path to the data configuration file.
        
    Returns:
        Path to the downloaded dataset directory.
    """
    config = load_config(config_path)
    kaggle_id = config["dataset"]["kaggle_id"]
    
    if output_dir is None:
        output_dir = config["paths"]["raw_dir"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {kaggle_id}")
    print(f"Output directory: {output_path.absolute()}")
    
    # Download using kagglehub
    downloaded_path = kagglehub.dataset_download(kaggle_id)
    downloaded_path = Path(downloaded_path)
    
    print(f"Downloaded to: {downloaded_path}")
    
    # Organize into class directories
    class_names = config["dataset"]["class_names"]
    class_dirs = {v: output_path / v for v in class_names.values()}
    
    for class_dir in class_dirs.values():
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and organize images
    stats = {"total": 0, "valid": 0, "invalid": 0, "by_class": {}}
    
    for class_name in class_names.values():
        stats["by_class"][class_name] = 0
    
    # Look for images in the downloaded directory
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    
    for root, dirs, files in os.walk(downloaded_path):
        root_path = Path(root)
        
        for filename in tqdm(files, desc=f"Processing {root_path.name}", leave=False):
            if Path(filename).suffix.lower() not in image_extensions:
                continue
                
            src_path = root_path / filename
            stats["total"] += 1
            
            if not validate_image(src_path):
                stats["invalid"] += 1
                continue
            
            # Determine class from directory structure
            class_name = None
            for cname in class_names.values():
                if cname.lower() in root_path.name.lower() or cname.lower() in str(root_path).lower():
                    class_name = cname
                    break
            
            if class_name is None:
                # Try to infer from parent directory name
                parent_name = root_path.name.lower()
                # Handle common variations
                if "glioma" in parent_name:
                    class_name = "glioma"
                elif "meningioma" in parent_name:
                    class_name = "meningioma"
                elif "pituitary" in parent_name:
                    class_name = "pituitary"
                elif "no" in parent_name or "notumor" in parent_name or "healthy" in parent_name:
                    class_name = "no_tumor"
                else:
                    continue
            
            # Copy to organized directory
            file_hash = get_file_hash(src_path)[:12]
            new_filename = f"{class_name}_{file_hash}{src_path.suffix.lower()}"
            dst_path = class_dirs[class_name] / new_filename
            
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                stats["valid"] += 1
                stats["by_class"][class_name] += 1
    
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    print(f"Total files found: {stats['total']}")
    print(f"Valid images: {stats['valid']}")
    print(f"Invalid/skipped: {stats['invalid']}")
    print("\nImages per class:")
    for class_name, count in stats["by_class"].items():
        print(f"  {class_name}: {count}")
    print(f"\nDataset saved to: {output_path.absolute()}")
    
    return output_path


def verify_dataset(data_dir: str = "data/raw") -> dict:
    """
    Verify the downloaded dataset structure and count images.
    
    Args:
        data_dir: Path to the raw data directory.
        
    Returns:
        Dictionary with verification results.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return {"error": f"Directory not found: {data_path}"}
    
    results = {"classes": {}, "total": 0, "issues": []}
    
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        images = list(class_dir.glob("*"))
        valid_images = [img for img in images if validate_image(img)]
        
        results["classes"][class_name] = {
            "total": len(images),
            "valid": len(valid_images),
            "invalid": len(images) - len(valid_images)
        }
        results["total"] += len(valid_images)
        
        if len(images) - len(valid_images) > 0:
            results["issues"].append(
                f"{class_name}: {len(images) - len(valid_images)} invalid images"
            )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download brain tumor MRI dataset")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for dataset")
    parser.add_argument("--config", "-c", type=str, default="configs/data.yaml",
                        help="Path to data config file")
    parser.add_argument("--verify", "-v", action="store_true",
                        help="Verify existing dataset instead of downloading")
    
    args = parser.parse_args()
    
    if args.verify:
        results = verify_dataset(args.output or "data/raw")
        print("\nVerification Results:")
        print(yaml.dump(results, default_flow_style=False))
    else:
        download_dataset(args.output, args.config)
