"""
Deduplication module for OncoSense.
Handles exact duplicate detection (SHA256) and near-duplicate clustering (pHash).
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import yaml


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_phash(filepath: Path, hash_size: int = 16) -> str:
    """
    Compute perceptual hash of an image.
    
    Args:
        filepath: Path to image file.
        hash_size: Size of the hash (default 16 for 256-bit hash).
        
    Returns:
        Hexadecimal string representation of the perceptual hash.
    """
    try:
        with Image.open(filepath) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            phash = imagehash.phash(img, hash_size=hash_size)
            return str(phash)
    except Exception as e:
        print(f"Error computing phash for {filepath}: {e}")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hexadecimal hash strings.
    
    Args:
        hash1: First hash string.
        hash2: Second hash string.
        
    Returns:
        Hamming distance (number of differing bits).
    """
    if hash1 is None or hash2 is None:
        return float("inf")
    
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def find_exact_duplicates(image_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Find exact duplicate images using SHA256 hash.
    
    Args:
        image_paths: List of paths to image files.
        
    Returns:
        Dictionary mapping SHA256 hash to list of duplicate file paths.
    """
    hash_to_paths = defaultdict(list)
    
    for path in tqdm(image_paths, desc="Computing SHA256 hashes"):
        file_hash = compute_sha256(path)
        hash_to_paths[file_hash].append(path)
    
    # Filter to only duplicates
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    
    return duplicates


def find_near_duplicates(
    image_paths: List[Path],
    threshold: int = 8,
    hash_size: int = 16
) -> Dict[str, List[Path]]:
    """
    Find near-duplicate images using perceptual hashing.
    
    Args:
        image_paths: List of paths to image files.
        threshold: Maximum Hamming distance for near-duplicates.
        hash_size: Size of the perceptual hash.
        
    Returns:
        Dictionary mapping representative hash to list of near-duplicate paths.
    """
    # Compute all perceptual hashes
    path_to_phash = {}
    for path in tqdm(image_paths, desc="Computing perceptual hashes"):
        phash = compute_phash(path, hash_size)
        if phash is not None:
            path_to_phash[path] = phash
    
    # Find near-duplicates by comparing hashes
    near_duplicates = defaultdict(list)
    processed = set()
    
    paths = list(path_to_phash.keys())
    hashes = [path_to_phash[p] for p in paths]
    
    for i, (path1, hash1) in enumerate(tqdm(
        zip(paths, hashes), 
        total=len(paths), 
        desc="Finding near-duplicates"
    )):
        if path1 in processed:
            continue
            
        group = [path1]
        processed.add(path1)
        
        for j in range(i + 1, len(paths)):
            path2, hash2 = paths[j], hashes[j]
            if path2 in processed:
                continue
                
            dist = hamming_distance(hash1, hash2)
            if dist <= threshold:
                group.append(path2)
                processed.add(path2)
        
        if len(group) > 1:
            near_duplicates[hash1] = group
    
    return near_duplicates


def cluster_by_phash(
    df: pd.DataFrame,
    phash_col: str = "phash",
    distance_threshold: int = 10
) -> pd.DataFrame:
    """
    Cluster images by perceptual hash similarity using agglomerative clustering.
    
    Args:
        df: DataFrame with image metadata including phash column.
        phash_col: Name of the column containing perceptual hashes.
        distance_threshold: Distance threshold for clustering.
        
    Returns:
        DataFrame with added cluster_id column.
    """
    df = df.copy()
    
    # Filter out rows with invalid hashes
    valid_mask = df[phash_col].notna()
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        df["cluster_id"] = -1
        return df
    
    # Convert hashes to binary arrays for distance computation
    hashes = valid_df[phash_col].values
    n_samples = len(hashes)
    
    print(f"Computing distance matrix for {n_samples} images...")
    
    # Compute pairwise distance matrix
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in tqdm(range(n_samples), desc="Computing distances"):
        for j in range(i + 1, n_samples):
            dist = hamming_distance(hashes[i], hashes[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # Apply agglomerative clustering
    print("Clustering images...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average"
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Assign cluster IDs
    valid_df["cluster_id"] = cluster_labels
    
    # Merge back
    df.loc[valid_mask, "cluster_id"] = valid_df["cluster_id"]
    df.loc[~valid_mask, "cluster_id"] = -1
    
    return df


def create_manifest(
    data_dir: str,
    config_path: str = "configs/data.yaml",
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a manifest DataFrame with image metadata and hashes.
    
    Args:
        data_dir: Path to the raw data directory.
        config_path: Path to data configuration file.
        output_path: Optional path to save the manifest.
        
    Returns:
        DataFrame with image manifest.
    """
    config = load_config(config_path)
    class_names = config["dataset"]["class_names"]
    
    data_path = Path(data_dir)
    records = []
    
    # Collect all images
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    
    for class_idx, class_name in class_names.items():
        class_dir = data_path / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
            
        for img_path in tqdm(
            list(class_dir.glob("*")),
            desc=f"Processing {class_name}"
        ):
            if img_path.suffix.lower() not in image_extensions:
                continue
                
            try:
                sha256 = compute_sha256(img_path)
                phash = compute_phash(img_path)
                
                # Get image metadata
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                
                records.append({
                    "sample_id": sha256,
                    "filepath": str(img_path.relative_to(data_path.parent)),
                    "filename": img_path.name,
                    "label": int(class_idx),
                    "label_name": class_name,
                    "source": "kaggle_7023",
                    "phash": phash,
                    "width": width,
                    "height": height,
                    "mode": mode
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    df = pd.DataFrame(records)
    
    print(f"\nTotal images processed: {len(df)}")
    print(f"Class distribution:\n{df['label_name'].value_counts()}")
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        print(f"Manifest saved to: {output_file}")
    
    return df


def deduplicate_dataset(
    manifest_path: str = "data/manifests/manifest_raw.parquet",
    config_path: str = "configs/data.yaml",
    output_path: str = "data/manifests/manifest_dedup.parquet"
) -> pd.DataFrame:
    """
    Deduplicate the dataset by removing exact duplicates and clustering near-duplicates.
    
    Args:
        manifest_path: Path to the raw manifest file.
        config_path: Path to data configuration file.
        output_path: Path to save deduplicated manifest.
        
    Returns:
        Deduplicated DataFrame with cluster assignments.
    """
    config = load_config(config_path)
    dedup_config = config["deduplication"]
    
    print("Loading manifest...")
    df = pd.read_parquet(manifest_path)
    original_count = len(df)
    
    # Remove exact duplicates (keep first occurrence)
    print("\nRemoving exact duplicates...")
    df = df.drop_duplicates(subset=["sample_id"], keep="first")
    after_exact = len(df)
    print(f"Removed {original_count - after_exact} exact duplicates")
    
    # Cluster by perceptual hash
    print("\nClustering by perceptual similarity...")
    df = cluster_by_phash(
        df,
        phash_col="phash",
        distance_threshold=dedup_config["cluster_distance_threshold"]
    )
    
    n_clusters = df["cluster_id"].nunique()
    print(f"Found {n_clusters} unique clusters")
    
    # Report cluster statistics
    cluster_sizes = df["cluster_id"].value_counts()
    print(f"\nCluster size distribution:")
    print(f"  Single images: {(cluster_sizes == 1).sum()}")
    print(f"  2-5 images: {((cluster_sizes > 1) & (cluster_sizes <= 5)).sum()}")
    print(f"  >5 images: {(cluster_sizes > 5).sum()}")
    
    # Save deduplicated manifest
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"\nDeduplicated manifest saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deduplicate brain tumor dataset")
    parser.add_argument("--data-dir", "-d", type=str, default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--config", "-c", type=str, default="configs/data.yaml",
                        help="Path to data config file")
    parser.add_argument("--create-manifest", action="store_true",
                        help="Create initial manifest from raw data")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for manifest")
    
    args = parser.parse_args()
    
    if args.create_manifest:
        output = args.output or "data/manifests/manifest_raw.parquet"
        create_manifest(args.data_dir, args.config, output)
    else:
        input_path = args.output or "data/manifests/manifest_raw.parquet"
        output_path = "data/manifests/manifest_dedup.parquet"
        deduplicate_dataset(input_path, args.config, output_path)
