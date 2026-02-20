"""
Data splitting module for OncoSense.
Creates train/val/test splits with cluster-aware stratification to prevent data leakage.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stratified_cluster_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Split data by clusters while maintaining class stratification.
    
    This ensures that images from the same near-duplicate cluster are never
    split across train/val/test sets, preventing data leakage.
    
    Args:
        df: DataFrame with cluster_id and label columns.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        random_seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with added 'split' column.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    df = df.copy()
    np.random.seed(random_seed)
    
    # Get unique clusters with their primary class (most common class in cluster)
    cluster_info = df.groupby("cluster_id").agg({
        "label": lambda x: x.mode()[0],  # Primary class
        "sample_id": "count"  # Cluster size
    }).reset_index()
    cluster_info.columns = ["cluster_id", "primary_label", "size"]
    
    # Split clusters (not individual samples) to prevent leakage
    clusters_by_class = defaultdict(list)
    for _, row in cluster_info.iterrows():
        clusters_by_class[row["primary_label"]].append(row["cluster_id"])
    
    train_clusters = []
    val_clusters = []
    test_clusters = []
    
    # Stratified split for each class
    for label, clusters in clusters_by_class.items():
        clusters = np.array(clusters)
        np.random.shuffle(clusters)
        
        n_clusters = len(clusters)
        n_train = int(n_clusters * train_ratio)
        n_val = int(n_clusters * val_ratio)
        
        train_clusters.extend(clusters[:n_train])
        val_clusters.extend(clusters[n_train:n_train + n_val])
        test_clusters.extend(clusters[n_train + n_val:])
    
    # Assign splits based on cluster membership
    train_set = set(train_clusters)
    val_set = set(val_clusters)
    test_set = set(test_clusters)
    
    def assign_split(cluster_id):
        if cluster_id in train_set:
            return "train"
        elif cluster_id in val_set:
            return "val"
        elif cluster_id in test_set:
            return "test"
        else:
            return "train"  # Default for unclustered
    
    df["split"] = df["cluster_id"].apply(assign_split)
    
    return df


def create_splits(
    manifest_path: str = "data/manifests/manifest_dedup.parquet",
    config_path: str = "configs/data.yaml",
    output_path: str = "data/manifests/manifest.parquet"
) -> pd.DataFrame:
    """
    Create train/val/test splits from the deduplicated manifest.
    
    Args:
        manifest_path: Path to the deduplicated manifest.
        config_path: Path to data configuration file.
        output_path: Path to save the final manifest with splits.
        
    Returns:
        DataFrame with split assignments.
    """
    config = load_config(config_path)
    split_config = config["splits"]
    
    print("Loading deduplicated manifest...")
    df = pd.read_parquet(manifest_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Unique clusters: {df['cluster_id'].nunique()}")
    
    # Apply cluster-aware stratified split
    print("\nApplying cluster-aware stratified split...")
    df = stratified_cluster_split(
        df,
        train_ratio=split_config["train_ratio"],
        val_ratio=split_config["val_ratio"],
        test_ratio=split_config["test_ratio"],
        random_seed=split_config["random_seed"]
    )
    
    # Report split statistics
    print("\n" + "=" * 50)
    print("Split Statistics")
    print("=" * 50)
    
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        print(f"\n{split_name.upper()} set:")
        print(f"  Total samples: {len(split_df)}")
        print(f"  Unique clusters: {split_df['cluster_id'].nunique()}")
        print("  Class distribution:")
        for label_name in df["label_name"].unique():
            count = len(split_df[split_df["label_name"] == label_name])
            pct = count / len(split_df) * 100
            print(f"    {label_name}: {count} ({pct:.1f}%)")
    
    # Verify no cluster leakage
    train_clusters = set(df[df["split"] == "train"]["cluster_id"])
    val_clusters = set(df[df["split"] == "val"]["cluster_id"])
    test_clusters = set(df[df["split"] == "test"]["cluster_id"])
    
    train_val_overlap = train_clusters & val_clusters
    train_test_overlap = train_clusters & test_clusters
    val_test_overlap = val_clusters & test_clusters
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\nWARNING: Cluster leakage detected!")
        print(f"  Train-Val overlap: {len(train_val_overlap)}")
        print(f"  Train-Test overlap: {len(train_test_overlap)}")
        print(f"  Val-Test overlap: {len(val_test_overlap)}")
    else:
        print("\nNo cluster leakage detected.")
    
    # Save final manifest
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"\nFinal manifest saved to: {output_file}")
    
    return df


def get_split_dataframes(
    manifest_path: str = "data/manifests/manifest.parquet"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load manifest and return separate DataFrames for each split.
    
    Args:
        manifest_path: Path to the manifest file.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = pd.read_parquet(manifest_path)
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    return train_df, val_df, test_df


def print_manifest_summary(manifest_path: str = "data/manifests/manifest.parquet"):
    """Print summary statistics for the manifest."""
    df = pd.read_parquet(manifest_path)
    
    print("=" * 60)
    print("MANIFEST SUMMARY")
    print("=" * 60)
    print(f"\nTotal samples: {len(df)}")
    print(f"Unique SHA256 hashes: {df['sample_id'].nunique()}")
    print(f"Unique clusters: {df['cluster_id'].nunique()}")
    
    print("\n--- Class Distribution ---")
    class_dist = df["label_name"].value_counts()
    for label, count in class_dist.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n--- Split Distribution ---")
    split_dist = df["split"].value_counts()
    for split, count in split_dist.items():
        print(f"  {split}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n--- Class x Split Distribution ---")
    cross_tab = pd.crosstab(df["label_name"], df["split"])
    print(cross_tab)
    
    print("\n--- Image Dimensions ---")
    print(f"  Width range: {df['width'].min()} - {df['width'].max()}")
    print(f"  Height range: {df['height'].min()} - {df['height'].max()}")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--input", "-i", type=str, 
                        default="data/manifests/manifest_dedup.parquet",
                        help="Path to deduplicated manifest")
    parser.add_argument("--config", "-c", type=str, default="configs/data.yaml",
                        help="Path to data config file")
    parser.add_argument("--output", "-o", type=str, 
                        default="data/manifests/manifest.parquet",
                        help="Output path for final manifest")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Print summary of existing manifest")
    
    args = parser.parse_args()
    
    if args.summary:
        print_manifest_summary(args.output)
    else:
        create_splits(args.input, args.config, args.output)
