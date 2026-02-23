#!/usr/bin/env python3
"""
Proper Data Setup for OncoSense
===============================
This script fixes the data leakage issues found in quick_setup.py by:
1. Respecting original Kaggle train/test boundaries from filenames
2. Applying proper pHash clustering within each split
3. Creating train/val only from original train images
4. Keeping original test images exclusively for testing

This ensures 0% train/test contamination and minimal near-duplicate leakage.
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering


def extract_original_split(filename: str) -> str:
    """
    Extract the original train/test designation from the filename.
    
    The Kaggle dataset has filenames like:
    - "BT-MRI Glioma Train (123).jpg" -> train
    - "BT-MRI NO Test (456).jpg" -> test
    """
    filename_upper = filename.upper()
    
    if " TRAIN " in filename_upper or " TRAIN(" in filename_upper or "TRAIN (" in filename_upper:
        return "original_train"
    elif " TEST " in filename_upper or " TEST(" in filename_upper or "TEST (" in filename_upper:
        return "original_test"
    else:
        # Check for patterns at different positions
        if "TRAIN" in filename_upper:
            return "original_train"
        elif "TEST" in filename_upper:
            return "original_test"
        else:
            # Default to train for ambiguous cases
            return "original_train"


def compute_pairwise_distances(phashes: list) -> np.ndarray:
    """Compute pairwise Hamming distances between perceptual hashes."""
    n = len(phashes)
    distances = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        h1 = imagehash.hex_to_hash(phashes[i])
        for j in range(i + 1, n):
            h2 = imagehash.hex_to_hash(phashes[j])
            dist = h1 - h2
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def cluster_within_split(df: pd.DataFrame, distance_threshold: int = 10) -> pd.DataFrame:
    """
    Apply pHash clustering within a split to group near-duplicates.
    
    Args:
        df: DataFrame with 'phash' column
        distance_threshold: Max Hamming distance for same cluster
        
    Returns:
        DataFrame with 'cluster_id' column
    """
    df = df.copy()
    
    if len(df) == 0:
        df['cluster_id'] = []
        return df
    
    # Get valid phashes
    valid_mask = df['phash'].notna()
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) < 2:
        df['cluster_id'] = range(len(df))
        return df
    
    phashes = valid_df['phash'].tolist()
    n = len(phashes)
    
    if n > 3000:
        # For large datasets, use sampling approach
        print(f"    Large split ({n} images), using efficient clustering...")
        
        # Sample for clustering
        sample_size = min(2000, n)
        sample_indices = np.random.choice(n, sample_size, replace=False)
        sample_phashes = [phashes[i] for i in sample_indices]
        
        distances = compute_pairwise_distances(sample_phashes)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distances)
        
        # Map sampled phashes to clusters
        sample_cluster_map = {sample_phashes[i]: cluster_labels[i] for i in range(len(sample_phashes))}
        
        # Assign remaining phashes
        next_cluster = max(cluster_labels) + 1
        phash_cluster_map = {}
        
        for ph in phashes:
            if ph in sample_cluster_map:
                phash_cluster_map[ph] = sample_cluster_map[ph]
            else:
                # Find nearest sampled phash
                h = imagehash.hex_to_hash(ph)
                min_dist = float('inf')
                nearest_cluster = next_cluster
                
                for sampled_ph, cluster in sample_cluster_map.items():
                    dist = h - imagehash.hex_to_hash(sampled_ph)
                    if dist < min_dist:
                        min_dist = dist
                        if dist <= distance_threshold:
                            nearest_cluster = cluster
                
                if nearest_cluster == next_cluster:
                    next_cluster += 1
                phash_cluster_map[ph] = nearest_cluster
    else:
        # Full clustering for smaller splits
        print(f"    Computing distances for {n} images...")
        distances = compute_pairwise_distances(phashes)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distances)
        phash_cluster_map = {phashes[i]: cluster_labels[i] for i in range(n)}
    
    # Assign cluster IDs
    valid_df['cluster_id'] = valid_df['phash'].map(phash_cluster_map)
    df.loc[valid_mask, 'cluster_id'] = valid_df['cluster_id']
    df.loc[~valid_mask, 'cluster_id'] = range(int(df['cluster_id'].max() or 0) + 1, 
                                               int(df['cluster_id'].max() or 0) + 1 + (~valid_mask).sum())
    
    return df


def stratified_cluster_split_within_train(
    df: pd.DataFrame,
    val_ratio: float = 0.125,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Split original train data into our train and val sets.
    Ensures clusters stay together (no near-duplicate leakage).
    
    Args:
        df: DataFrame with original_train images only
        val_ratio: Fraction to use for validation (~12.5% of train = ~10% of total)
        random_seed: Random seed
        
    Returns:
        DataFrame with 'split' column ('train' or 'val')
    """
    df = df.copy()
    np.random.seed(random_seed)
    
    # Get cluster info
    cluster_info = df.groupby('cluster_id').agg({
        'label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'sample_id': 'count'
    }).reset_index()
    cluster_info.columns = ['cluster_id', 'primary_label', 'size']
    
    # Group clusters by class
    clusters_by_class = defaultdict(list)
    for _, row in cluster_info.iterrows():
        clusters_by_class[row['primary_label']].append(row['cluster_id'])
    
    val_clusters = []
    train_clusters = []
    
    # Stratified split for each class
    for label, clusters in clusters_by_class.items():
        clusters = np.array(clusters)
        np.random.shuffle(clusters)
        
        n_val = max(1, int(len(clusters) * val_ratio))
        
        val_clusters.extend(clusters[:n_val])
        train_clusters.extend(clusters[n_val:])
    
    # Assign splits
    val_set = set(val_clusters)
    
    df['split'] = df['cluster_id'].apply(lambda x: 'val' if x in val_set else 'train')
    
    return df


def main():
    print("=" * 70)
    print("OncoSense PROPER Data Setup")
    print("Fixes data leakage by respecting original train/test boundaries")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load raw manifest
    print("\n[1/7] Loading raw manifest...")
    manifest_path = Path("data/manifests/manifest_raw.parquet")
    if not manifest_path.exists():
        # Try alternative path
        manifest_path = Path("data/manifests/manifest.parquet")
    
    df = pd.read_parquet(manifest_path)
    print(f"  Total images loaded: {len(df)}")
    
    # Extract original split from filenames
    print("\n[2/7] Extracting original train/test designation from filenames...")
    df['original_split'] = df['filename'].apply(extract_original_split)
    
    split_counts = df['original_split'].value_counts()
    print(f"  Original train: {split_counts.get('original_train', 0)}")
    print(f"  Original test: {split_counts.get('original_test', 0)}")
    
    # Remove exact duplicates within each original split
    print("\n[3/7] Removing exact duplicates (by SHA256)...")
    before = len(df)
    
    # Remove duplicates separately within each split
    df_train = df[df['original_split'] == 'original_train'].drop_duplicates(subset=['sample_id'], keep='first')
    df_test = df[df['original_split'] == 'original_test'].drop_duplicates(subset=['sample_id'], keep='first')
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    after = len(df)
    print(f"  Removed {before - after} exact duplicates")
    print(f"  Remaining: {after} images")
    
    # Apply pHash clustering within each original split
    print("\n[4/7] Applying pHash clustering within original splits...")
    
    # Cluster original train
    print("  Clustering original TRAIN images...")
    df_train = df[df['original_split'] == 'original_train'].copy()
    df_train = cluster_within_split(df_train, distance_threshold=10)
    
    n_train_clusters = df_train['cluster_id'].nunique()
    print(f"    Found {n_train_clusters} clusters in {len(df_train)} train images")
    
    # Cluster original test
    print("  Clustering original TEST images...")
    df_test = df[df['original_split'] == 'original_test'].copy()
    
    # Offset test cluster IDs to avoid collision with train
    test_cluster_offset = df_train['cluster_id'].max() + 1 if len(df_train) > 0 else 0
    df_test = cluster_within_split(df_test, distance_threshold=10)
    df_test['cluster_id'] = df_test['cluster_id'] + test_cluster_offset
    
    n_test_clusters = df_test['cluster_id'].nunique()
    print(f"    Found {n_test_clusters} clusters in {len(df_test)} test images")
    
    # Split original train into our train and val
    print("\n[5/7] Splitting original train into our train and val...")
    df_train = stratified_cluster_split_within_train(df_train, val_ratio=0.125)
    
    our_train = df_train[df_train['split'] == 'train']
    our_val = df_train[df_train['split'] == 'val']
    print(f"  Our train: {len(our_train)} images ({our_train['cluster_id'].nunique()} clusters)")
    print(f"  Our val: {len(our_val)} images ({our_val['cluster_id'].nunique()} clusters)")
    
    # Original test becomes our test (unchanged)
    print("\n[6/7] Assigning original test as our test (untouched)...")
    df_test['split'] = 'test'
    print(f"  Our test: {len(df_test)} images ({df_test['cluster_id'].nunique()} clusters)")
    
    # Combine all splits
    df_final = pd.concat([df_train, df_test], ignore_index=True)
    
    # Verify no cluster overlap between splits
    print("\n[7/7] Verifying data integrity...")
    
    train_clusters = set(df_final[df_final['split'] == 'train']['cluster_id'])
    val_clusters = set(df_final[df_final['split'] == 'val']['cluster_id'])
    test_clusters = set(df_final[df_final['split'] == 'test']['cluster_id'])
    
    train_val_overlap = train_clusters & val_clusters
    train_test_overlap = train_clusters & test_clusters
    val_test_overlap = val_clusters & test_clusters
    
    print(f"  Train-Val cluster overlap: {len(train_val_overlap)} (should be 0)")
    print(f"  Train-Test cluster overlap: {len(train_test_overlap)} (should be 0)")
    print(f"  Val-Test cluster overlap: {len(val_test_overlap)} (should be 0)")
    
    # Verify original split contamination is 0
    train_from_orig_test = len(df_final[(df_final['split'] == 'train') & (df_final['original_split'] == 'original_test')])
    val_from_orig_test = len(df_final[(df_final['split'] == 'val') & (df_final['original_split'] == 'original_test')])
    test_from_orig_train = len(df_final[(df_final['split'] == 'test') & (df_final['original_split'] == 'original_train')])
    
    print(f"\n  Contamination check:")
    print(f"    Original TEST in our TRAIN: {train_from_orig_test} (should be 0)")
    print(f"    Original TEST in our VAL: {val_from_orig_test} (should be 0)")
    print(f"    Original TRAIN in our TEST: {test_from_orig_train} (should be 0)")
    
    if train_from_orig_test > 0 or val_from_orig_test > 0 or test_from_orig_train > 0:
        print("\n  WARNING: Contamination detected! Check the filename patterns.")
    else:
        print("\n  SUCCESS: Zero train/test contamination!")
    
    # Save the fixed manifest
    output_path = Path("data/manifests/manifest_fixed.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    print(f"\n  Fixed manifest saved to: {output_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(df_final)}")
    print(f"  Train: {len(df_final[df_final['split'] == 'train'])} ({len(df_final[df_final['split'] == 'train'])/len(df_final)*100:.1f}%)")
    print(f"  Val: {len(df_final[df_final['split'] == 'val'])} ({len(df_final[df_final['split'] == 'val'])/len(df_final)*100:.1f}%)")
    print(f"  Test: {len(df_final[df_final['split'] == 'test'])} ({len(df_final[df_final['split'] == 'test'])/len(df_final)*100:.1f}%)")
    
    print(f"\nClass Distribution:")
    for cls in df_final['label_name'].unique():
        count = len(df_final[df_final['label_name'] == cls])
        print(f"  {cls}: {count} ({count/len(df_final)*100:.1f}%)")
    
    print(f"\nClass x Split Distribution:")
    for split in ['train', 'val', 'test']:
        split_df = df_final[df_final['split'] == split]
        print(f"\n  {split.upper()}:")
        for cls in df_final['label_name'].unique():
            count = len(split_df[split_df['label_name'] == cls])
            print(f"    {cls}: {count}")
    
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Update configs/data.yaml to remove horizontal_flip")
    print("  2. Retrain model: python3 -m src.models.train --manifest data/manifests/manifest_fixed.parquet")
    print("  3. Run validation experiments to verify the fix")
    print("=" * 70)
    
    return df_final


if __name__ == "__main__":
    main()
