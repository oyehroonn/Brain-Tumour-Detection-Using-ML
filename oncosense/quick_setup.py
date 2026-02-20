#!/usr/bin/env python3
"""Quick data setup - skips expensive clustering, uses simple dedup."""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path

print("=" * 60)
print("OncoSense Quick Data Setup")
print("=" * 60)

# Load raw manifest
print("\n[1/3] Loading raw manifest...")
df = pd.read_parquet("data/manifests/manifest_raw.parquet")
print(f"  Total images: {len(df)}")

# Simple deduplication - just remove exact duplicates by hash
print("\n[2/3] Removing exact duplicates...")
before = len(df)
df = df.drop_duplicates(subset=["sample_id"], keep="first")
after = len(df)
print(f"  Removed {before - after} exact duplicates")
print(f"  Remaining: {after} images")

# Assign unique cluster IDs (each image is its own cluster)
df["cluster_id"] = range(len(df))

# Save deduplicated manifest
df.to_parquet("data/manifests/manifest_dedup.parquet", index=False)
print("  Saved to data/manifests/manifest_dedup.parquet")

# Create splits
print("\n[3/3] Creating train/val/test splits...")
from src.data.split import create_splits

df_final = create_splits(
    manifest_path="data/manifests/manifest_dedup.parquet",
    config_path="configs/data.yaml",
    output_path="data/manifests/manifest.parquet"
)

print("\n" + "=" * 60)
print("Setup Complete!")
print("=" * 60)

# Show summary
print(f"\nDataset Summary:")
print(f"  Total: {len(df_final)} images")
print(f"  Train: {len(df_final[df_final['split'] == 'train'])} ({len(df_final[df_final['split'] == 'train'])/len(df_final)*100:.1f}%)")
print(f"  Val:   {len(df_final[df_final['split'] == 'val'])} ({len(df_final[df_final['split'] == 'val'])/len(df_final)*100:.1f}%)")
print(f"  Test:  {len(df_final[df_final['split'] == 'test'])} ({len(df_final[df_final['split'] == 'test'])/len(df_final)*100:.1f}%)")

print("\n  Class distribution:")
for cls in ["glioma", "meningioma", "pituitary", "no_tumor"]:
    count = len(df_final[df_final['label_name'] == cls])
    print(f"    {cls}: {count}")

print("\n" + "=" * 60)
print("Ready to train! Run:")
print("  python3 -m src.models.train --model densenet121")
print("=" * 60)
