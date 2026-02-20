#!/usr/bin/env python3
"""Run the data processing pipeline."""

import sys
import os

# Change to oncosense directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

print("=" * 60)
print("OncoSense Data Processing Pipeline")
print("=" * 60)

# Step 1: Check manifest exists
print("\n[1/3] Checking raw manifest...")
import pandas as pd
from pathlib import Path

manifest_raw = Path("data/manifests/manifest_raw.parquet")
if manifest_raw.exists():
    df = pd.read_parquet(manifest_raw)
    print(f"  Raw manifest found: {len(df)} images")
    print(f"  Classes: {df['label_name'].value_counts().to_dict()}")
else:
    print("  ERROR: Raw manifest not found!")
    sys.exit(1)

# Step 2: Deduplicate
print("\n[2/3] Deduplicating dataset...")
from src.data.dedup import deduplicate_dataset

try:
    df_dedup = deduplicate_dataset(
        manifest_path="data/manifests/manifest_raw.parquet",
        config_path="configs/data.yaml",
        output_path="data/manifests/manifest_dedup.parquet"
    )
    print(f"  Deduplicated: {len(df_dedup)} images remain")
except Exception as e:
    print(f"  Error during dedup: {e}")
    # Copy raw to dedup if dedup fails
    import shutil
    shutil.copy("data/manifests/manifest_raw.parquet", "data/manifests/manifest_dedup.parquet")
    df_dedup = pd.read_parquet("data/manifests/manifest_dedup.parquet")
    df_dedup['cluster_id'] = range(len(df_dedup))
    df_dedup.to_parquet("data/manifests/manifest_dedup.parquet")
    print(f"  Fallback: assigned unique cluster IDs to {len(df_dedup)} images")

# Step 3: Create splits
print("\n[3/3] Creating train/val/test splits...")
from src.data.split import create_splits

df_final = create_splits(
    manifest_path="data/manifests/manifest_dedup.parquet",
    config_path="configs/data.yaml",
    output_path="data/manifests/manifest.parquet"
)

print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)
print(f"\nFinal dataset: {len(df_final)} images")
print(f"Train: {len(df_final[df_final['split'] == 'train'])}")
print(f"Val: {len(df_final[df_final['split'] == 'val'])}")
print(f"Test: {len(df_final[df_final['split'] == 'test'])}")
print("\nNext step: Train models with:")
print("  python3 -m src.models.train --model densenet121")
