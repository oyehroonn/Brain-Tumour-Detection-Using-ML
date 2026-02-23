"""
Validation module for OncoSense.
Implements comprehensive experiments to validate model accuracy and detect data leakage.

Experiments:
    1. Random Label Test - Detect data leakage via random label training
    2. Center Crop Test - Detect spurious pattern learning
    3. External Validation - Test generalization on Figshare dataset
    4. Proper Clustering - Re-run with correct pHash clustering
    5. Grad-CAM Analysis - Visualize model attention
    6. Leakage Check - Find near-duplicates across splits
    7. Split Analysis - Analyze original train/test contamination
"""

from pathlib import Path

VALIDATION_RESULTS_DIR = Path(__file__).parent.parent.parent / "validation_results"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

__all__ = [
    "random_label_test",
    "center_crop_test",
    "external_validation",
    "proper_clustering",
    "gradcam_analysis",
    "leakage_check",
    "split_analysis",
    "run_all_experiments",
    "generate_report",
]
