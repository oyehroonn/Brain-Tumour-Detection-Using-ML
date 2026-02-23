#!/usr/bin/env python3
"""
Master Validation Experiment Runner

Runs all validation experiments in the recommended order:
1. Phase 1 - Quick Diagnostics (no training required)
   - Experiment 7: Split Analysis
   - Experiment 6: Leakage Check  
   - Experiment 5: Grad-CAM Analysis

2. Phase 2 - Training Experiments
   - Experiment 1: Random Label Test
   - Experiment 2: Center Crop Test

3. Phase 3 - Full Validation
   - Experiment 4: Proper Clustering
   - Experiment 3: External Dataset

Results are saved to validation_results/ and a summary report is generated.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR


def run_phase1_diagnostics(
    skip_gradcam: bool = False,
    checkpoint_path: str = None
) -> Dict[str, Dict]:
    """
    Run Phase 1: Quick diagnostics (no training required).
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("PHASE 1: QUICK DIAGNOSTICS")
    print("=" * 70)
    
    # Experiment 7: Split Analysis
    print("\n" + "-" * 60)
    print("Running Experiment 7: Original Train/Test Split Analysis")
    print("-" * 60)
    try:
        from src.validation.split_analysis import run_experiment as run_split
        results['experiment_7_split'] = run_split()
    except Exception as e:
        print(f"ERROR: {e}")
        results['experiment_7_split'] = {'error': str(e)}
    
    # Experiment 6: Leakage Check
    print("\n" + "-" * 60)
    print("Running Experiment 6: Near-Duplicate Leakage Check")
    print("-" * 60)
    try:
        from src.validation.leakage_check import run_experiment as run_leakage
        results['experiment_6_leakage'] = run_leakage(sample_size=300)
    except Exception as e:
        print(f"ERROR: {e}")
        results['experiment_6_leakage'] = {'error': str(e)}
    
    # Experiment 5: Grad-CAM Analysis
    if not skip_gradcam:
        print("\n" + "-" * 60)
        print("Running Experiment 5: Grad-CAM Analysis")
        print("-" * 60)
        try:
            from src.validation.gradcam_analysis import run_experiment as run_gradcam
            results['experiment_5_gradcam'] = run_gradcam(
                checkpoint_path=checkpoint_path,
                num_samples=50
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results['experiment_5_gradcam'] = {'error': str(e)}
    else:
        print("\n[Skipping Experiment 5: Grad-CAM Analysis]")
        results['experiment_5_gradcam'] = {'skipped': True}
    
    return results


def run_phase2_training(
    skip_random_label: bool = False,
    skip_center_crop: bool = False,
    num_epochs: int = 10,
    device: str = None
) -> Dict[str, Dict]:
    """
    Run Phase 2: Training experiments.
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING EXPERIMENTS")
    print("=" * 70)
    
    # Experiment 1: Random Label Test
    if not skip_random_label:
        print("\n" + "-" * 60)
        print("Running Experiment 1: Random Label Shuffle Test")
        print("-" * 60)
        try:
            from src.validation.random_label_test import run_experiment as run_random
            results['experiment_1_random_labels'] = run_random(
                num_epochs=num_epochs,
                device=device
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results['experiment_1_random_labels'] = {'error': str(e)}
    else:
        print("\n[Skipping Experiment 1: Random Label Test]")
        results['experiment_1_random_labels'] = {'skipped': True}
    
    # Experiment 2: Center Crop Test
    if not skip_center_crop:
        print("\n" + "-" * 60)
        print("Running Experiment 2: Center Crop Only Test")
        print("-" * 60)
        try:
            from src.validation.center_crop_test import run_experiment as run_center
            results['experiment_2_center_crop'] = run_center(
                num_epochs=num_epochs,
                device=device
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results['experiment_2_center_crop'] = {'error': str(e)}
    else:
        print("\n[Skipping Experiment 2: Center Crop Test]")
        results['experiment_2_center_crop'] = {'skipped': True}
    
    return results


def run_phase3_full_validation(
    skip_clustering: bool = False,
    skip_external: bool = False,
    checkpoint_path: str = None,
    original_accuracy: float = 0.98,
    device: str = None
) -> Dict[str, Dict]:
    """
    Run Phase 3: Full validation experiments.
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("PHASE 3: FULL VALIDATION")
    print("=" * 70)
    
    # Experiment 4: Proper Clustering
    if not skip_clustering:
        print("\n" + "-" * 60)
        print("Running Experiment 4: Proper pHash Clustering")
        print("-" * 60)
        try:
            from src.validation.proper_clustering import run_experiment as run_cluster
            results['experiment_4_clustering'] = run_cluster(
                original_best_accuracy=original_accuracy,
                device=device
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results['experiment_4_clustering'] = {'error': str(e)}
    else:
        print("\n[Skipping Experiment 4: Proper Clustering]")
        results['experiment_4_clustering'] = {'skipped': True}
    
    # Experiment 3: External Validation
    if not skip_external:
        print("\n" + "-" * 60)
        print("Running Experiment 3: External Dataset Validation")
        print("-" * 60)
        try:
            from src.validation.external_validation import run_experiment as run_external
            results['experiment_3_external'] = run_external(
                checkpoint_path=checkpoint_path,
                original_accuracy=original_accuracy
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results['experiment_3_external'] = {'error': str(e)}
    else:
        print("\n[Skipping Experiment 3: External Validation]")
        results['experiment_3_external'] = {'skipped': True}
    
    return results


def generate_summary(all_results: Dict) -> Dict:
    """Generate a summary of all experiment results."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments_run': [],
        'experiments_passed': [],
        'experiments_warning': [],
        'experiments_critical': [],
        'experiments_skipped': [],
        'experiments_error': [],
        'overall_verdict': None
    }
    
    for exp_name, exp_results in all_results.items():
        summary['experiments_run'].append(exp_name)
        
        if 'skipped' in exp_results:
            summary['experiments_skipped'].append(exp_name)
        elif 'error' in exp_results:
            summary['experiments_error'].append(exp_name)
        elif 'verdict' in exp_results:
            severity = exp_results['verdict'].get('severity', 'unknown')
            if severity == 'pass':
                summary['experiments_passed'].append(exp_name)
            elif severity == 'warning':
                summary['experiments_warning'].append(exp_name)
            elif severity in ['critical', 'notice']:
                summary['experiments_critical'].append(exp_name)
    
    # Overall verdict
    if summary['experiments_critical']:
        summary['overall_verdict'] = 'CRITICAL: Issues detected - accuracy likely inflated'
    elif summary['experiments_warning']:
        summary['overall_verdict'] = 'WARNING: Some concerns - requires further investigation'
    elif summary['experiments_error']:
        summary['overall_verdict'] = 'INCOMPLETE: Some experiments failed to run'
    else:
        summary['overall_verdict'] = 'PASS: No major issues detected'
    
    return summary


def print_final_summary(summary: Dict):
    """Print a formatted summary of all results."""
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\nExperiments Run: {len(summary['experiments_run'])}")
    print(f"  Passed: {len(summary['experiments_passed'])}")
    print(f"  Warnings: {len(summary['experiments_warning'])}")
    print(f"  Critical: {len(summary['experiments_critical'])}")
    print(f"  Skipped: {len(summary['experiments_skipped'])}")
    print(f"  Errors: {len(summary['experiments_error'])}")
    
    if summary['experiments_passed']:
        print(f"\n✓ PASSED: {', '.join(summary['experiments_passed'])}")
    
    if summary['experiments_warning']:
        print(f"\n⚠ WARNINGS: {', '.join(summary['experiments_warning'])}")
    
    if summary['experiments_critical']:
        print(f"\n✗ CRITICAL: {', '.join(summary['experiments_critical'])}")
    
    if summary['experiments_error']:
        print(f"\n✗ ERRORS: {', '.join(summary['experiments_error'])}")
    
    print("\n" + "=" * 70)
    print(f"OVERALL VERDICT: {summary['overall_verdict']}")
    print("=" * 70)


def run_all_experiments(
    phases: List[int] = None,
    skip_experiments: List[int] = None,
    checkpoint_path: str = None,
    original_accuracy: float = 0.98,
    num_epochs: int = 10,
    device: str = None
) -> Dict:
    """
    Run all validation experiments.
    
    Args:
        phases: List of phases to run (1, 2, 3). Default: all.
        skip_experiments: List of experiment numbers to skip (1-7).
        checkpoint_path: Path to trained model checkpoint.
        original_accuracy: Original model's best accuracy.
        num_epochs: Number of epochs for training experiments.
        device: Device to use.
        
    Returns:
        Dictionary with all results and summary.
    """
    os.chdir(Path(__file__).parent.parent.parent)
    
    if phases is None:
        phases = [1, 2, 3]
    if skip_experiments is None:
        skip_experiments = []
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/densenet121_best.pt"
    
    print("=" * 70)
    print("OncoSense Model Validation Suite")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Phases to run: {phases}")
    print(f"Experiments to skip: {skip_experiments}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Original accuracy: {original_accuracy*100:.1f}%")
    
    all_results = {}
    
    # Phase 1: Quick Diagnostics
    if 1 in phases:
        phase1_results = run_phase1_diagnostics(
            skip_gradcam=(5 in skip_experiments),
            checkpoint_path=checkpoint_path
        )
        all_results.update(phase1_results)
    
    # Phase 2: Training Experiments
    if 2 in phases:
        phase2_results = run_phase2_training(
            skip_random_label=(1 in skip_experiments),
            skip_center_crop=(2 in skip_experiments),
            num_epochs=num_epochs,
            device=device
        )
        all_results.update(phase2_results)
    
    # Phase 3: Full Validation
    if 3 in phases:
        phase3_results = run_phase3_full_validation(
            skip_clustering=(4 in skip_experiments),
            skip_external=(3 in skip_experiments),
            checkpoint_path=checkpoint_path,
            original_accuracy=original_accuracy,
            device=device
        )
        all_results.update(phase3_results)
    
    # Generate summary
    summary = generate_summary(all_results)
    all_results['summary'] = summary
    
    # Print summary
    print_final_summary(summary)
    
    # Save all results
    results_file = VALIDATION_RESULTS_DIR / "all_experiments_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OncoSense validation experiments")
    
    parser.add_argument("--phases", "-p", type=int, nargs='+', default=[1, 2, 3],
                       help="Phases to run (1, 2, 3)")
    parser.add_argument("--skip", "-s", type=int, nargs='+', default=[],
                       help="Experiment numbers to skip (1-7)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--accuracy", "-a", type=float, default=0.98,
                       help="Original model accuracy")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                       help="Training epochs for experiments")
    parser.add_argument("--device", "-d", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick mode: only Phase 1 diagnostics")
    
    args = parser.parse_args()
    
    if args.quick:
        args.phases = [1]
    
    run_all_experiments(
        phases=args.phases,
        skip_experiments=args.skip,
        checkpoint_path=args.checkpoint,
        original_accuracy=args.accuracy,
        num_epochs=args.epochs,
        device=args.device
    )
