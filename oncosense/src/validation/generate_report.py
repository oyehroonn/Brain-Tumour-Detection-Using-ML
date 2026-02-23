#!/usr/bin/env python3
"""
Validation Report Generator

Aggregates all experiment results into a comprehensive markdown report.

The report includes:
1. Executive Summary (Pass/Fail for each experiment)
2. Detailed findings per experiment
3. Statistical analysis
4. Visual evidence (references to generated images)
5. Conclusion: Is 98% accuracy valid?
6. Recommendations for improvement
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import VALIDATION_RESULTS_DIR


def load_experiment_results(experiment_dir: Path) -> Optional[Dict]:
    """Load results.json from an experiment directory."""
    results_file = experiment_dir / "results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def get_verdict_emoji(severity: str) -> str:
    """Get emoji for verdict severity."""
    return {
        'pass': '✅',
        'notice': '⚠️',
        'warning': '⚠️',
        'critical': '❌'
    }.get(severity, '❓')


def generate_executive_summary(all_results: Dict) -> str:
    """Generate the executive summary section."""
    summary = """## Executive Summary

| Experiment | Status | Key Finding |
|------------|--------|-------------|
"""
    
    experiment_order = [
        ('experiment_7_split', 'Split Analysis'),
        ('experiment_6_leakage', 'Leakage Check'),
        ('experiment_5_gradcam', 'Grad-CAM Analysis'),
        ('experiment_1_random_labels', 'Random Label Test'),
        ('experiment_2_center_crop', 'Center Crop Test'),
        ('experiment_4_clustering', 'Proper Clustering'),
        ('experiment_3_external', 'External Validation'),
    ]
    
    for exp_key, exp_name in experiment_order:
        if exp_key in all_results:
            result = all_results[exp_key]
            if 'skipped' in result:
                summary += f"| {exp_name} | ⏭️ Skipped | N/A |\n"
            elif 'error' in result:
                summary += f"| {exp_name} | ❌ Error | {result.get('error', 'Unknown error')[:50]} |\n"
            elif 'verdict' in result:
                verdict = result['verdict']
                emoji = get_verdict_emoji(verdict.get('severity', 'unknown'))
                finding = verdict.get('summary', 'No summary')[:60]
                summary += f"| {exp_name} | {emoji} {verdict.get('severity', 'unknown').upper()} | {finding} |\n"
        else:
            summary += f"| {exp_name} | ❓ Not Run | N/A |\n"
    
    return summary


def generate_experiment_details(all_results: Dict) -> str:
    """Generate detailed sections for each experiment."""
    sections = ""
    
    # Experiment 7: Split Analysis
    if 'experiment_7_split' in all_results:
        result = all_results['experiment_7_split']
        sections += """
---

## Experiment 7: Original Train/Test Split Analysis

**Purpose:** Check if our random split violated the original train/test labels encoded in filenames.

"""
        if 'verdict' in result:
            sections += f"**Verdict:** {get_verdict_emoji(result['verdict'].get('severity', ''))} {result['verdict'].get('summary', '')}\n\n"
            sections += f"{result['verdict'].get('explanation', '')}\n\n"
            
            if 'contamination_analysis' in result:
                ca = result['contamination_analysis']
                sections += f"""**Key Metrics:**
- Original test images in our train: {ca.get('original_test_in_our_train', 'N/A')} / {ca.get('original_test_total', 'N/A')} = {ca.get('test_to_train_contamination_rate', 0):.1f}%
- Original train images in our test: {ca.get('original_train_in_our_test', 'N/A')} / {ca.get('original_train_total', 'N/A')} = {ca.get('train_to_test_contamination_rate', 0):.1f}%

"""
            sections += "**Visualizations:** `validation_results/experiment_7_split/`\n"
    
    # Experiment 6: Leakage Check
    if 'experiment_6_leakage' in all_results:
        result = all_results['experiment_6_leakage']
        sections += """
---

## Experiment 6: Near-Duplicate Leakage Check

**Purpose:** Quantify perceptually similar images across train/val/test splits using pHash.

"""
        if 'verdict' in result:
            sections += f"**Verdict:** {get_verdict_emoji(result['verdict'].get('severity', ''))} {result['verdict'].get('summary', '')}\n\n"
            sections += f"{result['verdict'].get('explanation', '')}\n\n"
            
            if 'stats' in result:
                stats = result['stats']
                sections += f"""**Key Metrics:**
- Test→Train leakage: {stats.get('test_train_leak_count', 'N/A')} images ({stats.get('test_train_leak_rate', 0):.1f}%)
- Val→Train leakage: {stats.get('val_train_leak_count', 'N/A')} images ({stats.get('val_train_leak_rate', 0):.1f}%)
- Threshold used: {stats.get('threshold_used', 'N/A')} (Hamming distance)

"""
            sections += "**Visualizations:** `validation_results/experiment_6_leakage/`\n"
    
    # Experiment 5: Grad-CAM Analysis
    if 'experiment_5_gradcam' in all_results:
        result = all_results['experiment_5_gradcam']
        sections += """
---

## Experiment 5: Grad-CAM Analysis

**Purpose:** Visualize what regions the model focuses on for predictions.

"""
        if 'verdict' in result:
            sections += f"**Verdict:** {get_verdict_emoji(result['verdict'].get('severity', ''))} {result['verdict'].get('summary', '')}\n\n"
            sections += f"{result['verdict'].get('explanation', '')}\n\n"
            
            if 'global_stats' in result:
                gs = result['global_stats']
                sections += f"""**Key Metrics:**
- Center attention ratio: {gs.get('mean_center_attention', 0)*100:.1f}%
- Edge attention ratio: {gs.get('mean_edge_attention', 0)*100:.1f}%
- Peak in center: {gs.get('peak_in_center_rate', 0)*100:.1f}% of images
- Validation accuracy: {gs.get('accuracy', 0)*100:.1f}%

"""
            sections += "**Visualizations:** `validation_results/experiment_5_gradcam/`\n"
    
    # Experiment 1: Random Label Test
    if 'experiment_1_random_labels' in all_results:
        result = all_results['experiment_1_random_labels']
        sections += """
---

## Experiment 1: Random Label Shuffle Test

**Purpose:** Train with shuffled labels to detect data leakage. Expected: ~25% accuracy.

"""
        if 'verdict' in result:
            sections += f"**Verdict:** {get_verdict_emoji(result['verdict'].get('severity', ''))} {result['verdict'].get('summary', '')}\n\n"
            sections += f"{result['verdict'].get('explanation', '')}\n\n"
            
            if 'final_metrics' in result:
                fm = result['final_metrics']
                sections += f"""**Key Metrics:**
- Expected (random chance): 25%
- Achieved validation accuracy: {fm.get('best_val_accuracy', 0)*100:.1f}%
- Ratio vs random: {fm.get('best_val_accuracy', 0)/0.25:.1f}x

"""
            sections += "**Visualizations:** `validation_results/experiment_1_random_labels/`\n"
    
    # Experiment 2: Center Crop Test
    if 'experiment_2_center_crop' in all_results:
        result = all_results['experiment_2_center_crop']
        sections += """
---

## Experiment 2: Center Crop Only Test

**Purpose:** Train on center region only (no tumor info). Expected: ~25% accuracy.

"""
        if 'verdict' in result:
            sections += f"**Verdict:** {get_verdict_emoji(result['verdict'].get('severity', ''))} {result['verdict'].get('summary', '')}\n\n"
            sections += f"{result['verdict'].get('explanation', '')}\n\n"
            
            if 'final_metrics' in result:
                fm = result['final_metrics']
                sections += f"""**Key Metrics:**
- Center crop size: {result.get('center_size', 64)}x{result.get('center_size', 64)} pixels
- Expected (random chance): 25%
- Achieved validation accuracy: {fm.get('best_val_accuracy', 0)*100:.1f}%

"""
            sections += "**Visualizations:** `validation_results/experiment_2_center_crop/`\n"
    
    # Experiment 4: Proper Clustering
    if 'experiment_4_clustering' in all_results:
        result = all_results['experiment_4_clustering']
        sections += """
---

## Experiment 4: Proper pHash Clustering

**Purpose:** Re-run with proper near-duplicate clustering (bypassed in quick_setup).

"""
        if 'verdict' in result:
            sections += f"**Verdict:** {get_verdict_emoji(result['verdict'].get('severity', ''))} {result['verdict'].get('summary', '')}\n\n"
            sections += f"{result['verdict'].get('explanation', '')}\n\n"
            
            if 'final_metrics' in result:
                fm = result['final_metrics']
                sections += f"""**Key Metrics:**
- Original accuracy: {fm.get('original_accuracy', 0)*100:.1f}%
- With proper clustering: {fm.get('best_val_accuracy', 0)*100:.1f}%
- Accuracy drop: {fm.get('accuracy_drop', 0)*100:.1f}%

"""
            sections += "**Visualizations:** `validation_results/experiment_4_clustering/`\n"
    
    # Experiment 3: External Validation
    if 'experiment_3_external' in all_results:
        result = all_results['experiment_3_external']
        sections += """
---

## Experiment 3: External Dataset Validation

**Purpose:** Test on completely different dataset (Figshare/Jun Cheng).

"""
        if 'verdict' in result:
            sections += f"**Verdict:** {get_verdict_emoji(result['verdict'].get('severity', ''))} {result['verdict'].get('summary', '')}\n\n"
            sections += f"{result['verdict'].get('explanation', '')}\n\n"
            
            if 'external_metrics' in result:
                em = result['external_metrics']
                sections += f"""**Key Metrics:**
- Original accuracy (Kaggle): {result.get('original_accuracy', 0)*100:.1f}%
- External accuracy: {em.get('accuracy', 0)*100:.1f}%
- Generalization gap: {result['verdict'].get('generalization_gap', 0)*100:.1f}%
- External F1 score: {em.get('macro_f1', 0):.4f}

"""
            sections += "**Visualizations:** `validation_results/experiment_3_external/`\n"
    
    return sections


def generate_conclusion(all_results: Dict) -> str:
    """Generate the conclusion and recommendations section."""
    
    # Count issues
    critical_count = 0
    warning_count = 0
    pass_count = 0
    
    for exp_key, result in all_results.items():
        if isinstance(result, dict) and 'verdict' in result:
            severity = result['verdict'].get('severity', '')
            if severity in ['critical']:
                critical_count += 1
            elif severity in ['warning', 'notice']:
                warning_count += 1
            elif severity == 'pass':
                pass_count += 1
    
    conclusion = """
---

## Conclusion

"""
    
    if critical_count > 0:
        conclusion += f"""### ❌ The 98% accuracy is likely NOT valid

**{critical_count} critical issue(s)** were detected during validation. The high accuracy
is most likely due to:

1. **Data leakage** from near-duplicate images across train/test splits
2. **Train/test contamination** from the original dataset labels
3. **Spurious pattern learning** from non-tumor features
4. **Dataset-specific artifacts** that don't generalize

The model has learned dataset-specific shortcuts rather than genuine tumor features.

"""
    elif warning_count > 0:
        conclusion += f"""### ⚠️ The 98% accuracy is questionable

**{warning_count} warning(s)** were detected. While no critical issues were found,
there are concerns about the validity of the accuracy claim:

- Some degree of data leakage or bias appears present
- Generalization capability may be limited
- Further investigation is recommended

"""
    else:
        conclusion += f"""### ✅ The 98% accuracy appears valid

All validation experiments passed without major issues. The model appears to have
learned genuine tumor features. However, continued monitoring and external validation
is always recommended for medical AI applications.

"""
    
    # Recommendations
    conclusion += """
## Recommendations

### Immediate Actions
1. **Do NOT deploy this model clinically** until validation issues are resolved
2. Review the individual experiment results and visualizations
3. Address any critical issues identified

### Data Improvements
1. Re-run data preparation with proper pHash clustering enabled
2. Obtain patient IDs if possible and ensure patient-level splits
3. Consider acquiring data from additional sources

### Model Improvements
1. Implement domain adaptation techniques
2. Add explicit bias detection mechanisms
3. Use ensemble methods with uncertainty quantification

### Validation Best Practices
1. Always validate on truly external datasets
2. Use Grad-CAM to verify model reasoning
3. Test with corrupted/perturbed inputs

"""
    
    return conclusion


def generate_report(
    results_path: str = None,
    output_path: str = None
) -> str:
    """
    Generate a comprehensive validation report.
    
    Args:
        results_path: Path to all_experiments_results.json
        output_path: Path to save the report
        
    Returns:
        The report content as a string
    """
    os.chdir(Path(__file__).parent.parent.parent)
    
    if results_path is None:
        results_path = VALIDATION_RESULTS_DIR / "all_experiments_results.json"
    if output_path is None:
        output_path = VALIDATION_RESULTS_DIR / "final_validation_report.md"
    
    print("=" * 60)
    print("Generating Validation Report")
    print("=" * 60)
    
    # Load results
    all_results = {}
    
    # Try to load aggregated results
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded aggregated results from {results_path}")
    else:
        # Load individual experiment results
        print("No aggregated results found. Loading individual experiments...")
        
        experiment_dirs = [
            'experiment_1_random_labels',
            'experiment_2_center_crop',
            'experiment_3_external',
            'experiment_4_clustering',
            'experiment_5_gradcam',
            'experiment_6_leakage',
            'experiment_7_split'
        ]
        
        for exp_dir in experiment_dirs:
            exp_path = VALIDATION_RESULTS_DIR / exp_dir
            if exp_path.exists():
                result = load_experiment_results(exp_path)
                if result:
                    all_results[exp_dir] = result
                    print(f"  Loaded {exp_dir}")
    
    if not all_results:
        print("\nNo experiment results found!")
        print("Run experiments first using:")
        print("  python3 -m src.validation.run_all_experiments")
        return ""
    
    # Generate report sections
    report = f"""# OncoSense Model Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report summarizes the results of comprehensive validation experiments
designed to verify whether the reported 98% accuracy is valid or an artifact
of data leakage, bias, or other issues.

"""
    
    report += generate_executive_summary(all_results)
    report += generate_experiment_details(all_results)
    report += generate_conclusion(all_results)
    
    # Add appendix
    report += """
---

## Appendix: Running Individual Experiments

```bash
# Phase 1: Quick diagnostics (no training)
python3 -m src.validation.split_analysis
python3 -m src.validation.leakage_check
python3 -m src.validation.gradcam_analysis

# Phase 2: Training experiments
python3 -m src.validation.random_label_test
python3 -m src.validation.center_crop_test

# Phase 3: Full validation
python3 -m src.validation.proper_clustering
python3 -m src.validation.external_validation

# Run all experiments
python3 -m src.validation.run_all_experiments

# Run only Phase 1 (quick mode)
python3 -m src.validation.run_all_experiments --quick
```

"""
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate validation report")
    parser.add_argument("--results", "-r", type=str, default=None,
                       help="Path to results JSON")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output path for report")
    
    args = parser.parse_args()
    generate_report(args.results, args.output)
