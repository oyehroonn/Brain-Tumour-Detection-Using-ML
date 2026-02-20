"""
JSON report rendering for OncoSense.
Assembles and formats the final JSON report.
"""

import json
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path


def render_json_report(
    prediction: Dict,
    evidence: Dict,
    stability_result: Dict,
    coverage_result: Dict,
    llm_report: Dict,
    image_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict:
    """
    Render complete JSON report with all metadata.
    
    Args:
        prediction: Model prediction dictionary.
        evidence: Retrieved evidence.
        stability_result: Saliency stability result.
        coverage_result: Coverage analysis result.
        llm_report: Generated LLM report.
        image_path: Path to input image.
        output_path: Optional path to save report.
        
    Returns:
        Complete report dictionary.
    """
    report = {
        "metadata": {
            "version": "1.0.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "system": "OncoSense",
            "disclaimer": "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE"
        },
        "input": {
            "image_path": image_path
        },
        "prediction": {
            "predicted_class": prediction.get("predicted_class"),
            "predicted_label": prediction.get("predicted_label"),
            "confidence": prediction.get("confidence"),
            "probabilities": prediction.get("probabilities"),
            "abstained": prediction.get("abstain", False)
        },
        "uncertainty": prediction.get("uncertainty", {}),
        "fusion": {
            "weights": prediction.get("fusion_weights"),
            "per_model": prediction.get("per_model")
        } if "fusion_weights" in prediction else None,
        "explainability": {
            "stable": stability_result.get("show_explanation", False),
            "stability_score": stability_result.get("stability_score"),
            "reason": stability_result.get("reason")
        },
        "evidence_retrieval": {
            "coverage_score": coverage_result.get("coverage_score"),
            "covered_fields": coverage_result.get("covered_fields"),
            "uncovered_fields": coverage_result.get("uncovered_fields"),
            "meets_threshold": coverage_result.get("meets_threshold")
        },
        "gating": {
            "passed": llm_report.get("gating_passed", True),
            "reason": llm_report.get("gating_reason")
        },
        "report": {
            "summary": llm_report.get("summary"),
            "predicted_class_description": llm_report.get("predicted_class_description"),
            "typical_imaging_features": llm_report.get("typical_imaging_features"),
            "next_steps": llm_report.get("next_steps"),
            "limitations": llm_report.get("limitations"),
            "uncertainty_note": llm_report.get("uncertainty_note")
        },
        "evidence_used": llm_report.get("evidence_used", [])
    }
    
    # Remove None values
    report = {k: v for k, v in report.items() if v is not None}
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")
    
    return report


def format_report_summary(report: Dict) -> str:
    """
    Format report as human-readable summary.
    
    Args:
        report: Complete report dictionary.
        
    Returns:
        Formatted string summary.
    """
    pred = report.get("prediction", {})
    unc = report.get("uncertainty", {})
    exp = report.get("explainability", {})
    gating = report.get("gating", {})
    content = report.get("report", {})
    
    lines = [
        "=" * 60,
        "ONCOSENSE ANALYSIS REPORT",
        "=" * 60,
        f"Generated: {report.get('metadata', {}).get('generated_at', 'N/A')}",
        "",
        "PREDICTION",
        "-" * 40,
        f"Class: {pred.get('predicted_label', 'N/A')} (ID: {pred.get('predicted_class')})",
        f"Confidence: {pred.get('confidence', 0):.1%}",
        f"Abstained: {'Yes' if pred.get('abstained') else 'No'}",
        "",
        "UNCERTAINTY",
        "-" * 40,
        f"Entropy: {unc.get('entropy', 'N/A')}",
        "",
        "EXPLAINABILITY",
        "-" * 40,
        f"Explanation Stable: {'Yes' if exp.get('stable') else 'No'}",
        f"Stability Score: {exp.get('stability_score', 'N/A')}",
        "",
        "GATING",
        "-" * 40,
        f"Report Generated: {'Yes' if gating.get('passed') else 'No'}",
        f"Reason: {gating.get('reason', 'All conditions met')}",
        "",
        "SUMMARY",
        "-" * 40,
        content.get("summary", "N/A"),
        "",
        "LIMITATIONS",
        "-" * 40,
        content.get("limitations", "N/A"),
        "",
        "=" * 60,
        "DISCLAIMER: RESEARCH PROTOTYPE - NOT FOR CLINICAL USE",
        "=" * 60
    ]
    
    return "\n".join(lines)
