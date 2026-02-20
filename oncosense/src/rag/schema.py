"""
JSON schema validation for OncoSense reports.
Defines and validates the structured output format.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json


@dataclass
class ModelFindings:
    """Model prediction findings."""
    predicted_class: str
    predicted_label: str
    confidence: float
    uncertainty_entropy: float
    abstained: bool
    note: Optional[str] = None


@dataclass
class Localization:
    """Localization/explanation information."""
    stable: bool
    description: str
    heatmap_available: bool = False


@dataclass
class EvidenceReference:
    """Reference to evidence chunk."""
    evidence_id: int
    quote_span: str
    title: Optional[str] = None


@dataclass
class OncoSenseReport:
    """
    Complete OncoSense structured report.
    
    All fields must cite evidence from the knowledge base.
    """
    summary: str
    model_findings: ModelFindings
    localization: Localization
    evidence_used: List[EvidenceReference]
    predicted_class_description: str
    typical_imaging_features: str
    next_steps: str
    limitations: str
    
    # Optional fields
    uncertainty_note: Optional[str] = None
    gating_passed: bool = True
    gating_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "OncoSenseReport":
        """Create from dictionary."""
        # Handle nested dataclasses
        if isinstance(data.get("model_findings"), dict):
            data["model_findings"] = ModelFindings(**data["model_findings"])
        if isinstance(data.get("localization"), dict):
            data["localization"] = Localization(**data["localization"])
        if data.get("evidence_used"):
            data["evidence_used"] = [
                EvidenceReference(**e) if isinstance(e, dict) else e
                for e in data["evidence_used"]
            ]
        
        return cls(**data)


# JSON Schema for validation
REPORT_SCHEMA = {
    "type": "object",
    "required": [
        "summary",
        "model_findings",
        "localization",
        "evidence_used",
        "predicted_class_description",
        "typical_imaging_features",
        "next_steps",
        "limitations"
    ],
    "properties": {
        "summary": {
            "type": "string",
            "description": "Brief summary of the analysis (1-2 sentences)"
        },
        "model_findings": {
            "type": "object",
            "required": ["predicted_class", "predicted_label", "confidence", "abstained"],
            "properties": {
                "predicted_class": {"type": "integer"},
                "predicted_label": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "uncertainty_entropy": {"type": "number"},
                "abstained": {"type": "boolean"},
                "note": {"type": "string"}
            }
        },
        "localization": {
            "type": "object",
            "required": ["stable", "description"],
            "properties": {
                "stable": {"type": "boolean"},
                "description": {"type": "string"},
                "heatmap_available": {"type": "boolean"}
            }
        },
        "evidence_used": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["evidence_id", "quote_span"],
                "properties": {
                    "evidence_id": {"type": "integer"},
                    "quote_span": {"type": "string"},
                    "title": {"type": "string"}
                }
            }
        },
        "predicted_class_description": {"type": "string"},
        "typical_imaging_features": {"type": "string"},
        "next_steps": {"type": "string"},
        "limitations": {"type": "string"},
        "uncertainty_note": {"type": "string"},
        "gating_passed": {"type": "boolean"},
        "gating_reason": {"type": "string"}
    }
}


def validate_report(report: Dict) -> tuple[bool, List[str]]:
    """
    Validate a report against the schema.
    
    Args:
        report: Report dictionary to validate.
        
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []
    
    # Check required fields
    for field in REPORT_SCHEMA["required"]:
        if field not in report:
            errors.append(f"Missing required field: {field}")
    
    # Check model_findings
    if "model_findings" in report:
        mf = report["model_findings"]
        if not isinstance(mf, dict):
            errors.append("model_findings must be an object")
        else:
            for req in ["predicted_class", "predicted_label", "confidence", "abstained"]:
                if req not in mf:
                    errors.append(f"model_findings missing: {req}")
    
    # Check evidence_used
    if "evidence_used" in report:
        if not isinstance(report["evidence_used"], list):
            errors.append("evidence_used must be an array")
        else:
            for i, ev in enumerate(report["evidence_used"]):
                if not isinstance(ev, dict):
                    errors.append(f"evidence_used[{i}] must be an object")
                elif "evidence_id" not in ev or "quote_span" not in ev:
                    errors.append(f"evidence_used[{i}] missing required fields")
    
    # Check localization
    if "localization" in report:
        loc = report["localization"]
        if not isinstance(loc, dict):
            errors.append("localization must be an object")
        elif "stable" not in loc or "description" not in loc:
            errors.append("localization missing required fields")
    
    return len(errors) == 0, errors


def validate_citations(
    report: Dict,
    available_evidence_ids: List[int]
) -> tuple[bool, List[str]]:
    """
    Validate that all citations reference valid evidence.
    
    Args:
        report: Report dictionary.
        available_evidence_ids: List of valid evidence IDs.
        
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []
    evidence_ids_set = set(available_evidence_ids)
    
    if "evidence_used" in report:
        for ev in report["evidence_used"]:
            if isinstance(ev, dict) and "evidence_id" in ev:
                if ev["evidence_id"] not in evidence_ids_set:
                    errors.append(
                        f"Invalid evidence_id: {ev['evidence_id']}"
                    )
    
    return len(errors) == 0, errors


def compute_citation_coverage(
    report: Dict,
    text_fields: List[str] = ["summary", "predicted_class_description", 
                              "typical_imaging_features", "next_steps"]
) -> Dict:
    """
    Compute citation coverage statistics.
    
    Args:
        report: Report dictionary.
        text_fields: Fields that should have citations.
        
    Returns:
        Dictionary with coverage statistics.
    """
    evidence_used = report.get("evidence_used", [])
    evidence_ids = {ev["evidence_id"] for ev in evidence_used if isinstance(ev, dict)}
    
    # Count sentences vs citations (rough estimate)
    total_sentences = 0
    cited_mentions = len(evidence_used)
    
    for field in text_fields:
        if field in report and isinstance(report[field], str):
            # Rough sentence count
            text = report[field]
            sentences = text.count(". ") + text.count("! ") + text.count("? ") + 1
            total_sentences += sentences
    
    coverage = cited_mentions / max(total_sentences, 1)
    
    return {
        "total_sentences_estimate": total_sentences,
        "total_citations": cited_mentions,
        "unique_evidence_ids": len(evidence_ids),
        "citation_coverage": min(coverage, 1.0),
        "meets_target": coverage >= 0.95
    }


if __name__ == "__main__":
    # Test schema validation
    test_report = {
        "summary": "Analysis shows likely glioma [1].",
        "model_findings": {
            "predicted_class": 0,
            "predicted_label": "glioma",
            "confidence": 0.85,
            "uncertainty_entropy": 0.4,
            "abstained": False
        },
        "localization": {
            "stable": True,
            "description": "Highlighted region in frontal lobe.",
            "heatmap_available": True
        },
        "evidence_used": [
            {"evidence_id": 1, "quote_span": "Gliomas are the most common..."},
            {"evidence_id": 2, "quote_span": "Key imaging features include..."}
        ],
        "predicted_class_description": "Glioma description [1].",
        "typical_imaging_features": "Features described [2].",
        "next_steps": "Correlation recommended [9].",
        "limitations": "Research prototype only [12]."
    }
    
    is_valid, errors = validate_report(test_report)
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test citation coverage
    coverage = compute_citation_coverage(test_report)
    print(f"\nCitation coverage: {coverage}")
