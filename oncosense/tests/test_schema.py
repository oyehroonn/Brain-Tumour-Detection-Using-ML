"""
Tests for report schema validation.
"""

import pytest

from src.rag.schema import (
    validate_report,
    validate_citations,
    compute_citation_coverage,
    OncoSenseReport,
    ModelFindings,
    Localization,
    EvidenceReference
)


@pytest.fixture
def valid_report():
    """Create a valid report dictionary."""
    return {
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


def test_validate_report_valid(valid_report):
    """Test validation of valid report."""
    is_valid, errors = validate_report(valid_report)
    
    assert is_valid
    assert len(errors) == 0


def test_validate_report_missing_field(valid_report):
    """Test validation catches missing fields."""
    del valid_report["summary"]
    
    is_valid, errors = validate_report(valid_report)
    
    assert not is_valid
    assert any("summary" in e for e in errors)


def test_validate_report_missing_model_findings_field(valid_report):
    """Test validation catches missing model_findings fields."""
    del valid_report["model_findings"]["confidence"]
    
    is_valid, errors = validate_report(valid_report)
    
    assert not is_valid
    assert any("confidence" in e for e in errors)


def test_validate_report_invalid_evidence(valid_report):
    """Test validation catches invalid evidence structure."""
    valid_report["evidence_used"] = [
        {"evidence_id": 1}  # Missing quote_span
    ]
    
    is_valid, errors = validate_report(valid_report)
    
    assert not is_valid


def test_validate_citations(valid_report):
    """Test citation validation."""
    available_ids = [1, 2, 9, 12]
    
    is_valid, errors = validate_citations(valid_report, available_ids)
    
    assert is_valid
    assert len(errors) == 0


def test_validate_citations_invalid(valid_report):
    """Test citation validation catches invalid IDs."""
    available_ids = [1]  # Only ID 1 available
    
    is_valid, errors = validate_citations(valid_report, available_ids)
    
    assert not is_valid
    assert len(errors) > 0


def test_compute_citation_coverage(valid_report):
    """Test citation coverage computation."""
    coverage = compute_citation_coverage(valid_report)
    
    assert "total_sentences_estimate" in coverage
    assert "total_citations" in coverage
    assert "citation_coverage" in coverage
    assert coverage["total_citations"] == 2


def test_oncosense_report_dataclass():
    """Test OncoSenseReport dataclass."""
    model_findings = ModelFindings(
        predicted_class="0",
        predicted_label="glioma",
        confidence=0.85,
        uncertainty_entropy=0.4,
        abstained=False
    )
    
    localization = Localization(
        stable=True,
        description="Test description"
    )
    
    evidence = [EvidenceReference(evidence_id=1, quote_span="Test")]
    
    report = OncoSenseReport(
        summary="Test summary",
        model_findings=model_findings,
        localization=localization,
        evidence_used=evidence,
        predicted_class_description="Description",
        typical_imaging_features="Features",
        next_steps="Next steps",
        limitations="Limitations"
    )
    
    # Test to_dict
    report_dict = report.to_dict()
    assert report_dict["summary"] == "Test summary"
    assert report_dict["model_findings"]["confidence"] == 0.85
    
    # Test to_json
    json_str = report.to_json()
    assert "glioma" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
