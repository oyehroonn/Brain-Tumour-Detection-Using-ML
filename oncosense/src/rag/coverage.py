"""
Evidence coverage metric for OncoSense RAG system.
Measures whether retrieved evidence adequately covers required report fields.
"""

from typing import Dict, List, Optional, Tuple
import yaml


def load_config(config_path: str = "configs/rag.yaml") -> dict:
    """Load RAG configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Required fields for a complete report
REQUIRED_FIELDS = [
    "predicted_class_description",
    "typical_imaging_features",
    "general_next_steps_suggestion",
    "uncertainty_note"
]


class CoverageCalculator:
    """
    Calculates evidence coverage for RAG gating.
    
    Coverage = (fields with ≥1 relevant chunk) / total_required_fields
    """
    
    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        config_path: str = "configs/rag.yaml"
    ):
        """
        Initialize coverage calculator.
        
        Args:
            required_fields: List of required field names.
            similarity_threshold: Minimum similarity for relevant chunk.
            config_path: Path to configuration.
        """
        try:
            config = load_config(config_path)
            coverage_config = config.get("coverage", {})
            self.required_fields = coverage_config.get(
                "required_fields", 
                required_fields or REQUIRED_FIELDS
            )
            self.similarity_threshold = coverage_config.get(
                "similarity_threshold",
                similarity_threshold
            )
            self.min_coverage = coverage_config.get("min_coverage", 0.75)
        except FileNotFoundError:
            self.required_fields = required_fields or REQUIRED_FIELDS
            self.similarity_threshold = similarity_threshold
            self.min_coverage = 0.75
    
    def compute_coverage(
        self,
        retrieved_evidence: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Compute coverage metric from retrieved evidence.
        
        Args:
            retrieved_evidence: Dict mapping categories to evidence lists.
            
        Returns:
            Dictionary with coverage metrics.
        """
        covered_fields = []
        uncovered_fields = []
        field_details = {}
        
        for field in self.required_fields:
            evidence_list = retrieved_evidence.get(field, [])
            
            # Check if any evidence meets threshold
            relevant_evidence = [
                e for e in evidence_list
                if e.get("similarity", 0) >= self.similarity_threshold
            ]
            
            if relevant_evidence:
                covered_fields.append(field)
                field_details[field] = {
                    "covered": True,
                    "num_evidence": len(relevant_evidence),
                    "max_similarity": max(e["similarity"] for e in relevant_evidence),
                    "evidence_ids": [e["evidence_id"] for e in relevant_evidence]
                }
            else:
                uncovered_fields.append(field)
                field_details[field] = {
                    "covered": False,
                    "num_evidence": len(evidence_list),
                    "max_similarity": max((e.get("similarity", 0) for e in evidence_list), default=0),
                    "reason": "No evidence above similarity threshold" if evidence_list else "No evidence retrieved"
                }
        
        coverage_score = len(covered_fields) / len(self.required_fields)
        
        return {
            "coverage_score": coverage_score,
            "covered_fields": covered_fields,
            "uncovered_fields": uncovered_fields,
            "field_details": field_details,
            "meets_threshold": coverage_score >= self.min_coverage,
            "threshold": self.min_coverage
        }
    
    def get_coverage_summary(
        self,
        coverage_result: Dict
    ) -> str:
        """
        Get human-readable coverage summary.
        
        Args:
            coverage_result: Result from compute_coverage.
            
        Returns:
            Summary string.
        """
        score = coverage_result["coverage_score"]
        covered = len(coverage_result["covered_fields"])
        total = len(self.required_fields)
        
        status = "PASS" if coverage_result["meets_threshold"] else "FAIL"
        
        summary = f"Coverage: {score:.1%} ({covered}/{total} fields) - {status}\n"
        
        if coverage_result["uncovered_fields"]:
            summary += f"Missing: {', '.join(coverage_result['uncovered_fields'])}"
        
        return summary


def compute_coverage(
    retrieved_evidence: Dict[str, List[Dict]],
    config_path: str = "configs/rag.yaml"
) -> Dict:
    """
    Convenience function to compute coverage.
    
    Args:
        retrieved_evidence: Retrieved evidence by category.
        config_path: Path to configuration.
        
    Returns:
        Coverage metrics dictionary.
    """
    calculator = CoverageCalculator(config_path=config_path)
    return calculator.compute_coverage(retrieved_evidence)


class ReportGate:
    """
    Gating logic for LLM report generation.
    
    Combines abstention, coverage, and stability checks.
    """
    
    def __init__(
        self,
        min_coverage: float = 0.75,
        min_stability: float = 0.7,
        config_path: str = "configs/rag.yaml"
    ):
        """
        Initialize report gate.
        
        Args:
            min_coverage: Minimum coverage score.
            min_stability: Minimum saliency stability score.
            config_path: Path to configuration.
        """
        try:
            config = load_config(config_path)
            gating_config = config.get("gating", {})
            self.require_not_abstained = gating_config.get("require_not_abstained", True)
            self.min_coverage = gating_config.get("min_coverage", min_coverage)
            self.min_stability = gating_config.get("min_saliency_stability", min_stability)
            self.safe_template = gating_config.get("safe_template", {})
        except FileNotFoundError:
            self.require_not_abstained = True
            self.min_coverage = min_coverage
            self.min_stability = min_stability
            self.safe_template = {}
    
    def check_gate(
        self,
        abstained: bool,
        coverage_score: float,
        stability_score: float
    ) -> Tuple[bool, str]:
        """
        Check if all gating conditions are met.
        
        Args:
            abstained: Whether model abstained from prediction.
            coverage_score: Evidence coverage score.
            stability_score: Saliency stability score.
            
        Returns:
            Tuple of (passed, reason).
        """
        reasons = []
        
        # Check abstention
        if self.require_not_abstained and abstained:
            reasons.append("model abstained from prediction")
        
        # Check coverage
        if coverage_score < self.min_coverage:
            reasons.append(
                f"insufficient evidence coverage ({coverage_score:.1%} < {self.min_coverage:.1%})"
            )
        
        # Check stability
        if stability_score < self.min_stability:
            reasons.append(
                f"explanation unstable ({stability_score:.2f} < {self.min_stability})"
            )
        
        if reasons:
            return False, "; ".join(reasons)
        
        return True, "all conditions met"
    
    def get_safe_response(self, reason: str) -> Dict:
        """
        Get safe template response when gating fails.
        
        Args:
            reason: Reason for gating failure.
            
        Returns:
            Safe template dictionary.
        """
        return {
            "summary": self.safe_template.get(
                "summary",
                "Analysis could not be completed with sufficient confidence."
            ),
            "model_findings": {
                "note": reason
            },
            "localization": {
                "stable": False,
                "description": "Explanation suppressed due to gating conditions."
            },
            "evidence_used": [],
            "next_steps": "This case requires manual expert review.",
            "limitations": self.safe_template.get(
                "limitations",
                "Gating conditions not met. Output suppressed for safety."
            ),
            "gating_failed": True,
            "gating_reason": reason
        }


if __name__ == "__main__":
    # Test coverage calculation
    test_evidence = {
        "predicted_class_description": [
            {"evidence_id": 1, "similarity": 0.8, "text": "Glioma description..."},
            {"evidence_id": 2, "similarity": 0.6, "text": "More about glioma..."}
        ],
        "typical_imaging_features": [
            {"evidence_id": 3, "similarity": 0.7, "text": "Imaging features..."}
        ],
        "general_next_steps_suggestion": [
            {"evidence_id": 4, "similarity": 0.4, "text": "Below threshold..."}
        ],
        "uncertainty_note": [
            {"evidence_id": 5, "similarity": 0.65, "text": "Uncertainty..."}
        ]
    }
    
    calculator = CoverageCalculator(similarity_threshold=0.5)
    result = calculator.compute_coverage(test_evidence)
    
    print(calculator.get_coverage_summary(result))
    print(f"\nDetailed results:")
    for field, details in result["field_details"].items():
        status = "✓" if details["covered"] else "✗"
        print(f"  {status} {field}: sim={details['max_similarity']:.2f}")
