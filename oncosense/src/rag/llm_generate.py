"""
LLM-based report generation for OncoSense.
Generates structured, citation-backed reports using retrieved evidence.
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from .schema import (
    OncoSenseReport, ModelFindings, Localization, EvidenceReference,
    validate_report, validate_citations
)
from .coverage import ReportGate


def load_config(config_path: str = "configs/rag.yaml") -> dict:
    """Load RAG configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# System prompt for structured report generation
SYSTEM_PROMPT = """You are a medical imaging analysis assistant generating structured research reports for brain MRI analysis.

CRITICAL RULES:
1. Output ONLY valid JSON matching the required schema - no markdown, no explanations outside JSON
2. Every factual statement MUST cite evidence using [evidence_id] format
3. NEVER make diagnostic claims or treatment recommendations
4. If evidence is insufficient for any field, state "Evidence insufficient for this field"
5. Always include the limitations section emphasizing this is a research prototype

You will receive:
- Model prediction results (class, confidence, uncertainty, abstention status)
- Retrieved evidence chunks with evidence_ids
- Localization stability information

Generate a report with these exact fields:
- summary: 1-2 sentence overview with citations
- model_findings: Object with prediction details
- localization: Object with stability and description
- evidence_used: Array of {evidence_id, quote_span, title}
- predicted_class_description: Description citing evidence
- typical_imaging_features: Features citing evidence  
- next_steps: Non-prescriptive suggestions citing evidence
- limitations: Always include research prototype disclaimer
- uncertainty_note: Explain confidence/uncertainty if relevant"""


class ReportGenerator:
    """
    LLM-based report generator with evidence grounding.
    """
    
    CLASS_NAMES = {
        0: "glioma",
        1: "meningioma", 
        2: "pituitary",
        3: "no_tumor"
    }
    
    def __init__(
        self,
        config_path: str = "configs/rag.yaml",
        use_openai: bool = True
    ):
        """
        Initialize report generator.
        
        Args:
            config_path: Path to RAG configuration.
            use_openai: Whether to use OpenAI API (vs template fallback).
        """
        try:
            self.config = load_config(config_path)
            self.llm_config = self.config.get("llm", {})
        except FileNotFoundError:
            self.config = {}
            self.llm_config = {}
        
        self.use_openai = use_openai and self._check_openai_available()
        self.gate = ReportGate(config_path=config_path)
    
    def _check_openai_available(self) -> bool:
        """Check if OpenAI API is available."""
        api_key = os.environ.get("OPENAI_API_KEY")
        return api_key is not None and len(api_key) > 0
    
    def _format_evidence_context(
        self,
        evidence: Dict[str, List[Dict]]
    ) -> str:
        """Format retrieved evidence for LLM context."""
        context_parts = []
        
        for category, chunks in evidence.items():
            if chunks:
                context_parts.append(f"\n## {category}")
                for chunk in chunks:
                    context_parts.append(
                        f"[{chunk['evidence_id']}] {chunk.get('title', 'Untitled')}: "
                        f"{chunk['text'][:500]}..."
                    )
        
        return "\n".join(context_parts)
    
    def _build_prompt(
        self,
        prediction: Dict,
        evidence: Dict[str, List[Dict]],
        stability_result: Dict
    ) -> str:
        """Build the user prompt for LLM."""
        evidence_context = self._format_evidence_context(evidence)
        
        prompt = f"""Generate a structured JSON report for this brain MRI analysis.

## Model Prediction
- Predicted Class: {prediction['predicted_class']} ({prediction['predicted_label']})
- Confidence: {prediction['confidence']:.1%}
- Entropy (uncertainty): {prediction.get('uncertainty', {}).get('entropy', 'N/A')}
- Abstained: {prediction.get('abstain', False)}

## Localization
- Explanation Stable: {stability_result.get('show_explanation', False)}
- Stability Score: {stability_result.get('stability_score', 'N/A')}

## Retrieved Evidence
{evidence_context}

Generate the complete JSON report with all required fields. Cite evidence using [evidence_id] format."""
        
        return prompt
    
    def generate_with_openai(
        self,
        prediction: Dict,
        evidence: Dict[str, List[Dict]],
        stability_result: Dict
    ) -> Dict:
        """Generate report using OpenAI API."""
        try:
            from openai import OpenAI
            
            client = OpenAI()
            
            user_prompt = self._build_prompt(prediction, evidence, stability_result)
            
            response = client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.llm_config.get("temperature", 0.3),
                max_tokens=self.llm_config.get("max_tokens", 1024),
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self.generate_template(prediction, evidence, stability_result)
    
    def generate_template(
        self,
        prediction: Dict,
        evidence: Dict[str, List[Dict]],
        stability_result: Dict
    ) -> Dict:
        """Generate report using template (fallback when LLM unavailable)."""
        pred_label = prediction["predicted_label"]
        confidence = prediction["confidence"]
        
        # Get best evidence for each category
        def get_best_evidence(category: str) -> tuple[str, List[Dict]]:
            chunks = evidence.get(category, [])
            if not chunks:
                return "Evidence not available.", []
            
            best = chunks[0]
            text = f"{best['text'][:300]} [{best['evidence_id']}]"
            refs = [{"evidence_id": best["evidence_id"], 
                     "quote_span": best["text"][:100],
                     "title": best.get("title")}]
            return text, refs
        
        desc, desc_refs = get_best_evidence("predicted_class_description")
        features, features_refs = get_best_evidence("typical_imaging_features")
        next_steps, next_refs = get_best_evidence("general_next_steps_suggestion")
        uncertainty, unc_refs = get_best_evidence("uncertainty_note")
        
        all_refs = desc_refs + features_refs + next_refs + unc_refs
        
        return {
            "summary": f"Analysis suggests {pred_label} with {confidence:.1%} confidence. "
                       f"See detailed findings below.",
            "model_findings": {
                "predicted_class": prediction["predicted_class"],
                "predicted_label": pred_label,
                "confidence": confidence,
                "uncertainty_entropy": prediction.get("uncertainty", {}).get("entropy", 0),
                "abstained": prediction.get("abstain", False)
            },
            "localization": {
                "stable": stability_result.get("show_explanation", False),
                "description": stability_result.get("reason", "Stability analysis performed."),
                "heatmap_available": stability_result.get("heatmap") is not None
            },
            "evidence_used": all_refs,
            "predicted_class_description": desc,
            "typical_imaging_features": features,
            "next_steps": next_steps,
            "limitations": "This is a research prototype and NOT intended for clinical diagnosis. "
                          "All findings require expert radiological review. [12]",
            "uncertainty_note": uncertainty,
            "gating_passed": True
        }
    
    def generate(
        self,
        prediction: Dict,
        evidence: Dict[str, List[Dict]],
        stability_result: Dict,
        coverage_result: Dict
    ) -> Dict:
        """
        Generate complete report with gating checks.
        
        Args:
            prediction: Model prediction dictionary.
            evidence: Retrieved evidence by category.
            stability_result: Saliency stability result.
            coverage_result: Evidence coverage result.
            
        Returns:
            Complete report dictionary.
        """
        # Check gating conditions
        passed, reason = self.gate.check_gate(
            abstained=prediction.get("abstain", False),
            coverage_score=coverage_result.get("coverage_score", 0),
            stability_score=stability_result.get("stability_score", 0)
        )
        
        if not passed:
            return self.gate.get_safe_response(reason)
        
        # Generate report
        if self.use_openai:
            report = self.generate_with_openai(prediction, evidence, stability_result)
        else:
            report = self.generate_template(prediction, evidence, stability_result)
        
        # Validate
        is_valid, errors = validate_report(report)
        if not is_valid:
            print(f"Report validation errors: {errors}")
        
        report["gating_passed"] = True
        return report


def generate_report(
    prediction: Dict,
    evidence: Dict[str, List[Dict]],
    stability_result: Dict,
    coverage_result: Dict,
    config_path: str = "configs/rag.yaml"
) -> Dict:
    """
    Convenience function to generate a report.
    
    Args:
        prediction: Model prediction.
        evidence: Retrieved evidence.
        stability_result: Stability analysis result.
        coverage_result: Coverage analysis result.
        config_path: Config path.
        
    Returns:
        Generated report dictionary.
    """
    generator = ReportGenerator(config_path)
    return generator.generate(prediction, evidence, stability_result, coverage_result)


if __name__ == "__main__":
    # Test report generation
    test_prediction = {
        "predicted_class": 0,
        "predicted_label": "glioma",
        "confidence": 0.85,
        "uncertainty": {"entropy": 0.4},
        "abstain": False
    }
    
    test_evidence = {
        "predicted_class_description": [
            {
                "evidence_id": 1,
                "title": "Glioma Description",
                "text": "Gliomas are the most common primary brain tumors..."
            }
        ],
        "typical_imaging_features": [
            {
                "evidence_id": 2,
                "title": "Glioma Features",
                "text": "On MRI, gliomas appear as irregular masses..."
            }
        ],
        "general_next_steps_suggestion": [
            {
                "evidence_id": 9,
                "title": "Next Steps",
                "text": "Clinical correlation recommended..."
            }
        ],
        "uncertainty_note": [
            {
                "evidence_id": 11,
                "title": "Uncertainty",
                "text": "Model predictions include uncertainty estimates..."
            }
        ]
    }
    
    test_stability = {
        "show_explanation": True,
        "stability_score": 0.8,
        "reason": "stable"
    }
    
    test_coverage = {
        "coverage_score": 1.0,
        "meets_threshold": True
    }
    
    generator = ReportGenerator(use_openai=False)
    report = generator.generate(
        test_prediction, test_evidence, test_stability, test_coverage
    )
    
    print(json.dumps(report, indent=2))
