"""
PDF report rendering for OncoSense.
Generates professional PDF reports from analysis results.
"""

from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import numpy as np
from PIL import Image


def create_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d')
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2c5282')
    ))
    
    styles.add(ParagraphStyle(
        name='BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    ))
    
    styles.add(ParagraphStyle(
        name='Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#c53030'),
        alignment=TA_CENTER,
        spaceBefore=20
    ))
    
    styles.add(ParagraphStyle(
        name='Citation',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        leftIndent=20
    ))
    
    return styles


def render_pdf_report(
    report: Dict,
    output_path: str,
    heatmap_array: Optional[np.ndarray] = None,
    original_image: Optional[np.ndarray] = None,
    overlay_image: Optional[np.ndarray] = None
) -> str:
    """
    Render PDF report from analysis results.
    
    Args:
        report: Complete report dictionary.
        output_path: Path to save PDF.
        heatmap_array: Optional Grad-CAM heatmap array.
        original_image: Optional original image array.
        overlay_image: Optional overlay visualization.
        
    Returns:
        Path to generated PDF.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = create_styles()
    story = []
    
    # Title
    story.append(Paragraph("OncoSense Analysis Report", styles['CustomTitle']))
    
    # Disclaimer
    story.append(Paragraph(
        "<b>RESEARCH PROTOTYPE - NOT FOR CLINICAL DIAGNOSIS</b>",
        styles['Disclaimer']
    ))
    story.append(Spacer(1, 20))
    
    # Metadata
    metadata = report.get("metadata", {})
    story.append(Paragraph(
        f"Generated: {metadata.get('generated_at', 'N/A')}",
        styles['BodyText']
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
    story.append(Spacer(1, 20))
    
    # Prediction Section
    story.append(Paragraph("Model Prediction", styles['SectionHeader']))
    
    pred = report.get("prediction", {})
    pred_data = [
        ["Predicted Class", f"{pred.get('predicted_label', 'N/A')}"],
        ["Confidence", f"{pred.get('confidence', 0):.1%}"],
        ["Abstained", "Yes" if pred.get("abstained") else "No"]
    ]
    
    pred_table = Table(pred_data, colWidths=[2*inch, 3*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray)
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 15))
    
    # Probabilities
    if pred.get("probabilities"):
        story.append(Paragraph("Class Probabilities:", styles['BodyText']))
        class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        prob_data = [[name, f"{prob:.1%}"] 
                     for name, prob in zip(class_names, pred["probabilities"])]
        
        prob_table = Table(prob_data, colWidths=[2*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(prob_table)
    
    story.append(Spacer(1, 20))
    
    # Visualization Section (if images provided)
    if original_image is not None or overlay_image is not None:
        story.append(Paragraph("Visualization", styles['SectionHeader']))
        
        images_to_add = []
        
        if original_image is not None:
            img_buffer = io.BytesIO()
            Image.fromarray(original_image).save(img_buffer, format='PNG')
            img_buffer.seek(0)
            images_to_add.append(("Original Image", img_buffer))
        
        if overlay_image is not None:
            overlay_buffer = io.BytesIO()
            Image.fromarray(overlay_image).save(overlay_buffer, format='PNG')
            overlay_buffer.seek(0)
            images_to_add.append(("Grad-CAM Overlay", overlay_buffer))
        
        for title, buffer in images_to_add:
            story.append(Paragraph(title, styles['BodyText']))
            story.append(RLImage(buffer, width=3*inch, height=3*inch))
            story.append(Spacer(1, 10))
    
    # Explainability Section
    story.append(Paragraph("Explainability Analysis", styles['SectionHeader']))
    
    exp = report.get("explainability", {})
    exp_text = f"""
    <b>Explanation Stability:</b> {'Stable' if exp.get('stable') else 'Unstable'}<br/>
    <b>Stability Score:</b> {exp.get('stability_score', 'N/A')}<br/>
    <b>Status:</b> {exp.get('reason', 'N/A')}
    """
    story.append(Paragraph(exp_text, styles['BodyText']))
    story.append(Spacer(1, 15))
    
    # Report Content Section
    content = report.get("report", {})
    
    if content.get("summary"):
        story.append(Paragraph("Summary", styles['SectionHeader']))
        story.append(Paragraph(content["summary"], styles['BodyText']))
        story.append(Spacer(1, 10))
    
    if content.get("predicted_class_description"):
        story.append(Paragraph("Predicted Class Description", styles['SectionHeader']))
        story.append(Paragraph(content["predicted_class_description"], styles['BodyText']))
        story.append(Spacer(1, 10))
    
    if content.get("typical_imaging_features"):
        story.append(Paragraph("Typical Imaging Features", styles['SectionHeader']))
        story.append(Paragraph(content["typical_imaging_features"], styles['BodyText']))
        story.append(Spacer(1, 10))
    
    if content.get("next_steps"):
        story.append(Paragraph("Suggested Next Steps", styles['SectionHeader']))
        story.append(Paragraph(content["next_steps"], styles['BodyText']))
        story.append(Spacer(1, 10))
    
    # Evidence Section
    evidence_used = report.get("evidence_used", [])
    if evidence_used:
        story.append(Paragraph("Evidence References", styles['SectionHeader']))
        for i, ev in enumerate(evidence_used):
            ev_text = f"[{ev.get('evidence_id', i+1)}] {ev.get('title', 'Untitled')}: {ev.get('quote_span', '')[:200]}..."
            story.append(Paragraph(ev_text, styles['Citation']))
        story.append(Spacer(1, 15))
    
    # Limitations Section
    story.append(Paragraph("Limitations", styles['SectionHeader']))
    story.append(Paragraph(
        content.get("limitations", "This is a research prototype and is not intended for clinical use."),
        styles['BodyText']
    ))
    
    # Final Disclaimer
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#c53030')))
    story.append(Paragraph(
        "<b>IMPORTANT:</b> This report is generated by a research prototype system. "
        "It is NOT a clinical diagnostic tool and should NOT be used for medical decision-making. "
        "All findings require review by qualified healthcare professionals.",
        styles['Disclaimer']
    ))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Test PDF generation
    test_report = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0"
        },
        "prediction": {
            "predicted_class": 0,
            "predicted_label": "glioma",
            "confidence": 0.85,
            "abstained": False,
            "probabilities": [0.85, 0.08, 0.05, 0.02]
        },
        "explainability": {
            "stable": True,
            "stability_score": 0.82,
            "reason": "stable"
        },
        "report": {
            "summary": "Analysis suggests glioma with high confidence.",
            "predicted_class_description": "Gliomas are primary brain tumors...",
            "typical_imaging_features": "Irregular masses with heterogeneous signal...",
            "next_steps": "Clinical correlation recommended...",
            "limitations": "Research prototype - not for clinical use."
        },
        "evidence_used": [
            {"evidence_id": 1, "title": "Glioma Description", "quote_span": "Gliomas are..."}
        ]
    }
    
    render_pdf_report(test_report, "test_report.pdf")
