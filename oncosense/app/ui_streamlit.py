"""
OncoSense Streamlit Demo UI.
Upload an MRI image and get analysis with explanations.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import tempfile

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="OncoSense - Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a365d;
        text-align: center;
        padding: 1rem 0;
    }
    .disclaimer {
        background-color: #fed7d7;
        border-left: 4px solid #c53030;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .result-card {
        background-color: #f7fafc;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        color: white;
    }
    .abstain-warning {
        background-color: #fef3cd;
        border-left: 4px solid #856404;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_models():
    """Load models (cached)."""
    try:
        from src.models.infer import OncoSenseInference
        
        # Check if checkpoints exist
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.pt")):
            return None, "No trained models found. Please train models first."
        
        inferencer = OncoSenseInference(
            checkpoint_dir=str(checkpoint_dir),
            device="cuda" if st.session_state.get("use_gpu", True) else "cpu",
            use_ensemble=st.session_state.get("use_ensemble", True)
        )
        return inferencer, None
    except Exception as e:
        return None, str(e)


def load_knowledge_base():
    """Load knowledge base (cached)."""
    try:
        from src.rag.build_kb import KnowledgeBase
        
        kb_dir = Path("knowledge_base")
        if not kb_dir.exists():
            return None, "Knowledge base not built. Run: python -m src.rag.build_kb"
        
        kb = KnowledgeBase()
        kb.load(str(kb_dir / "chunks.json"), str(kb_dir / "faiss_index.bin"))
        return kb, None
    except Exception as e:
        return None, str(e)


def create_probability_chart(probs, class_names):
    """Create probability bar chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probs,
            marker_color=['#e53e3e', '#dd6b20', '#38a169', '#3182ce'],
            text=[f'{p:.1%}' for p in probs],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=300,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    
    return fig


def create_uncertainty_gauge(entropy, max_entropy=2.0):
    """Create uncertainty gauge chart."""
    normalized = min(entropy / max_entropy, 1.0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=entropy,
        title={'text': "Prediction Entropy"},
        gauge={
            'axis': {'range': [0, max_entropy]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1.0], 'color': "yellow"},
                {'range': [1.0, max_entropy], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.2
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(t=40, b=20, l=20, r=20))
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 OncoSense</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096;">Brain Tumor MRI Analysis Research System</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ RESEARCH PROTOTYPE - NOT FOR CLINICAL USE</strong><br>
        This system is intended for research and educational purposes only. 
        It is NOT a clinical diagnostic tool and should NOT be used for medical decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        st.session_state["use_gpu"] = st.checkbox("Use GPU (if available)", value=True)
        st.session_state["use_ensemble"] = st.checkbox("Use Ensemble", value=True)
        
        st.markdown("---")
        
        st.header("Model Info")
        st.markdown("""
        **Backbones:**
        - DenseNet121
        - Xception
        - EfficientNet-B0
        
        **Classes:**
        - Glioma
        - Meningioma
        - Pituitary Tumor
        - No Tumor
        """)
        
        st.markdown("---")
        
        st.header("About")
        st.markdown("""
        OncoSense uses:
        - Multi-model ensemble (EGU-Fusion++)
        - MC-dropout uncertainty
        - Temperature calibration
        - Grad-CAM with stability gating
        - RAG-powered evidence reports
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload MRI Image")
        
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a 2D brain MRI slice (axial, sagittal, or coronal)"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Store in session state
            st.session_state["uploaded_image"] = image
            st.session_state["image_array"] = np.array(image)
    
    with col2:
        st.header("Analysis Results")
        
        if "uploaded_image" not in st.session_state:
            st.info("👆 Upload an image to begin analysis")
        else:
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Running analysis..."):
                    run_analysis()


def run_analysis():
    """Run the full analysis pipeline."""
    image = st.session_state.get("uploaded_image")
    image_array = st.session_state.get("image_array")
    
    if image is None:
        st.error("No image uploaded")
        return
    
    # Placeholder for demo mode (when models aren't trained yet)
    demo_mode = True
    
    try:
        from src.models.infer import OncoSenseInference
        from src.rag.retrieve import EvidenceRetriever
        from src.rag.coverage import compute_coverage
        from src.rag.llm_generate import ReportGenerator
        from src.xai.stability import ExplanationGate
        
        # Check for models
        if not Path("checkpoints").exists() or not any(Path("checkpoints").glob("*.pt")):
            demo_mode = True
        else:
            demo_mode = False
    except ImportError:
        demo_mode = True
    
    if demo_mode:
        # Demo mode - show example results
        st.warning("Demo Mode: Models not trained yet. Showing example results.")
        show_demo_results()
    else:
        # Real inference
        run_real_inference(image)


def show_demo_results():
    """Show demo results when models aren't trained."""
    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    
    # Simulated results
    prediction = {
        "predicted_class": 0,
        "predicted_label": "glioma",
        "confidence": 0.847,
        "probabilities": [0.847, 0.089, 0.042, 0.022],
        "abstain": False,
        "uncertainty": {"entropy": 0.52}
    }
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>Predicted Class</h3>
            <h2>GLIOMA</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence", f"{prediction['confidence']:.1%}")
    
    with col3:
        st.metric("Entropy", f"{prediction['uncertainty']['entropy']:.3f}")
    
    st.markdown("---")
    
    # Probability chart
    st.plotly_chart(
        create_probability_chart(prediction['probabilities'], class_names),
        use_container_width=True
    )
    
    # Uncertainty gauge
    st.plotly_chart(
        create_uncertainty_gauge(prediction['uncertainty']['entropy']),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Report section
    st.header("Generated Report")
    
    with st.expander("📋 Full Report", expanded=True):
        st.markdown("""
        **Summary:**
        Analysis suggests glioma with high confidence (84.7%). The prediction shows 
        low uncertainty, and the explanation is stable.
        
        **Predicted Class Description:**
        Gliomas are the most common primary brain tumors arising from glial cells. 
        On MRI, they typically appear as irregular, infiltrative masses with 
        heterogeneous signal intensity. [Evidence ID: 1]
        
        **Typical Imaging Features:**
        Key imaging features include mass effect, irregular borders, and 
        heterogeneous enhancement. T2/FLAIR sequences show hyperintense signal. 
        [Evidence ID: 2]
        
        **Suggested Next Steps:**
        Clinical correlation with patient symptoms recommended. Consider comparison 
        with prior imaging if available. Multidisciplinary discussion advised. 
        [Evidence ID: 9]
        
        **Limitations:**
        This is a research prototype and NOT intended for clinical diagnosis. 
        All findings require expert radiological review.
        """)
    
    # Export options
    st.markdown("---")
    st.header("Export Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_json = json.dumps({
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }, indent=2)
        
        st.download_button(
            "📄 Download JSON Report",
            data=report_json,
            file_name="oncosense_report.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        st.button(
            "📑 Generate PDF Report",
            disabled=True,
            help="PDF export requires trained models",
            use_container_width=True
        )


def run_real_inference(image):
    """Run real inference with trained models."""
    from src.models.infer import OncoSenseInference
    from src.rag.retrieve import EvidenceRetriever
    from src.rag.coverage import compute_coverage
    from src.rag.llm_generate import ReportGenerator
    from src.xai.gradcam import GradCAMGenerator
    from src.xai.stability import SaliencyStabilityScorer, ExplanationGate
    from src.report.render_json import render_json_report
    from src.report.render_pdf import render_pdf_report
    
    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    
    # Initialize components
    inferencer = OncoSenseInference(
        checkpoint_dir="checkpoints",
        device="cuda" if st.session_state.get("use_gpu") else "cpu",
        use_ensemble=st.session_state.get("use_ensemble", True)
    )
    
    # Run prediction
    prediction = inferencer.predict(image, return_all=True)
    
    # Display prediction results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        label = prediction["predicted_label"].upper()
        st.markdown(f"""
        <div class="metric-box">
            <h3>Predicted Class</h3>
            <h2>{label}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence", f"{prediction['confidence']:.1%}")
    
    with col3:
        entropy = prediction.get("uncertainty", {}).get("entropy", 0)
        st.metric("Entropy", f"{entropy:.3f}")
    
    # Check abstention
    if prediction.get("abstain", False):
        st.markdown("""
        <div class="abstain-warning">
            <strong>⚠️ Model Abstained</strong><br>
            The model has low confidence in this prediction. 
            This case requires expert review.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Probability chart
    st.plotly_chart(
        create_probability_chart(prediction['probabilities'], class_names),
        use_container_width=True
    )
    
    # Run explainability and RAG
    with st.spinner("Generating explanation and report..."):
        # Retrieve evidence
        retriever = EvidenceRetriever()
        evidence = retriever.retrieve_for_prediction(
            prediction["predicted_label"],
            prediction["confidence"]
        )
        
        # Compute coverage
        coverage = compute_coverage(evidence)
        
        # Generate report
        stability_result = {"show_explanation": True, "stability_score": 0.8}
        
        generator = ReportGenerator(use_openai=False)
        report = generator.generate(prediction, evidence, stability_result, coverage)
    
    # Display report
    st.header("Generated Report")
    
    with st.expander("📋 Full Report", expanded=True):
        if report.get("gating_passed", True):
            st.markdown(f"**Summary:** {report.get('summary', 'N/A')}")
            st.markdown(f"**Description:** {report.get('predicted_class_description', 'N/A')}")
            st.markdown(f"**Imaging Features:** {report.get('typical_imaging_features', 'N/A')}")
            st.markdown(f"**Next Steps:** {report.get('next_steps', 'N/A')}")
            st.markdown(f"**Limitations:** {report.get('limitations', 'N/A')}")
        else:
            st.warning(f"Report suppressed: {report.get('gating_reason', 'Unknown reason')}")
    
    # Export options
    st.markdown("---")
    st.header("Export Report")
    
    full_report = render_json_report(
        prediction, evidence, stability_result, coverage, report
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📄 Download JSON Report",
            data=json.dumps(full_report, indent=2),
            file_name="oncosense_report.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        if st.button("📑 Generate PDF Report", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                pdf_path = render_pdf_report(full_report, f.name)
                
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_file.read(),
                        file_name="oncosense_report.pdf",
                        mime="application/pdf"
                    )


if __name__ == "__main__":
    main()
