"""
OncoSense — Neural Diagnostic Analysis System
Premium UI with minimalist, luxury aesthetic
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="OncoSense",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium minimal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');
    
    :root {
        --bg: #FAFAF8;
        --card: #FFFFFF;
        --text: #1A1A1A;
        --text-secondary: #6B6B6B;
        --text-muted: #9A9A9A;
        --border: #E5E5E5;
        --accent: #2D2D2D;
    }
    
    .stApp {
        background-color: var(--bg) !important;
    }
    
    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 1200px !important;
    }
    
    #MainMenu, footer, header, .stDeployButton {display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .premium-title {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .premium-title .subtitle {
        font-size: 0.65rem;
        font-weight: 400;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    
    .premium-title .main-title {
        font-size: 2rem;
        font-weight: 200;
        letter-spacing: -0.01em;
        color: var(--text);
        margin: 0;
    }
    
    .premium-title .tagline {
        font-size: 0.75rem;
        font-weight: 300;
        color: var(--text-muted);
        margin-top: 0.5rem;
    }
    
    .section-label {
        font-size: 0.6rem;
        font-weight: 500;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }
    
    .result-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 2rem;
        text-align: center;
    }
    
    .result-label {
        font-size: 0.55rem;
        font-weight: 500;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }
    
    .result-diagnosis {
        font-size: 1.75rem;
        font-weight: 300;
        letter-spacing: 0.02em;
        color: var(--text);
        margin-bottom: 0.5rem;
    }
    
    .result-description {
        font-size: 0.75rem;
        font-weight: 300;
        color: var(--text-secondary);
    }
    
    .metrics-row {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
    }
    
    .metric-item {
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.25rem;
        font-weight: 300;
        color: var(--text);
    }
    
    .metric-label {
        font-size: 0.55rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    .stButton > button {
        background: var(--text) !important;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 0.75rem 2rem !important;
        font-size: 0.65rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: var(--accent) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    .stFileUploader > div > div {
        border: 1px dashed var(--border) !important;
        border-radius: 4px !important;
        background: var(--card) !important;
    }
    
    .stFileUploader label {
        font-size: 0.65rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    
    .upload-prompt {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--card);
        border: 1px dashed var(--border);
        border-radius: 4px;
    }
    
    .upload-prompt p {
        color: var(--text-muted);
        font-size: 0.8rem;
        font-weight: 300;
    }
    
    .disclaimer-footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
        font-size: 0.65rem;
        font-weight: 300;
        color: var(--text-muted);
        letter-spacing: 0.02em;
    }
    
    .tech-box {
        background: #F5F5F3;
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1rem;
        font-family: 'SF Mono', 'Fira Code', monospace !important;
        font-size: 0.7rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    [data-testid="stImage"] {
        border: 1px solid var(--border);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .stExpander {
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        background: var(--card) !important;
    }
    
    .streamlit-expanderHeader {
        font-size: 0.65rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--text-secondary) !important;
    }
</style>
""", unsafe_allow_html=True)


def create_probability_chart(probs, class_names, predicted_idx):
    """Create minimal horizontal bar chart."""
    sorted_idx = np.argsort(probs)[::-1]
    
    colors = ['#1A1A1A' if i == predicted_idx else '#D8D8D8' for i in sorted_idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[class_names[i] for i in sorted_idx],
        x=[probs[i] for i in sorted_idx],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{probs[i]:.1%}' for i in sorted_idx],
        textposition='outside',
        textfont=dict(family='Inter', size=11, color='#6B6B6B'),
        hovertemplate='%{y}: %{x:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 1.2]
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(family='Inter', size=11, color='#6B6B6B')
        ),
        bargap=0.4,
        showlegend=False
    )
    
    return fig


def create_confidence_gauge(confidence):
    """Create minimal confidence gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 28, 'family': 'Inter', 'color': '#1A1A1A'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "white"},
            'bar': {'color': "#1A1A1A", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': '#E8E8E8'}
            ],
        }
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter')
    )
    
    return fig


@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    try:
        import torch
        from src.models.backbones import get_backbone
        from src.data.transforms import get_val_transforms
        
        checkpoint_paths = [
            Path("checkpoints/densenet121_fixed.pt"),
            Path("checkpoints/densenet121_best.pt")
        ]
        
        checkpoint_path = None
        for p in checkpoint_paths:
            if p.exists():
                checkpoint_path = p
                break
        
        if checkpoint_path is None:
            return None, None, "No model checkpoint found"
        
        model = get_backbone('densenet121', num_classes=4, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        transform = get_val_transforms()
        
        return model, transform, None
        
    except Exception as e:
        return None, None, str(e)


def run_inference(model, transform, image):
    """Run inference on an image."""
    import torch
    import torch.nn.functional as F
    
    img_array = np.array(image.convert('RGB'))
    tensor = transform(image=img_array)['image'].unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)[0].numpy()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    
    return {
        'pred_idx': pred_idx,
        'confidence': confidence,
        'probs': probs.tolist(),
        'entropy': entropy
    }


def main():
    CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    CLASS_DESC = {
        'Glioma': 'Primary brain tumor arising from glial cells',
        'Meningioma': 'Tumor arising from the meninges',
        'Pituitary': 'Tumor of the pituitary gland',
        'No Tumor': 'No detectable tumor present'
    }
    
    # Header
    st.markdown("""
        <div class="premium-title">
            <div class="subtitle">Neural Analysis System</div>
            <div class="main-title">OncoSense</div>
            <div class="tagline">Brain Tumor Classification · DenseNet-121</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, transform, error = load_model()
    
    if error:
        st.error(f"Model Error: {error}")
    
    # Two column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-label">MRI Input</div>', unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Upload MRI scan",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
            st.session_state['image'] = image
        else:
            st.markdown("""
                <div class="upload-prompt">
                    <p>Drop MRI scan here or click to browse</p>
                    <p style="font-size: 0.7rem; margin-top: 0.5rem;">Supports JPG, PNG</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
        
        if 'image' not in st.session_state:
            st.markdown("""
                <div class="upload-prompt">
                    <p>Upload an MRI scan to begin</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("Analyze", type="primary"):
                if model is not None:
                    with st.spinner("Analyzing..."):
                        result = run_inference(model, transform, st.session_state['image'])
                        st.session_state['result'] = result
                else:
                    st.error("Model not loaded")
            
            if 'result' in st.session_state:
                r = st.session_state['result']
                pred_name = CLASS_NAMES[r['pred_idx']]
                
                # Result card
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Classification Result</div>
                        <div class="result-diagnosis">{pred_name}</div>
                        <div class="result-description">{CLASS_DESC[pred_name]}</div>
                        <div class="metrics-row">
                            <div class="metric-item">
                                <div class="metric-value">{r['confidence']:.1%}</div>
                                <div class="metric-label">Confidence</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-value">{r['entropy']:.3f}</div>
                                <div class="metric-label">Entropy</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-value">{'High' if r['confidence'] > 0.85 else 'Medium' if r['confidence'] > 0.6 else 'Low'}</div>
                                <div class="metric-label">Certainty</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Spacer
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
                fig = create_probability_chart(r['probs'], CLASS_NAMES, r['pred_idx'])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Confidence gauge
                st.markdown('<div class="section-label">Confidence Level</div>', unsafe_allow_html=True)
                gauge = create_confidence_gauge(r['confidence'])
                st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
                
                # Technical details
                with st.expander("Technical Details"):
                    st.markdown(f"""
                        <div class="tech-box">
                            Model: DenseNet-121 (4-class classifier)<br>
                            Input: 224×224×3 RGB normalized<br>
                            Prediction: {pred_name} (class {r['pred_idx']})<br>
                            Confidence: {r['confidence']:.6f}<br>
                            Entropy: {r['entropy']:.6f}<br>
                            Probabilities: {', '.join([f'{CLASS_NAMES[i]}={r["probs"][i]:.3f}' for i in range(4)])}<br>
                            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </div>
                    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="disclaimer-footer">
            Research prototype · Not for clinical use · Validated on external dataset (91.4% accuracy)
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
