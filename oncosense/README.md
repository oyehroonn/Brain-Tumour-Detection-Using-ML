# OncoSense

**Research-Grade MRI Brain Tumor Classification and Decision Support System**

> **DISCLAIMER**: This is a research prototype, NOT a clinical diagnostic product. It is intended for research and educational purposes only. Do not use for clinical diagnosis, treatment planning, or any medical decision-making. Always consult qualified healthcare professionals for medical advice.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Key Features](#key-features)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Data Pipeline](#data-pipeline)
7. [Model Training](#model-training)
8. [Inference Pipeline](#inference-pipeline)
9. [Configuration](#configuration)
10. [Running the System](#running-the-system)
11. [Technical Deep Dive](#technical-deep-dive)
12. [API Reference](#api-reference)
13. [Troubleshooting](#troubleshooting)

---

## Overview

OncoSense is a comprehensive brain tumor classification system that goes beyond simple image classification. It implements:

- **Multi-model ensemble** with uncertainty-weighted fusion (EGU-Fusion++)
- **Calibrated predictions** using temperature scaling
- **Selective prediction** (abstention on uncertain cases)
- **Explainable AI** with Grad-CAM and stability gating
- **RAG-powered reports** with evidence citations

The system is designed for 4-class classification:
| Class | Label | Description |
|-------|-------|-------------|
| 0 | Glioma | Primary brain tumors from glial cells |
| 1 | Meningioma | Benign tumors from meninges |
| 2 | Pituitary | Pituitary gland adenomas |
| 3 | No Tumor | Normal brain MRI |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OncoSense Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │  MRI Image  │───▶│               PREPROCESSING                       │   │
│  └─────────────┘    │  • Resize to 224×224                             │   │
│                     │  • Normalize (ImageNet stats)                     │   │
│                     │  • Convert grayscale → RGB                        │   │
│                     └──────────────────────────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        ENSEMBLE MODELS                                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │  │
│  │  │ DenseNet121  │  │   Xception   │  │   EfficientNet-B0        │   │  │
│  │  │ + MC-Dropout │  │ + MC-Dropout │  │   + MC-Dropout           │   │  │
│  │  │ + Temp Scale │  │ + Temp Scale │  │   + Temp Scale           │   │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘   │  │
│  │         │                 │                      │                   │  │
│  │         ▼                 ▼                      ▼                   │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │                 EGU-Fusion++ (Uncertainty-Weighted)          │   │  │
│  │  │                                                              │   │  │
│  │  │   w_m(x) = exp(-α·u_m(x)) / Σ_k exp(-α·u_k(x))              │   │  │
│  │  │   p(y|x) = Σ_m w_m(x) · p_m(y|x)                            │   │  │
│  │  │                                                              │   │  │
│  │  │   → Lower uncertainty = Higher weight in fusion              │   │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                    ┌───────────────────┼───────────────────┐               │
│                    ▼                   ▼                   ▼               │
│  ┌────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐    │
│  │   ABSTENTION       │  │   GRAD-CAM + SSG    │  │   RAG RETRIEVAL  │    │
│  │   DECISION         │  │                     │  │                  │    │
│  │                     │  │  • Generate heatmap │  │  • Query KB      │    │
│  │  if max_prob < 0.6  │  │  • K perturbations  │  │  • FAISS search  │    │
│  │  OR entropy > 1.2:  │  │  • Compute SSIM     │  │  • Get evidence  │    │
│  │    → ABSTAIN        │  │  • Gate if unstable │  │                  │    │
│  └────────────────────┘  └─────────────────────┘  └──────────────────┘    │
│                    │                   │                   │               │
│                    └───────────────────┼───────────────────┘               │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    REPORT GENERATION (LLM + Template)                 │  │
│  │                                                                       │  │
│  │  GATING CONDITIONS:                                                   │  │
│  │  ✓ Not abstained                                                      │  │
│  │  ✓ Evidence coverage ≥ 80%                                            │  │
│  │  ✓ Explanation stability ≥ 0.7                                        │  │
│  │                                                                       │  │
│  │  If all pass → Generate structured JSON report with citations         │  │
│  │  Otherwise   → Return safe fallback response                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│                           ┌──────────────────────┐                         │
│                           │   JSON/PDF REPORT    │                         │
│                           └──────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Multi-Model Ensemble
Three pretrained backbones fine-tuned on brain tumor MRI:
- **DenseNet121**: Dense connections for feature reuse
- **Xception**: Depthwise separable convolutions
- **EfficientNet-B0**: Compound scaling efficiency

### 2. Uncertainty Quantification
- **MC-Dropout**: Monte Carlo dropout (20 forward passes) for epistemic uncertainty
- **Predictive Entropy**: `H(p) = -Σ p·log(p)` measures prediction uncertainty
- **Mutual Information**: Epistemic uncertainty from MC samples

### 3. Model Calibration
- **Temperature Scaling**: Learned scalar `T` for calibrated probabilities
- **Metrics**: ECE (Expected Calibration Error), Brier Score, NLL

### 4. EGU-Fusion++ (Ensemble Fusion)
Uncertainty-weighted voting where confident models have higher influence:
```
w_m(x) = exp(-α·u_m(x)) / Σ_k exp(-α·u_k(x))
```

### 5. Selective Prediction (Abstention)
The system refuses to predict when uncertain:
- Max probability < 0.6 **OR** Entropy > 1.2 → **Abstain**

### 6. Explainability with Stability Gating
- **Grad-CAM**: Visual explanation highlighting important regions
- **Saliency Stability Scoring (SSG)**: Only show explanations if stable under perturbations

### 7. RAG-Powered Reports
- **Knowledge Base**: 20 curated medical text chunks
- **FAISS Retrieval**: Semantic search for relevant evidence
- **Citation-Backed Output**: Every claim references evidence

---

## Project Structure

```
oncosense/
├── configs/                     # YAML configuration files
│   ├── data.yaml               # Dataset and preprocessing settings
│   ├── train.yaml              # Training hyperparameters
│   ├── models.yaml             # Model architecture settings
│   ├── fusion.yaml             # EGU-Fusion++ and abstention settings
│   └── rag.yaml                # RAG and LLM settings
│
├── data/                        # Data storage (gitignored)
│   ├── raw/                    # Downloaded images by class
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── no_tumor/
│   ├── processed/              # Preprocessed data
│   └── manifests/              # Parquet metadata files
│       ├── manifest_raw.parquet
│       ├── manifest_dedup.parquet
│       └── manifest.parquet    # Final with splits
│
├── src/                         # Source code
│   ├── __init__.py
│   │
│   ├── data/                   # Data pipeline
│   │   ├── __init__.py
│   │   ├── download_kaggle.py  # Kaggle dataset download
│   │   ├── dedup.py            # SHA256 + pHash deduplication
│   │   ├── split.py            # Cluster-aware stratified splits
│   │   ├── transforms.py       # Albumentations + DataLoader
│   │   └── corruptions.py      # Robustness evaluation corruptions
│   │
│   ├── models/                 # Model training and inference
│   │   ├── __init__.py
│   │   ├── backbones.py        # DenseNet/Xception/EfficientNet wrappers
│   │   ├── uncertainty.py      # MC-dropout and entropy computation
│   │   ├── calibrate.py        # Temperature scaling calibration
│   │   ├── fusion.py           # EGU-Fusion++ ensemble
│   │   ├── train.py            # Training loop with focal loss
│   │   └── infer.py            # High-level inference API
│   │
│   ├── xai/                    # Explainability
│   │   ├── __init__.py
│   │   ├── gradcam.py          # Grad-CAM generation
│   │   └── stability.py        # Saliency Stability Scoring
│   │
│   ├── rag/                    # RAG system
│   │   ├── __init__.py
│   │   ├── schema.py           # Report JSON schema
│   │   ├── build_kb.py         # Knowledge base builder
│   │   ├── retrieve.py         # FAISS evidence retrieval
│   │   ├── coverage.py         # Evidence coverage computation
│   │   └── llm_generate.py     # LLM/template report generation
│   │
│   └── report/                 # Report rendering
│       ├── __init__.py
│       ├── render_json.py      # JSON report compilation
│       └── render_pdf.py       # PDF generation with ReportLab
│
├── app/                         # Demo UI
│   ├── __init__.py
│   └── ui_streamlit.py         # Streamlit web interface
│
├── knowledge_base/              # RAG knowledge store (after build)
│   ├── chunks.json             # Text chunks with metadata
│   └── faiss_index.bin         # Vector similarity index
│
├── checkpoints/                 # Trained model weights (after training)
│   ├── densenet121_best.pt
│   ├── xception_best.pt
│   └── efficientnet_b0_best.pt
│
├── scripts/                     # Utility scripts
│   ├── run_train.sh            # Full training pipeline
│   └── run_eval.sh             # Evaluation pipeline
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_dedup.py
│   ├── test_fusion.py
│   └── test_schema.py
│
├── quick_setup.py              # Fast data setup (skips clustering)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip or conda
- ~10GB disk space for data and models
- GPU recommended (CUDA-capable) for training

### Steps

```bash
# 1. Navigate to the oncosense directory
cd oncosense

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up OpenAI API for LLM reports
export OPENAI_API_KEY="your-api-key-here"
```

### Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import timm; print(f'Timm: {timm.__version__}')"
python3 -c "import faiss; print('FAISS: OK')"
```

---

## Data Pipeline

The data pipeline ensures clean, deduplicated, and properly split data:

```
┌────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Kaggle/Local  │ ──▶ │  Deduplication   │ ──▶ │  Cluster-Aware    │
│  Download      │     │  SHA256 + pHash  │     │  Stratified Split │
└────────────────┘     └──────────────────┘     └───────────────────┘
        │                      │                        │
        ▼                      ▼                        ▼
   data/raw/           manifest_raw.parquet      manifest.parquet
   ├── glioma/         manifest_dedup.parquet    (with train/val/test)
   ├── meningioma/
   ├── pituitary/
   └── no_tumor/
```

### Step 1: Prepare Raw Data

**Option A**: If you have the Kaggle dataset locally:
```bash
# Copy your downloaded dataset to data/raw/ organized by class
mkdir -p data/raw/glioma data/raw/meningioma data/raw/pituitary data/raw/no_tumor
# Copy images from your download to appropriate folders
```

**Option B**: Download via Kaggle API:
```bash
python3 -m src.data.download_kaggle
```

### Step 2: Create Manifest and Deduplicate

**Quick Setup (Recommended)** - Fast, skips expensive clustering:
```bash
# First create raw manifest
python3 -m src.data.dedup --data-dir data/raw --create-manifest

# Then run quick setup
python3 quick_setup.py
```

**Full Setup** - Complete pHash clustering (slow):
```bash
python3 -m src.data.dedup --data-dir data/raw --create-manifest
python3 -m src.data.dedup
python3 -m src.data.split
```

### Data Manifest Schema

The manifest (`.parquet`) contains:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | SHA256 hash (unique identifier) |
| `filepath` | str | Relative path to image |
| `label` | int | Class index (0-3) |
| `label_name` | str | Class name (glioma, meningioma, etc.) |
| `phash` | str | Perceptual hash for near-duplicate detection |
| `cluster_id` | int | Cluster assignment (prevents leakage) |
| `split` | str | train/val/test assignment |
| `width`, `height` | int | Image dimensions |

### Split Ratios

- **Training**: 70% of clusters
- **Validation**: 10% of clusters  
- **Test**: 20% of clusters

> **Important**: Splits are done at the cluster level to prevent data leakage from near-duplicates.

---

## Model Training

### Training Architecture

```python
# Each backbone follows this structure:
BrainTumorClassifier(
    backbone=pretrained_model,      # DenseNet121/Xception/EfficientNet-B0
    mc_dropout=MCDropout(p=0.3),    # Always active during inference
    classifier=Linear(features, 4),  # 4-class output
    temperature=1.0                  # For calibration
)
```

### Training Pipeline

```bash
# Train all three models sequentially
python3 -m src.models.train --model densenet121
python3 -m src.models.train --model xception
python3 -m src.models.train --model efficientnet_b0
```

### Training Configuration (`configs/train.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 32 | Images per batch |
| `epochs` | 50 | Maximum epochs |
| `lr` | 1e-4 | Initial learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `loss` | Focal Loss | Handles class imbalance |
| `gamma` | 2.0 | Focal loss focusing parameter |
| `early_stopping` | 8 epochs | Patience for validation F1 |

### Loss Function: Focal Loss

```
FL(p_t) = -α_t(1 - p_t)^γ · log(p_t)
```

- Down-weights easy examples (high p_t)
- Focuses learning on hard examples
- `γ=2.0` is the default focusing parameter

### Training Output

After training, you'll find in `checkpoints/`:
- `{model_name}_best.pt` - Best model weights
- `{model_name}_history.json` - Training metrics

---

## Inference Pipeline

### Single Image Prediction Flow

```python
from src.models.infer import OncoSenseInference

# Initialize
inferencer = OncoSenseInference(
    checkpoint_dir="checkpoints",
    device="cuda",
    use_ensemble=True
)

# Predict
result = inferencer.predict(image, return_all=True)

# Result contains:
# - predicted_class: int (0-3)
# - predicted_label: str ("glioma", etc.)
# - confidence: float (0-1)
# - probabilities: list[float] (4 values)
# - uncertainty: dict (entropy, variance, mutual_info)
# - abstain: bool
```

### Ensemble Fusion Process

1. **Run MC-Dropout** on each model (20 passes)
2. **Compute per-model uncertainty** (entropy)
3. **Calculate fusion weights** (inverse uncertainty)
4. **Weighted average** of probabilities
5. **Abstention check** on fused output

### Full Analysis Pipeline

```python
from src.models.infer import OncoSenseInference
from src.xai.gradcam import GradCAMGenerator
from src.xai.stability import SaliencyStabilityScorer
from src.rag.retrieve import EvidenceRetriever
from src.rag.llm_generate import ReportGenerator

# 1. Get prediction
prediction = inferencer.predict(image)

# 2. Generate explanation (if not abstained)
if not prediction['abstain']:
    gradcam = GradCAMGenerator(model, backbone_name)
    heatmap = gradcam.generate(tensor)
    
    # 3. Check stability
    scorer = SaliencyStabilityScorer(gradcam)
    stability = scorer.compute_stability_score(tensor)
    
    # 4. Retrieve evidence
    retriever = EvidenceRetriever()
    evidence = retriever.retrieve_for_prediction(prediction['predicted_label'])
    
    # 5. Generate report
    generator = ReportGenerator()
    report = generator.generate(prediction, evidence, stability, coverage)
```

---

## Configuration

### `configs/data.yaml`
```yaml
dataset:
  class_names:
    0: "glioma"
    1: "meningioma"
    2: "pituitary"
    3: "no_tumor"

preprocessing:
  input_size: 224
  normalize_mean: [0.485, 0.456, 0.406]  # ImageNet
  normalize_std: [0.229, 0.224, 0.225]

splits:
  train_ratio: 0.70
  val_ratio: 0.10
  test_ratio: 0.20
```

### `configs/fusion.yaml`
```yaml
egu_fusion:
  alpha: 1.0                    # Uncertainty weighting strength

abstention:
  max_prob_threshold: 0.6       # Abstain if confidence below
  entropy_threshold: 1.2        # Abstain if entropy above
  mode: "or"                    # "or" = either condition triggers

calibration:
  method: "temperature_scaling"
  optimize_metric: "nll"
```

### `configs/rag.yaml`
```yaml
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

retrieval:
  top_k: 5
  similarity_threshold: 0.5

saliency_stability:
  num_perturbations: 5
  similarity_metric: "ssim"
  min_stability_score: 0.7

llm:
  model: "gpt-4o-mini"
  temperature: 0.3
```

---

## Running the System

### Complete Pipeline

```bash
# 1. Setup data (one-time)
python3 -m src.data.dedup --data-dir data/raw --create-manifest
python3 quick_setup.py

# 2. Train models
python3 -m src.models.train --model densenet121
python3 -m src.models.train --model xception
python3 -m src.models.train --model efficientnet_b0

# 3. Calibrate models
python3 -m src.models.calibrate

# 4. Build knowledge base
python3 -m src.rag.build_kb

# 5. Launch demo UI
streamlit run app/ui_streamlit.py
```

### Quick Commands

| Task | Command |
|------|---------|
| Verify data | `python3 -m src.data.split --summary` |
| Train single model | `python3 -m src.models.train --model densenet121` |
| Test Grad-CAM | `python3 -m src.xai.gradcam` |
| Test fusion | `python3 -m src.models.fusion` |
| Build KB | `python3 -m src.rag.build_kb` |
| Run tests | `pytest tests/` |
| Launch UI | `streamlit run app/ui_streamlit.py` |

---

## Technical Deep Dive

### MC-Dropout Uncertainty

Standard dropout is disabled during inference. MC-Dropout keeps it active:

```python
class MCDropout(nn.Module):
    def forward(self, x):
        return F.dropout(x, p=0.3, training=True)  # Always on!
```

For T forward passes, we get T probability distributions. Uncertainty metrics:

```python
# Predictive Entropy (total uncertainty)
H(E[p]) = -Σ p̄ · log(p̄)  where p̄ = (1/T)Σ p_t

# Mutual Information (epistemic/model uncertainty)
MI = H(E[p]) - E[H(p)]
```

### Temperature Scaling

Calibration finds optimal temperature T via validation set:

```python
p_calibrated = softmax(logits / T)
```

Optimization minimizes NLL (or ECE/Brier):
```python
T* = argmin_T NLL(softmax(logits/T), labels)
```

### Saliency Stability Scoring

Tests if Grad-CAM explanation is robust to small perturbations:

1. Generate heatmap for original image
2. Generate K perturbed versions (brightness, noise, crop, contrast)
3. Generate heatmaps for each perturbation
4. Compute SSIM between original and perturbed heatmaps
5. Stability score = mean(SSIM values)

If stability < 0.7, the explanation is suppressed.

### Report Gating Logic

```python
def should_generate_report(prediction, coverage, stability):
    if prediction['abstain']:
        return False, "Model abstained"
    if coverage['score'] < 0.8:
        return False, "Insufficient evidence"
    if stability['score'] < 0.7:
        return False, "Unstable explanation"
    return True, "All gates passed"
```

---

## API Reference

### Core Classes

#### `BrainTumorClassifier`
```python
model = BrainTumorClassifier(
    backbone_name="densenet121",  # or "xception", "efficientnet_b0"
    num_classes=4,
    pretrained=True,
    dropout_rate=0.3,
    mc_dropout=True
)

# Forward pass
logits = model(images)  # (B, 4)
probs = model.predict_proba(images, calibrated=True)

# Set calibration temperature
model.set_temperature(1.5)
```

#### `EGUFusion`
```python
fusion = EGUFusion(
    alpha=1.0,
    max_prob_threshold=0.6,
    entropy_threshold=1.2
)

# Fuse predictions
result = fusion(probs_list, uncertainties)
# result['fused_probs'], result['predictions'], result['abstain']
```

#### `GradCAMGenerator`
```python
generator = GradCAMGenerator(model, "densenet121", device="cuda")
heatmap = generator.generate(input_tensor, target_class=None)
overlay, heatmap, info = generator.generate_overlay(tensor, original_image)
```

#### `KnowledgeBase`
```python
kb = KnowledgeBase("sentence-transformers/all-MiniLM-L6-v2")
kb.add_chunks(chunks)
kb.build_index()
results = kb.search("glioma imaging features", top_k=5)
```

---

## Troubleshooting

### Common Issues

#### SSL Certificate Error
```
ssl.SSLCertVerificationError: certificate verify failed
```
**Solution**:
```bash
# On macOS
/Applications/Python\ 3.x/Install\ Certificates.command

# Or use certifi
pip install certifi
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
```

#### CUDA Out of Memory
**Solution**: Reduce batch size in `configs/train.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

#### Module Not Found
**Solution**: Run from the `oncosense/` directory:
```bash
cd oncosense
python3 -m src.models.train --model densenet121
```

#### Empty Shell Output
If shell commands return empty, verify paths and run with full path:
```bash
python3 /full/path/to/oncosense/quick_setup.py
```

---

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Macro-F1 | ≥ best baseline | Classification performance |
| ECE | < uncalibrated | Calibration quality |
| Brier Score | < uncalibrated | Probability accuracy |
| Citation Coverage | ≥ 95% | Evidence backing |
| Unsupported Claims | < 5% | Report reliability |

---

## License

MIT License

---

## Citation

If you use OncoSense in your research, please cite:
```
@software{oncosense2024,
  title={OncoSense: Research-Grade MRI Brain Tumor Classification System},
  year={2024},
  url={https://github.com/yourusername/oncosense}
}
```

---

## Acknowledgments

- Brain tumor dataset from Kaggle
- PyTorch and Timm for model implementations
- Sentence Transformers and FAISS for RAG
- Streamlit for the demo UI
