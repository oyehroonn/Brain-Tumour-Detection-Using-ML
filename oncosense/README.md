# OncoSense

**Research-grade MRI tumor triage and decision-support system**

OncoSense is a low-cost MRI tumor classification research system that generates guidance only when predictions are calibrated, confident, and evidence-backed.

## Overview

This system trains multiple MRI classifiers, calibrates them, fuses predictions using uncertainty-weighted gating (EGU-Fusion++), and abstains on ambiguous/OOD cases. It generates weak localization (Grad-CAM) with stability scoring, and a retrieval-grounded LLM (RAG) produces structured, citation-backed reports only when gating conditions are satisfied.

## Features

- **4-Class Classification**: Glioma, Meningioma, Pituitary tumor, No tumor
- **Multi-Model Ensemble**: DenseNet121, Xception, EfficientNet-B0
- **EGU-Fusion++**: Uncertainty-weighted prediction fusion
- **Calibrated Predictions**: Temperature scaling for reliable confidence scores
- **Abstention**: Refuses to predict on low-confidence or OOD cases
- **Explainability**: Grad-CAM with saliency stability gating
- **RAG Reports**: Evidence-grounded structured reports with citations

## Installation

```bash
cd oncosense
pip install -r requirements.txt
```

## Quick Start

### 1. Download and Prepare Data

```bash
python -m src.data.download_kaggle
python -m src.data.dedup
python -m src.data.split
```

### 2. Train Models

```bash
python -m src.models.train --model densenet121
python -m src.models.train --model xception
python -m src.models.train --model efficientnet_b0
```

### 3. Calibrate and Evaluate

```bash
python -m src.models.calibrate
python -m src.models.fusion --evaluate
```

### 4. Run Demo UI

```bash
streamlit run app/ui_streamlit.py
```

## Project Structure

```
oncosense/
├── configs/           # YAML configuration files
├── data/              # Dataset storage
│   ├── raw/           # Downloaded images
│   ├── processed/     # Preprocessed images
│   └── manifests/     # Split manifests (parquet)
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model training and inference
│   ├── xai/           # Explainability (Grad-CAM)
│   ├── rag/           # RAG and report generation
│   └── report/        # Report rendering
├── app/               # Streamlit demo UI
├── knowledge_base/    # RAG document chunks
├── checkpoints/       # Model weights
└── tests/             # Unit tests
```

## Key Metrics

| Metric | Target |
|--------|--------|
| Macro-F1 | ≥ best single baseline |
| ECE | < uncalibrated baseline |
| Citation Coverage | ≥ 95% |
| Unsupported Claims | < 5% |

## Disclaimer

**This is a research prototype, NOT a clinical diagnostic product.**

This system is intended for research and educational purposes only. It should not be used for clinical diagnosis, treatment planning, or any medical decision-making. Always consult qualified healthcare professionals for medical advice.

## License

MIT License
