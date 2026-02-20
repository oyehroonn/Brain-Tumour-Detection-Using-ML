#!/bin/bash
# OncoSense Training Pipeline
# Run all steps from data download to model training

set -e

echo "=============================================="
echo "OncoSense Training Pipeline"
echo "=============================================="

cd "$(dirname "$0")/.."

# Step 1: Download data
echo ""
echo "[1/6] Downloading dataset from Kaggle..."
python -m src.data.download_kaggle

# Step 2: Create manifest
echo ""
echo "[2/6] Creating data manifest..."
python -m src.data.dedup --create-manifest --data-dir data/raw

# Step 3: Deduplicate
echo ""
echo "[3/6] Deduplicating dataset..."
python -m src.data.dedup

# Step 4: Create splits
echo ""
echo "[4/6] Creating train/val/test splits..."
python -m src.data.split

# Step 5: Train models
echo ""
echo "[5/6] Training models..."

echo "Training DenseNet121..."
python -m src.models.train --model densenet121

echo "Training Xception..."
python -m src.models.train --model xception

echo "Training EfficientNet-B0..."
python -m src.models.train --model efficientnet_b0

# Step 6: Build knowledge base
echo ""
echo "[6/6] Building knowledge base..."
python -m src.rag.build_kb

echo ""
echo "=============================================="
echo "Training pipeline complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Run evaluation: ./scripts/run_eval.sh"
echo "  2. Launch demo: streamlit run app/ui_streamlit.py"
