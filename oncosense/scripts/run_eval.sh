#!/bin/bash
# OncoSense Evaluation Pipeline
# Run calibration, fusion evaluation, and robustness tests

set -e

echo "=============================================="
echo "OncoSense Evaluation Pipeline"
echo "=============================================="

cd "$(dirname "$0")/.."

# Create results directory
mkdir -p results

# Step 1: Calibrate models
echo ""
echo "[1/4] Calibrating models..."
python -c "
from src.models.calibrate import calibrate_model
from src.models.backbones import get_backbone
from src.data.transforms import create_dataloaders

_, val_loader, _ = create_dataloaders()

for model_name in ['densenet121', 'xception', 'efficientnet_b0']:
    print(f'\n--- Calibrating {model_name} ---')
    model = get_backbone(model_name, config_path='configs/models.yaml')
    from src.models.backbones import load_checkpoint
    model = load_checkpoint(model, f'checkpoints/{model_name}_best.pt', 'cuda')
    temp, metrics = calibrate_model(model, val_loader)
    print(f'Optimal temperature: {temp:.4f}')
"

# Step 2: Evaluate fusion
echo ""
echo "[2/4] Evaluating EGU-Fusion++..."
python -c "
from src.models.backbones import get_ensemble
from src.models.fusion import EGUFusion, evaluate_fusion
from src.data.transforms import create_dataloaders

_, _, test_loader = create_dataloaders()
models = get_ensemble(config_path='configs/models.yaml')

fusion = EGUFusion(alpha=1.0)
metrics = evaluate_fusion(models, test_loader, fusion)

print('\n--- Fusion Evaluation Results ---')
for k, v in metrics.items():
    if isinstance(v, float):
        print(f'{k}: {v:.4f}')
"

# Step 3: Robustness evaluation
echo ""
echo "[3/4] Running robustness evaluation..."
python -c "
from src.data.corruptions import evaluate_robustness, CORRUPTIONS
from src.models.backbones import get_backbone, load_checkpoint
from src.data.transforms import create_dataloaders, BrainTumorDataset
import pandas as pd
import json

_, _, test_loader = create_dataloaders()
test_dataset = test_loader.dataset

model = get_backbone('densenet121')
model = load_checkpoint(model, 'checkpoints/densenet121_best.pt', 'cuda')
model.eval()

# Test subset of corruptions
corruptions_to_test = ['gaussian_noise', 'gaussian_blur', 'brightness', 'contrast']
results = evaluate_robustness(model, test_dataset, corruptions_to_test, severities=[1, 3, 5])

print('\n--- Robustness Results (DenseNet121) ---')
for corr, severities in results.items():
    print(f'{corr}:')
    for sev, acc in severities.items():
        print(f'  Severity {sev}: {acc:.2f}%')
"

# Step 4: Generate report
echo ""
echo "[4/4] Generating evaluation report..."
python -c "
import json
from datetime import datetime

report = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'status': 'Evaluation complete',
    'note': 'See individual metric outputs above'
}

with open('results/eval_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('\nEvaluation report saved to results/eval_report.json')
"

echo ""
echo "=============================================="
echo "Evaluation pipeline complete!"
echo "=============================================="
