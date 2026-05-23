#!/usr/bin/env bash
# Run on RunPod after syncing the project.
# Usage: bash scripts/setup_and_run_phase3.sh
set -euo pipefail

PROJ="/workspace/sam2-peft-project"
cd "$PROJ"

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "=== Python/PyTorch check ==="
python3 --version
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== Install dependencies ==="
pip install -q --upgrade pip
pip install -q pycocotools matplotlib numpy pillow huggingface_hub supervision
pip install -q git+https://github.com/facebookresearch/sam2.git || true
pip install -q -e . 2>/dev/null || true

echo "=== Problem 3.3 — identity init check ==="
python3 scripts/check_adapter_block.py

echo "=== Problems 3.1 + 3.2 — param count + frozen encoder ==="
python3 scripts/check_phase3_peft_setup.py --dataset-root dataset

echo "=== Problem 3.4 — one-epoch smoke run ==="
python3 scripts/run_phase3_training_probe.py \
    --mode smoke \
    --dataset-root dataset \
    --output-dir outputs/phase3_training_probe

echo "=== Problem 3.5 — 20-image overfit probe (5 epochs) ==="
python3 scripts/run_phase3_training_probe.py \
    --mode overfit \
    --dataset-root dataset \
    --output-dir outputs/phase3_training_probe \
    --viz-path viz/phase3_overfit_probe.png

echo "=== Phase 3 complete — files to sync back ==="
ls -lh viz/phase3_overfit_probe.png \
        outputs/phase3_training_probe/smoke/summary.json \
        outputs/phase3_training_probe/smoke/losses.csv \
        outputs/phase3_training_probe/overfit/summary.json \
        outputs/phase3_training_probe/overfit/losses.csv
