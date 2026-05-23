#!/usr/bin/env bash
# Phase 5 — full SAM2 fine-tune on RunPod.
# Run inside tmux: tmux new -s phase5
# Usage: bash scripts/setup_and_run_phase5.sh
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

echo "=== Phase 5 — full SAM2 fine-tune (all params, grad checkpointing) ==="
mkdir -p outputs/phase5 viz logs

python3 scripts/train_full_finetune.py \
    --dataset-root dataset \
    --output-dir outputs/phase5 \
    --checkpoint-path outputs/phase5/best_model.pt \
    --peft-summary outputs/phase4/summary.json \
    --epochs 30 \
    --lr 1e-5 \
    --patience 8 \
    2>&1 | tee logs/phase5_train.log

echo "=== Phase 5 complete — files to sync back ==="
ls -lh outputs/phase5/summary.json \
        outputs/phase5/training_log.json \
        outputs/phase5/step_losses.csv \
        viz/phase5_training_curves.png \
        viz/phase5_per_class_iou.png \
        viz/phase5_qualitative_grid.png \
        viz/phase5_all_methods_comparison.png
