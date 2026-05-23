# SAM2 PEFT — Humanoid Robot Component Segmentation

Parameter-efficient fine-tuning of SAM2 for segmenting humanoid robot parts (`arm`, `leg`, `torso`, `head`) from a small custom dataset. PEFT adapter blocks recover **95.4% of the full fine-tune gain using only 3.42% of SAM2's parameters.**

Implementation history and experiment notes: [`IMPLEMENTATION.md`](IMPLEMENTATION.md)  
Dataset details: [`DATASET.md`](DATASET.md)  
Agent instructions: [`AGENTS.md`](AGENTS.md)

---

## Final Results

| Method | Test mIoU | arm | leg | torso | head | Trained params | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot SAM2 | 21.78% | 14.47% | 19.20% | 32.41% | 22.55% | 0 | 237 ms |
| ViT-B/16 supervised | 42.23% | 39.18% | 30.19% | 40.06% | 56.66% | 89.6M (100%) | 19 ms |
| **SAM2 PEFT (ours)** | **80.59%** | **84.50%** | **78.81%** | **74.22%** | **84.82%** | **7.8M (3.42%)** | ~237 ms |
| Full SAM2 fine-tune | 83.43% | 86.24% | 80.98% | 79.11% | 87.37% | 224.4M (100%) | ~237 ms |

**PEFT recovery ratio: 95.4%** — PEFT-SAM2 recovers 95.4% of the full fine-tune mIoU gain over zero-shot using only 3.42% of SAM2's parameters.

Zero-shot and ViT latency measured on RTX 3090. PEFT and full FT latency approximate zero-shot (same SAM2 encoder; PEFT adapters add <0.5% compute).

---

## Dataset

COCO-format segmentation dataset of humanoid robot components. Four foreground classes: `arm`, `leg`, `torso`, `head`. Background label `0` is not a project class.

| Split | Images | Annotations |
|---|---:|---:|
| train | 280 | 1773 |
| val | 60 | 311 |
| test | 58 | 333 |
| **total** | **398** | **2417** |

---

## Method

### Adapter Architecture

Identity-initialized bottleneck adapters inserted after every Hiera transformer block in SAM2's image encoder:

```
input → LayerNorm → Linear(dim→64) → GELU → Linear(64→dim) → + input
```

- 48 adapters total (one per Hiera block)
- Up-projection zero-initialized → identity pass-through at training start
- Only adapters + mask decoder are trained; encoder backbone frozen

### Training Setup — Phase 4 (PEFT)

- Model: `facebook/sam2.1-hiera-large`
- Loss: focal loss + dice loss, L2 regularisation on adapter weights
- Optimizer: AdamW, cosine annealing LR (T_max=50), early stopping patience=10
- Image-grouped encoder caching: encoder runs once per image under `@torch.no_grad()`, decoder once per annotation — 5.3× speedup over naive per-annotation encoding
- Platform: RunPod RTX 3090, ~68 minutes, peak 1.9 GB VRAM

### Training Setup — Phase 5 (Full fine-tune)

- All 224M SAM2 parameters unfrozen
- BF16 mixed precision + batched decoder + RAM image cache + torch.compile
- Platform: RunPod RTX 5090, ~21 minutes, peak 14.3 GB VRAM
- Combined optimization speedup: **6.98×** over FP32 sequential baseline

### Key Training Optimizations (benchmarked on RTX 5090)

| Optimization | Speedup | Notes |
|---|---|---|
| BF16 mixed precision | 2.12× | Dominant win — Blackwell tensor cores |
| Batched decoder | 3.3× | All annotations per image in one GPU call |
| RAM image cache | 0% on 5090 | GPU-bound; helps on CPU-bound hardware |
| torch.compile | ~15% | Enabled automatically |
| **Combined** | **6.98×** | 44s → 6.3s per 50 images |

---

## Reproduction

### Environment

```bash
# RunPod PyTorch template (CUDA 12.4+, Ubuntu 22.04, Python 3.11)
pip install pycocotools matplotlib numpy pillow huggingface_hub supervision
pip install git+https://github.com/facebookresearch/sam2.git
pip install -e .
```

### Phase 1 — Dataset validation

```bash
python scripts/validate_coco_dataset.py --dataset-root dataset
python scripts/smoke_test_dataloader.py --dataset-root dataset
```

### Phase 2 — Baselines

```bash
# Zero-shot SAM2
python scripts/evaluate_zero_shot_sam2.py --dataset-root dataset

# ViT-B/16 supervised baseline
python scripts/train_vit_baseline.py --dataset-root dataset
```

### Phase 3 — Adapter checks

```bash
python scripts/check_adapter_block.py
python scripts/check_phase3_peft_setup.py --dataset-root dataset
python scripts/run_phase3_training_probe.py --mode smoke  --dataset-root dataset --output-dir outputs/phase3_training_probe
python scripts/run_phase3_training_probe.py --mode overfit --dataset-root dataset --output-dir outputs/phase3_training_probe
```

### Phase 4 — PEFT training

```bash
python scripts/train_peft.py \
    --dataset-root dataset \
    --output-dir outputs/phase4 \
    --checkpoint-path outputs/phase4/best_model.pt \
    --epochs 50 \
    --patience 10
```

### Phase 5 — Full fine-tune

```bash
# RTX 5090 (32GB) — no gradient checkpointing needed
python scripts/train_full_finetune.py \
    --dataset-root dataset \
    --output-dir outputs/phase5 \
    --checkpoint-path outputs/phase5/best_model.pt \
    --peft-summary outputs/phase4/summary.json \
    --no-grad-ckpt \
    --epochs 30 \
    --patience 8

# RTX 3090 (24GB) — gradient checkpointing enabled by default
python scripts/train_full_finetune.py \
    --dataset-root dataset \
    --output-dir outputs/phase5 \
    --epochs 30 \
    --patience 8
```

### Optimization benchmark

```bash
python scripts/benchmark_training_optimizations.py \
    --dataset-root dataset \
    --max-images 50 \
    --no-grad-ckpt
```

---

## Output Artifacts

```
outputs/
  phase4/
    summary.json          # PEFT test mIoU, per-class IoU, parameter counts
    training_log.json     # per-epoch train loss and val mIoU
    step_losses.csv
  phase5/
    summary.json          # full fine-tune results + PEFT recovery ratio
    training_log.json
    step_losses.csv
    best_model.pt         # full fine-tune checkpoint (~900MB)

viz/
  phase4_training_curves.png
  phase4_per_class_iou.png
  phase4_qualitative_grid.png
  phase5_training_curves.png
  phase5_per_class_iou.png
  phase5_qualitative_grid.png
  phase5_all_methods_comparison.png
```

---

## Project Structure

```
sam2-peft-project/
  scripts/
    train_peft.py                         # Phase 4 PEFT training
    train_full_finetune.py                # Phase 5 full fine-tune
    benchmark_training_optimizations.py  # optimization benchmark
    evaluate_zero_shot_sam2.py           # Phase 2 zero-shot evaluation
    train_vit_baseline.py                # Phase 2 ViT baseline
    check_adapter_block.py               # Phase 3 identity init check
    check_phase3_peft_setup.py           # Phase 3 param count + frozen check
    run_phase3_training_probe.py         # Phase 3 smoke + overfit probe
    validate_coco_dataset.py             # Phase 1 dataset validation
  src/sam2_peft/
    phase3.py                            # dataset, loss, inference utilities
    models/
      adapters.py                        # AdapterBlock, AdapterWrappedBlock
      sam2_peft.py                       # insert_hiera_adapters, configure_peft_trainable
  dataset/                               # COCO-format dataset (not in repo)
  outputs/                               # training outputs (not in repo)
  viz/                                   # visualizations (not in repo)
  IMPLEMENTATION.md                      # full experiment log
  DATASET.md                             # dataset construction details
```
