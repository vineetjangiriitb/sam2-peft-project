# SAM2 PEFT Project

This project adapts SAM2 for humanoid robot component segmentation using parameter-efficient fine-tuning under severe annotation scarcity.

The execution source of truth is [`SAM2_PEFT_Project_Plan.md`](SAM2_PEFT_Project_Plan.md). AI coding agents must also follow [`AGENTS.md`](AGENTS.md).

## Current Status

- Phase 0 T4 smoke test has run.
- Official A100 Phase 0 target-environment check is pending.
- Phase 1 dataset construction is complete for the four-class dataset: `arm`, `leg`, `torso`, `head`.
- Dataset-independent utilities for COCO loading, mIoU, and adapter identity checks are present.
- Phase 2 zero-shot SAM2 evaluation script is present and has passed a one-image local MPS smoke test.
- Full Phase 2 zero-shot evaluation should be run on Colab/A100 for official numbers.
- ViT baseline evaluation is blocked until the original ViT model code/checkpoint is provided.
- Heavy SAM2 training and validation are expected to run on Google Colab Pro with an A100 GPU.

## Completion Contract

The project is complete only after the four-way benchmark is finished and documented:

| Method | Test mIoU | Params Trained | Latency | Status |
|---|---:|---:|---:|---|
| Zero-shot SAM2 | TBD | 0 | TBD | Not run |
| ViT baseline | TBD | TBD | TBD | Not run |
| SAM2 PEFT | TBD | <2% of SAM2 | TBD | Not run |
| Full SAM2 fine-tune | TBD | ~100% of SAM2 | TBD | Not run |

Required final evidence:

- SAM2 adapted to humanoid robot component segmentation.
- PEFT adapters plus mask-decoder training update fewer than 2% of parameters.
- Dataset size stays around 350-500 annotated images.
- PEFT recovers at least 85% of the full fine-tune mIoU improvement over zero-shot SAM2.
- Out-of-domain comparison shows PEFT resists catastrophic forgetting better than, or at least equal to, full fine-tuning.
- Final visualizations and reproduction instructions are included.

## Phase 0

Open `notebooks/phase0_sam2_sanity_check.ipynb` in Colab and run Problems 0.1-0.3.

Phase 0 passes only when:

- SAM2 imports successfully.
- Colab reports an A100 GPU.
- Single-image SAM2 inference creates `viz/phase0_inference_check.png`.
- Model memory leaves enough VRAM headroom for training.

## Phase 1

Build the dataset according to [`DATASET.md`](DATASET.md).

Expected layout:

```text
dataset/
  images/
    train/
    val/
    test/
  annotations/
    train.json
    val.json
    test.json
```

After exporting COCO segmentation data, run:

```bash
python scripts/import_roboflow_coco.py --source path/to/roboflow-export.zip
python scripts/validate_coco_dataset.py
python scripts/visualize_phase1_dataset.py
python scripts/smoke_test_dataloader.py
```

Phase 1 passes only when the structure checks pass and the visualizations are saved:

- `viz/phase1_mask_alignment.png`
- `viz/phase1_class_balance.png`

Current split:

```text
train: 280 images
val:    60 images
test:   58 images
```

## Phase 2

Run the zero-shot SAM2 baseline on the held-out test split:

```bash
python scripts/evaluate_zero_shot_sam2.py
```

For a quick smoke test, limit the run:

```bash
python scripts/evaluate_zero_shot_sam2.py --max-images 2
```

Outputs:

- `outputs/phase2_zero_shot_sam2/results.csv`
- `outputs/phase2_zero_shot_sam2/summary.json`
- `viz/phase2_failure_modes.png`
- `viz/phase2_iou_distribution.png`

The local Mac can smoke-test this script with MPS, but official Phase 2 metrics should be generated on the Colab/A100 environment used for the rest of the SAM2 experiments.

## Local Checks Without Dataset

These checks do not require the Phase 1 dataset:

```bash
python scripts/check_metrics.py
python scripts/check_adapter_block.py
```

The adapter script verifies the identity initialization required by Phase 3 Problem 3.3.
