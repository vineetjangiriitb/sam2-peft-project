# Implementation Log

This file records how the project is implemented phase by phase, including what worked, what failed, and workflow changes made during development. Keep it updated after every completed check, training run, failed experiment, or infrastructure change.

## Current Execution Context

- Primary training platform: RunPod over SSH.
- Target template: RunPod PyTorch 2.4.0.
- CUDA: 12.4.1.
- OS: Ubuntu 22.04.
- Python: 3.11.
- GPU target: RTX 3090 with 24GB VRAM, or RTX 4090-class GPU.
- Current pod style: on-demand pod at about `$0.46/hr`.
- Current disk setup: 20GB temporary container disk, 0GB volume disk, no network volume.
- Main operational constraint: container disk is temporary and erased on stop, so checkpoints, raw metrics, logs, and generated visualizations must be copied out before stopping the pod.
- Preferred interface: SSH command-line execution. Jupyter is not the primary workflow unless explicitly needed for inspection.
- Long-running training/evaluation commands should run inside `tmux`.
- After each training or evaluation run, sync outputs, checkpoints, logs, visualizations, and updated notes back to the local Mac because the pod has no persistent paid storage.

## Class Taxonomy

The project has exactly four foreground robot-part classes:

- `arm`
- `leg`
- `torso`
- `head`

Semantic segmentation masks use label `0` for background. Background is required internally for masks and metrics, but it is not counted as a fifth project class.

## Phase 0 - Environment And Sanity Check

### Intended Implementation

Phase 0 proves that the target GPU environment can import SAM2, load the selected SAM2 model, run single-image inference, save a visualization, and leave enough VRAM headroom for the next training smoke tests.

### What Worked

- A legacy smoke-test artifact exists:
  - `viz/phase0_inference_check.png`
- The project has a notebook artifact:
  - `notebooks/phase0_sam2_sanity_check.ipynb`

### What Failed Or Changed

- The original workflow assumed Colab/A100.
- Colab became unreliable for the project workflow.
- The official target environment has changed to RunPod SSH on RTX 3090/4090-class hardware.

### Current Status

Pending official rerun on RunPod. The existing T4/Colab-style artifact is useful as a smoke reference but does not complete the new target-environment check.

### Next Required Evidence

- `nvidia-smi` output from the RunPod pod.
- Python, PyTorch, CUDA, and SAM2 import checks.
- Single-image SAM2 inference output saved to `viz/phase0_inference_check.png`.
- VRAM usage noted after loading/inference.

## Phase 1 - Dataset Construction

### Intended Implementation

Build a COCO-format segmentation dataset of humanoid robot components with train/val/test splits and polygon masks for exactly four foreground classes.

### What Worked

- Dataset exists under:
  - `dataset/images/train`
  - `dataset/images/val`
  - `dataset/images/test`
  - `dataset/annotations/train.json`
  - `dataset/annotations/val.json`
  - `dataset/annotations/test.json`
- Validated foreground classes:
  - `arm`
  - `leg`
  - `torso`
  - `head`
- Current annotated split from the COCO JSON files:
  - train: 280 images, 1773 annotations
  - val: 60 images, 311 annotations
  - test: 58 images, 333 annotations
  - total: 398 annotated images
- The total annotated image count is within the required 350-500 image range.
- The class-balance validator reports the most dominant class at 34.5% of annotations.
- Visualizations exist:
  - `viz/phase1_mask_alignment.png`
  - `viz/phase1_class_balance.png`

### Checks That Passed

```bash
python3 scripts/validate_coco_dataset.py --dataset-root dataset
python3 scripts/smoke_test_dataloader.py --dataset-root dataset
```

The dataloader smoke test loads a batch of `1024x1024` images and masks. Masks contain background label `0` plus the four foreground class labels.

### What Failed Or Changed

- Earlier wording could make label `0` sound like a fifth class. That has been corrected: `0` is background only.
- The dataset contract now requires exactly the four foreground classes instead of allowing a minimum of three usable classes.
- There are a few image files on disk that are not referenced by the COCO JSON split. The official dataset count should come from the JSON files, not raw file count.

### Current Status

Complete and locally verified.

## Phase 2 - Baseline Evaluations

### Intended Implementation

Evaluate:

- zero-shot SAM2 on the held-out test split
- the original ViT segmentation baseline on the same test split

Both methods must use the same dataset split, preprocessing contract where applicable, and mIoU implementation.

### What Worked

- Zero-shot SAM2 evaluation script exists:
  - `scripts/evaluate_zero_shot_sam2.py`
- The script can compute per-image mIoU, per-class IoU, latency, CSV results, JSON summary, and visualizations.
- The script has passed a one-image local MPS smoke test according to project status notes.
- The official RunPod zero-shot evaluation completed on the full 58-image test split.
- Official zero-shot SAM2 results:
  - images: 58
  - annotations: 333
  - mean mIoU: 0.2178034019288507
  - mean latency: 236.8234081659466 ms/image
  - p95 latency: 499.03584364801645 ms/image
  - per-class IoU:
    - arm: 0.14472603446980756
    - leg: 0.1920250337511719
    - torso: 0.3241467426419656
    - head: 0.22550119375955785

- ViT baseline built from scratch: ViT-B/16 backbone (ImageNet pretrained) + 4-stage ConvTranspose2d decoder.
- ViT baseline trained on the Phase 1 train split (280 images, 30 epochs, AdamW + CosineAnnealingLR, batch 16, lr=1e-4).
- ViT baseline evaluated on the 58-image test split on RunPod RTX 3090.
- Official ViT baseline results:
  - images: 58
  - mean mIoU: 0.4223
  - mean latency: 19.4 ms/image
  - p95 latency: 67.8 ms/image
  - per-class IoU:
    - arm: 0.3918
    - leg: 0.3019
    - torso: 0.4006
    - head: 0.5666
  - best checkpoint: epoch 20/30, val mIoU 0.4558
  - total trainable parameters: 89,634,117

### What Failed Or Changed

- Zero-shot SAM2 is weaker than the rough plan expectation of 30-55% mIoU under the current bbox-center point-prompt setup.
- `arm` is the weakest class in zero-shot SAM2; `torso` is the strongest.
- ViT baseline was built from scratch rather than recovered from an existing checkpoint (no prior artifact existed).
- Leg is the weakest ViT class (30.2%); head is the strongest (56.7%). Torso did not rank highest as originally expected — likely due to arm/torso boundary overlap in tight robot shots.
- P95 latency (67.8 ms) slightly exceeds the 40 ms expected range due to first-batch GPU warmup on some images.
- Official Phase 2 execution ran on RunPod over SSH, not Colab.

### Current Status

**Phase 2 is complete.** Both baselines evaluated on the same 58-image test split using the same mIoU implementation.

| Method | mIoU | arm | leg | torso | head | Mean Latency |
|---|---|---|---|---|---|---|
| Zero-shot SAM2 | 21.78% | 14.47% | 19.20% | 32.41% | 22.55% | 236.8 ms |
| ViT-B/16 supervised | 42.23% | 39.18% | 30.19% | 40.06% | 56.66% | 19.4 ms |

ViT is +20.5 pts mIoU and 12× faster than zero-shot SAM2. This sets the supervised ceiling that PEFT-SAM2 must approach using <2% of SAM2's parameters.

### Artifacts

- `outputs/phase2_vit_baseline/best.pt` — best ViT checkpoint (epoch 20)
- `outputs/phase2_vit_baseline/results.csv` — per-image mIoU and per-class IoU
- `outputs/phase2_vit_baseline/summary.json` — aggregate metrics
- `outputs/phase2_vit_baseline/training_history.json` — per-epoch train loss and val mIoU
- `viz/phase2_vit_iou_distribution.png` — test mIoU histogram
- `viz/phase2_vit_failure_modes.png` — 8 worst prediction grids
- `logs/train_vit_baseline.log` — full training stdout

## Phase 3 - Adapter Implementation

### Intended Implementation

Insert identity-initialized adapter blocks into SAM2's image encoder, keep the non-adapter encoder frozen, and train only adapters plus the mask decoder.

### What Worked

- Standalone adapter block exists:
  - `src/sam2_peft/models/adapters.py`
- Adapter identity check passes locally:

```bash
python3 scripts/check_adapter_block.py
```

Observed local check:

- Adapter dim: 256
- Bottleneck: 64
- Max identity diff: 0.00000000

### Checks That Passed

All five Phase 3 problems passed on RunPod (RTX 3090, CUDA 12.4, PyTorch 2.4.1).

**Problem 3.3 — Identity init (local + pod)**
- Max adapter output diff before training: 0.00000000
- PASS

**Problem 3.1 — Parameter count**
- Inserted adapters: 48 (one per Hiera block)
- Trainable: 4,461,493 / 224,693,026 = 1.99%
- PASS (threshold: ≤6M params, ≤2%)

**Problem 3.2 — Frozen encoder gradients**
- One backward pass run; all non-adapter encoder weights confirmed grad=None
- PASS

**Problem 3.4 — Smoke run (50 samples, 1 epoch)**
- No NaN loss at any step
- Max GPU memory: 8.31 GB / 24 GB
- PASS

**Problem 3.5 — 20-image overfit probe (5 epochs)**
- 20 images, 125 annotated instances, 5 epochs = 625 steps
- Epoch 5 mean loss: 0.1020 (dropped from 0.427 at step 1)
- Train mIoU on 20 images: **0.8437 (84.4%)**
- PASS (threshold: ≥80%)

**Bug fixed during run:** Adapters inserted after model.to(device) call left adapter LayerNorm weights on CPU. Fixed by calling .to(device) on each AdapterWrappedBlock immediately after insertion in insert_hiera_adapters().

### Output Artifacts

- `viz/phase3_overfit_probe.png` — loss curve + mIoU bar + 4 prediction sample rows
- `outputs/phase3_training_probe/smoke/summary.json`
- `outputs/phase3_training_probe/smoke/losses.csv`
- `outputs/phase3_training_probe/overfit/summary.json`
- `outputs/phase3_training_probe/overfit/losses.csv`

### Current Status

Complete. All five problems pass. Ready to proceed to Phase 4.

## Phase 4 - PEFT Training

### Intended Implementation

Train the SAM2 PEFT model on the Phase 1 dataset using adapters plus mask-decoder training, then evaluate on the held-out test split.

### What Worked

No official PEFT training run has been completed yet.

### What Is Not Implemented Yet

- Full PEFT training script.
- Config for RTX 3090/4090 24GB-aware training.
- Checkpoint saving and resume behavior.
- Training and validation curves.
- Test-set mIoU.
- Per-class IoU chart.
- Qualitative prediction grid.

### Current Status

Not started beyond prerequisite utilities.

## Phase 5 - Full Fine-Tune Comparison

### Intended Implementation

Train a full SAM2 fine-tune on the same split for an upper-bound comparison, then compare full fine-tune against PEFT on mIoU, trainable parameters, latency, and out-of-domain behavior.

### What Worked

No full fine-tune run has been completed yet.

### What Is Not Implemented Yet

- Full fine-tune training script/config.
- Full fine-tune checkpoint.
- Full fine-tune test mIoU.
- PEFT recovery ratio.
- Out-of-domain comparison grid.
- Final all-method comparison figure.

### Current Status

Not started.

## Phase 6 - Results And Write-Up

### Intended Implementation

Fill the final results table, update README reproduction steps, add final visualizations, and write the final project summary using real measured values.

### What Worked

- README contains the current project status and expected final table structure.
- Project plan defines the final comparison requirements.

### What Is Not Implemented Yet

- Final results table with real numbers.
- Resume bullet with measured mIoU, recovery ratio, and parameter count.
- Complete reproduction commands for RunPod.
- Final benchmark evidence.

### Current Status

Not ready because Phases 2-5 do not yet have official result artifacts.

## Running Notes To Keep Updated

Add dated notes below whenever a meaningful experiment or workflow change happens.

### 2026-05-23 - Compute Workflow Changed

- Changed official compute target from Colab/A100 to RunPod over SSH.
- Target pod details recorded:
  - PyTorch 2.4.0 template
  - CUDA 12.4.1
  - Ubuntu 22.04
  - Python 3.11
  - RTX 3090 24GB VRAM or RTX 4090-class GPU
  - 20GB temporary container disk
- Updated docs to treat SSH command-line execution as the primary workflow.
- Added `tmux` as the default way to run long GPU jobs.
- Added sync-back-after-each-run as a hard workflow step because the pod has no persistent volume.
- Clarified that the project has four foreground robot-part classes and that label `0` is background only.

### 2026-05-23 - RunPod Dependency Setup Notes

- Phase 1 dataset validation passed on the RunPod pod.
- `scripts/smoke_test_dataloader.py` initially failed because `pycocotools` was missing from the pod environment.
- SAM2 model loading initially failed because `huggingface_hub` was missing; SAM2's `from_pretrained("facebook/sam2.1-hiera-large")` path needs it to download model assets.
- Added `huggingface_hub` to `requirements.txt`.

### 2026-05-23 - Phase 2 Zero-Shot SAM2 RunPod Result

- Smoke run on 2 test images completed successfully.
- Full zero-shot SAM2 run completed on the 58-image test split.
- Output paths on the pod:
  - `outputs/phase2_zero_shot_sam2/results.csv`
  - `outputs/phase2_zero_shot_sam2/summary.json`
  - `viz/phase2_failure_modes.png`
  - `viz/phase2_iou_distribution.png`
- Summary:
  - mean mIoU: 0.2178034019288507
  - mean latency: 236.8234081659466 ms/image
  - p95 latency: 499.03584364801645 ms/image
  - arm IoU: 0.14472603446980756
  - leg IoU: 0.1920250337511719
  - torso IoU: 0.3241467426419656
  - head IoU: 0.22550119375955785
- Interpretation so far: zero-shot SAM2 gives a low baseline floor under bbox-center point prompting. This gives PEFT substantial room to improve, especially for arms and legs.
- Failure-mode inspection from `viz/phase2_failure_modes.png`: zero-shot SAM2 often segments an entire robot/object or a broad background region rather than the requested component part. Several worst cases show missed thin limbs, overfilled torso/head regions, and object-level masks that ignore the `arm`/`leg`/`torso`/`head` taxonomy. This supports the need for adapter fine-tuning rather than relying on zero-shot prompting alone.

### 2026-05-23 - Phase 2 ViT Baseline RunPod Training And Evaluation

- Built ViT-B/16 segmentation baseline from scratch on `phase2-vit-baseline` branch.
  - Architecture: `torchvision.vit_b_16` (ImageNet pretrained) + 4-stage ConvTranspose2d decoder (14→224).
  - Output: `(N, 5, 224, 224)` raw logits; 5 classes = background + arm, leg, torso, head.
  - Total parameters: 89,634,117 (full model, no freezing).
- Training on RunPod RTX 3090:
  - Dataset: Phase 1 train split (280 images), image_size=224, batch=16.
  - Optimizer: AdamW lr=1e-4, weight_decay=1e-4; scheduler: CosineAnnealingLR over 30 epochs.
  - Loss: CrossEntropyLoss over 5 classes.
  - Duration: ~3 minutes total (≈6 s/epoch).
  - Best checkpoint: epoch 20, val mIoU=0.4558.
- Evaluation on 58-image test split:
  - mean mIoU: 0.4223
  - mean latency: 19.4 ms/image
  - p95 latency: 67.8 ms/image
  - arm: 0.3918  |  leg: 0.3019  |  torso: 0.4006  |  head: 0.5666
- Leg was the hardest class (thin, articulated). Head was the easiest (compact shape). Torso ranked lower than expected — likely arm/torso boundary confusion in close-range robot shots.
- All outputs synced to local Mac. Pod stopped after sync confirmed.
- Phase 2 is now fully complete. PEFT-SAM2 target: beat 42.23% mIoU with ≤2% of SAM2 parameters.
