# ViT Baseline — Evaluation Metrics

Metrics used to evaluate the ViT segmentation baseline on the 58-image test split.
Fill in the **Actual** column after the RunPod evaluation run completes.

---

## Primary Metric

| Metric | Expected Range | Actual |
|---|---|---|
| Mean IoU (mIoU) — foreground classes only | 40 – 65% | **42.23%** |

mIoU is computed over the 4 foreground classes (arm, leg, torso, head), background excluded.
Each class IoU is computed only on images where that class appears (None values skipped).
Final mIoU is the mean over all non-None per-class IoUs across the test split.

---

## Per-Class IoU

| Class | Expected Range | Actual |
|---|---|---|
| arm | 30 – 55% | **39.18%** |
| leg | 35 – 55% | **30.19%** |
| torso | 50 – 70% | **40.06%** |
| head | 40 – 65% | **56.66%** |

**Why torso is highest:** torso annotations are the largest and most compact blobs — easiest to segment.
**Why arm is lowest:** arms are thin, articulated, and have high aspect-ratio variation.

---

## Latency

| Metric | Expected Range | Actual |
|---|---|---|
| Mean inference latency (ms/image) | 5 – 20 ms | **19.4 ms** |
| P95 inference latency (ms/image) | 10 – 40 ms | **67.8 ms** |

ViT inference at 224×224 is much faster than SAM2's 1024×1024 pipeline.
Latency measured as wall-clock time per image on the GPU pod (not including data loading).

---

## Training Diagnostics

| Metric | Expected Range | Actual |
|---|---|---|
| Best epoch (epoch where val mIoU peaks) | 15 – 28 | **epoch 20** |
| Final train loss (cross-entropy) | 0.2 – 0.6 | **0.5127** |
| Val mIoU at best checkpoint | 40 – 65% | **45.58%** |
| Trainable parameters | ~91M (full model) | **89,634,117** |

---

## Comparison Against Zero-Shot SAM2 Baseline

| Method | mIoU | arm IoU | leg IoU | torso IoU | head IoU | Mean Latency |
|---|---|---|---|---|---|---|
| Zero-shot SAM2 (bbox-center prompt) | 21.78% | 14.47% | 19.20% | 32.41% | 22.55% | 236.8 ms |
| ViT-B/16 supervised baseline | **42.23%** | **39.18%** | **30.19%** | **40.06%** | **56.66%** | **19.4 ms** |

The ViT baseline should substantially outperform zero-shot SAM2 because it is trained on the
domain-specific robot dataset. The gap sets the ceiling that PEFT-SAM2 must close.

---

## What Good Looks Like

- mIoU > 40%: baseline is meaningful; PEFT-SAM2 has a real target to chase
- mIoU > 55%: strong supervised baseline; PEFT recovery ratio will be the headline result
- mIoU < 30%: something is wrong — check label alignment, class index mapping, or image preprocessing

---

## Notes

- Image size: 224×224 (ViT-B/16 native patch size)
- Loss: CrossEntropyLoss over 5 classes (background + 4 foreground)
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Epochs: 30
- Best checkpoint saved by val mIoU, not train loss
