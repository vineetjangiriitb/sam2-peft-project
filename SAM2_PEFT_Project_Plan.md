# SAM2 PEFT Adaptation for Robotics — Project Execution Plan

> **Goal:** Adapt SAM2 to robot component segmentation using parameter-efficient fine-tuning (PEFT adapters + mask decoder fine-tune), compare against full fine-tune and zero-shot baselines, and demonstrate that <2% parameter update recovers >90% of full fine-tune mIoU.
>
> **Compute:** Google Colab Pro (A100)
> **Dataset target:** 300–500 images of humanoid robot components with segmentation masks

---

## How to use this plan

Each phase has:
- **What you build** — the concrete deliverable
- **Chapter problems** — specific checks you run to verify the phase is complete and correct
- **Pass threshold** — the number that tells you "this phase is done, move forward"

If you fail a check, you stay in that phase and debug. If you pass, you move to the next phase. Treat every check like a problem set — the output is your signal.

### Visualisation outputs

Every phase produces saved figures into a `viz/` folder. Create it once:
```bash
mkdir -p viz
```

By the end of the project, `viz/` will contain:

| File | Phase | What it shows |
|---|---|---|
| `phase0_inference_check.png` | 0 | Input image, SAM2 mask, boundary overlay |
| `phase1_mask_alignment.png` | 1 | 5 val images with GT mask colours overlaid |
| `phase1_class_balance.png` | 1 | Bar chart + pie chart of class distribution |
| `phase2_failure_modes.png` | 2 | 10 worst zero-shot predictions (GT vs predicted) |
| `phase2_iou_distribution.png` | 2 | Histogram of per-image IoU for zero-shot SAM2 |
| `phase3_overfit_probe.png` | 3 | Loss curve + mIoU bar for 20-image overfit test |
| `phase4_training_curves.png` | 4 | Train/val loss and mIoU across all epochs |
| `phase4_per_class_iou.png` | 4 | Horizontal bar chart, colour-coded pass/warn/fail |
| `phase4_qualitative_grid.png` | 4 | GT vs PEFT prediction, one row per class |
| `phase5_ood_comparison.png` | 5 | Out-of-domain: PEFT vs full fine-tune side by side |
| `phase5_final_comparison.png` | 5 | 3-panel summary: mIoU, params, latency across all methods |

`phase5_final_comparison.png` is the one that goes in your GitHub README and resume screenshots.

---

## Phase 0 — Environment & Sanity Check

**What you build:** A working Colab notebook that can run SAM2 inference on a single image.

### Setup steps
```bash
pip install sam2 supervision pycocotools

# Download SAM2.1 large checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Chapter problems

**Problem 0.1 — Import check**
```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print(torch.cuda.get_device_name(0))   # should print A100
print(torch.cuda.memory_allocated() / 1e9)  # should be near 0
```
✅ Pass: prints "A100" with no import errors

**Problem 0.2 — Single image inference + visualisation**
Download any robot image from the web. Run SAM2 with a point prompt at the centre of the image.
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

image = np.array(Image.open("robot_test.jpg").convert("RGB"))
h, w  = image.shape[:2]

predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=np.array([[w // 2, h // 2]]),
    point_labels=np.array([1]),
    multimask_output=False,
)

# --- Visualisation ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].plot(w // 2, h // 2, "r*", markersize=12)
axes[0].set_title("Input + prompt point")
axes[0].axis("off")

axes[1].imshow(image)
axes[1].imshow(masks[0], alpha=0.5, cmap="Reds")
axes[1].set_title(f"SAM2 mask (score: {scores[0]:.2f})")
axes[1].axis("off")

boundary = masks[0].astype(np.uint8)
axes[2].imshow(image)
contour_overlay = np.zeros_like(image)
contour_overlay[boundary == 1] = [255, 0, 0]
axes[2].imshow(contour_overlay, alpha=0.4)
axes[2].set_title("Mask boundary overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("viz/phase0_inference_check.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Mask coverage: {masks[0].sum() / (h*w) * 100:.1f}% of image")
```
✅ Pass: left panel shows prompt point, middle panel shows a coloured mask region, right panel shows clean boundary. Mask coverage should be 5–40% of image (not 0%, not 95%).

**Problem 0.3 — Memory budget check**
```python
# After loading model
print(f"Model memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
# Should be < 10 GB, leaving headroom for training
```
✅ Pass: model fits with >10 GB VRAM remaining

### Phase 0 pass threshold
All three problems pass. You have a running environment. Estimated time: **2–3 hours**

---

## Phase 1 — Dataset Construction

**What you build:** A COCO-format segmentation dataset of robot components, split into train/val/test.

### Target spec
- 350–500 images total
- Classes: `arm`, `leg`, `torso`, `head` (use what's available — minimum 3 classes)
- Format: COCO JSON with segmentation polygon masks
- Split: 70% train / 15% val / 15% test

### Data sources (in priority order)
1. Your existing ViT segmentation dataset — re-export masks as COCO JSON from your annotation tool
2. [Roboflow Universe](https://universe.roboflow.com) — search "humanoid robot segmentation", download under CC license
3. Manual annotation of new images using [Label Studio](https://labelstud.io) (free, self-hosted) or Roboflow free tier

### Chapter problems

**Problem 1.1 — COCO JSON structure check**
```python
import json

with open("dataset/annotations/train.json") as f:
    coco = json.load(f)

print("Images:", len(coco["images"]))
print("Annotations:", len(coco["annotations"]))
print("Categories:", [c["name"] for c in coco["categories"]])

# Every annotation must have a segmentation field (not just bbox)
has_seg = all("segmentation" in a for a in coco["annotations"])
print("All have segmentation masks:", has_seg)
```
✅ Pass: ≥280 train images, `has_seg = True`, ≥3 categories

**Problem 1.2 — Mask visualisation spot check**
Randomly pick 5 images from val set and overlay their GT masks. Each class gets a distinct colour.
```python
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

coco_val = COCO("dataset/annotations/val.json")
img_ids  = random.sample(coco_val.getImgIds(), 5)

CLASS_COLOURS = {
    "arm":           [255, 80,  80],
    "leg":           [80,  160, 255],
    "torso":         [80,  220, 120],
    "head":          [255, 200, 50],
}

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for col, img_id in enumerate(img_ids):
    img_info = coco_val.loadImgs(img_id)[0]
    image    = np.array(Image.open(f"dataset/images/val/{img_info['file_name']}").convert("RGB"))
    ann_ids  = coco_val.getAnnIds(imgIds=img_id)
    anns     = coco_val.loadAnns(ann_ids)

    overlay = image.copy()
    for ann in anns:
        mask     = coco_val.annToMask(ann)
        cat_name = coco_val.loadCats(ann["category_id"])[0]["name"]
        colour   = CLASS_COLOURS.get(cat_name, [128, 128, 128])
        overlay[mask == 1] = (np.array(overlay[mask == 1]) * 0.4 + np.array(colour) * 0.6).astype(np.uint8)

    axes[0, col].imshow(image)
    axes[0, col].set_title(f"Image {col+1}", fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(overlay)
    axes[1, col].set_title(f"{len(anns)} masks", fontsize=9)
    axes[1, col].axis("off")

legend_patches = [mpatches.Patch(color=[c/255 for c in v], label=k) for k, v in CLASS_COLOURS.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9)
plt.suptitle("Phase 1 — GT mask alignment check (top: raw, bottom: masked)", fontsize=11)
plt.tight_layout()
plt.savefig("viz/phase1_mask_alignment.png", dpi=150, bbox_inches="tight")
plt.show()
```
✅ Pass: coloured overlays land precisely on the correct robot parts in at least 4/5 images. If a mask covers the wrong region entirely, the annotation export pipeline is broken — fix before proceeding.

**Problem 1.3 — Class balance check**
```python
import matplotlib.pyplot as plt
from collections import Counter

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
cat_counts     = Counter(a["category_id"] for a in coco["annotations"])
names  = [cat_id_to_name[k] for k in sorted(cat_counts)]
counts = [cat_counts[k] for k in sorted(cat_counts)]
total  = sum(counts)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

bars = axes[0].bar(names, counts, color="steelblue", edgecolor="white")
axes[0].set_title("Annotation count per class")
axes[0].set_ylabel("Count")
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(count), ha="center", va="bottom", fontsize=9)

axes[1].pie(counts, labels=names, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Class distribution")

# Imbalance warning line
max_pct = max(counts) / total * 100
print(f"Most dominant class: {max_pct:.1f}% of all annotations")
if max_pct > 60:
    print("⚠️  WARNING: class imbalance detected — consider collecting more data for minority classes")

plt.tight_layout()
plt.savefig("viz/phase1_class_balance.png", dpi=150, bbox_inches="tight")
plt.show()
```
✅ Pass: no class exceeds 60% of annotations. If one class dominates, collect more images of other classes or merge rare classes.

**Problem 1.4 — DataLoader smoke test**
```python
from torch.utils.data import DataLoader
# Build a simple dataset class that loads image + mask
# Run one batch through without errors
loader = DataLoader(your_dataset, batch_size=2)
images, masks = next(iter(loader))
print(images.shape, masks.shape)  # (2, 3, 1024, 1024), (2, H, W)
```
✅ Pass: shapes are correct, no errors, loads in <5 seconds per batch

### Phase 1 pass threshold
All four problems pass. Estimated time: **3–5 days** (annotation is the bottleneck)

---

## Phase 2 — Baseline Evaluations

**What you build:** Quantified mIoU numbers for (1) zero-shot SAM2 and (2) your original ViT model, on your test set. These are your comparison baselines.

### mIoU definition
For each image, compute IoU between predicted mask and GT mask per class, then average across all classes and images.

```python
def compute_miou(pred_masks, gt_masks, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_masks == cls)
        gt_cls   = (gt_masks == cls)
        intersection = (pred_cls & gt_cls).sum()
        union        = (pred_cls | gt_cls).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious)
```

### Chapter problems

**Problem 2.1 — Zero-shot SAM2 mIoU**
For each test image:
- Use the centroid of the GT mask bounding box as the point prompt
- Run SAM2 inference
- Compute mIoU against GT mask

```python
# Log results to a CSV: image_id, class, iou, inference_time_ms
```
✅ Pass: you have a number. Expected range: 30–55% mIoU. Write it down — this is your floor.

**Problem 2.2 — ViT baseline mIoU**
Load your existing ViT segmentation model. Run inference on the same test set. Compute mIoU using the same function.

✅ Pass: you have a number. Expected range: 55–70% mIoU.

**Problem 2.3 — Inference latency measurement**
```python
import time

times = []
for image in test_set:
    start = time.perf_counter()
    run_inference(image)
    times.append(time.perf_counter() - start)

print(f"Mean latency: {sum(times)/len(times)*1000:.1f} ms")
print(f"P95 latency:  {sorted(times)[int(0.95*len(times))]*1000:.1f} ms")
```
✅ Pass: you have latency numbers for both SAM2 zero-shot and ViT. Record them.

**Problem 2.4 — Failure mode analysis with visualisation**
Find the 10 worst-performing test images and display them in a grid with their IoU scores and predicted vs GT masks.
```python
import matplotlib.pyplot as plt

# results_df: pandas DataFrame with columns [image_id, image_path, iou, pred_mask, gt_mask]
worst_10 = results_df.nsmallest(10, "iou")

fig, axes = plt.subplots(10, 3, figsize=(12, 40))

for row, (_, record) in enumerate(worst_10.iterrows()):
    image   = np.array(Image.open(record["image_path"]).convert("RGB"))
    gt_mask = record["gt_mask"]
    pr_mask = record["pred_mask"]

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"Input (IoU: {record['iou']:.2f})", fontsize=8)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(image)
    axes[row, 1].imshow(gt_mask, alpha=0.5, cmap="Greens")
    axes[row, 1].set_title("GT mask", fontsize=8)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(image)
    axes[row, 2].imshow(pr_mask, alpha=0.5, cmap="Reds")
    axes[row, 2].set_title("SAM2 prediction", fontsize=8)
    axes[row, 2].axis("off")

plt.suptitle("Phase 2 — 10 worst zero-shot SAM2 predictions\n(green = GT, red = predicted)", fontsize=11)
plt.tight_layout()
plt.savefig("viz/phase2_failure_modes.png", dpi=120, bbox_inches="tight")
plt.show()

# Also plot the full IoU distribution
fig2, ax = plt.subplots(figsize=(8, 4))
ax.hist(results_df["iou"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(results_df["iou"].mean(), color="red", linestyle="--", label=f"Mean mIoU: {results_df['iou'].mean():.3f}")
ax.set_xlabel("IoU per image")
ax.set_ylabel("Count")
ax.set_title("Zero-shot SAM2 — IoU distribution on test set")
ax.legend()
plt.tight_layout()
plt.savefig("viz/phase2_iou_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
```
After looking at the worst 10, write 2–3 sentences in a markdown comment in your notebook about the failure pattern (e.g. "metallic reflections cause boundary confusion", "occluded heads get merged with torso").

✅ Pass: failure grid is saved, IoU distribution plotted, failure pattern written in words. These two observations directly inform your augmentation choices in Phase 3.

### Phase 2 pass threshold
You have four numbers: zero-shot mIoU, ViT mIoU, zero-shot latency, ViT latency. Failure patterns documented. Estimated time: **1–2 days**

---

## Phase 3 — Adapter Implementation

**What you build:** The `AdapterBlock` module inserted into SAM2's image encoder, with all training safeguards in place.

### Architecture target
```
Frozen image encoder (Hiera-L, ~300M params)
  └── AdapterBlock after each transformer block (~1.5M params total, trainable)
Frozen prompt encoder
Trainable mask decoder (~4M params)
─────────────────────────────────────────────
Total trainable: ~5.5M params (~1.8% of model)
```

### AdapterBlock implementation
```python
import torch.nn as nn

class AdapterBlock(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, dim)
        # Critical: initialise up to zero so adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))
```

### Training config
```python
# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze adapters
for adapter in adapters:
    for param in adapter.parameters():
        param.requires_grad = True

# Unfreeze mask decoder (lower LR than adapters)
for param in model.mask_decoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {"params": adapter_params,  "lr": 1e-4},
    {"params": decoder_params,  "lr": 5e-5},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

### Loss function
```python
# Focal loss + Dice loss (standard for medical/robotics segmentation)
def focal_loss(pred, target, gamma=2.0):
    bce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt   = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()

def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    return focal_loss(pred, target) + dice_loss(pred, target)
```

### Chapter problems

**Problem 3.1 — Parameter count verification**
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
```
✅ Pass: trainable ≤ 6M params, percentage ≤ 2%

**Problem 3.2 — Frozen encoder verification**
```python
# Run one forward pass, check that encoder gradients are None
loss = combined_loss(model(sample_image), sample_mask)
loss.backward()

for name, param in model.image_encoder.named_parameters():
    if "adapter" not in name:
        assert param.grad is None, f"Encoder weight {name} has gradient — it's not frozen!"
print("Encoder correctly frozen")
```
✅ Pass: assertion does not fire, prints confirmation

**Problem 3.3 — Identity initialisation check**
```python
# Before any training, adapter output should equal its input
adapter = AdapterBlock(dim=256)
x = torch.randn(1, 100, 256)
out = adapter(x)
print("Max diff (should be ~0):", (out - x).abs().max().item())
```
✅ Pass: max diff < 1e-6

**Problem 3.4 — One epoch smoke test**
Run 1 full training epoch. Check:
```python
# Loss should decrease from step 1 to step N within the epoch
# No NaN values
# GPU memory stays under 35 GB
assert not torch.isnan(loss), "NaN loss detected"
print(f"Epoch 1 loss: {epoch_loss:.4f}")
print(f"GPU memory:   {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```
✅ Pass: no NaN, loss is a real number, memory ≤ 35 GB

**Problem 3.5 — Overfitting probe with loss curve**
Train for 5 epochs on just 20 images (intentional overfit). Plot the loss curve and visually verify predictions on those same 20 images.
```python
import matplotlib.pyplot as plt

overfit_losses = []  # collect loss per step during the 5-epoch overfit run

# ... (run 5 epochs on 20 images, appending loss each step) ...

# Loss curve
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(overfit_losses)
axes[0].set_title("Overfit probe — training loss (should drop sharply)")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].axhline(0.1, color="green", linestyle="--", label="Target < 0.1")
axes[0].legend()

# Prediction grid on the 20 training images
# Pick 4 at random and show: image | GT | prediction
sample_ids = random.sample(range(20), 4)
for i, idx in enumerate(sample_ids):
    image, gt_mask = overfit_dataset[idx]
    pred_mask = run_inference(model, image)

    col_offset = i * 3
    # (use a larger grid if needed; this is the pattern)
    print(f"Sample {idx} — pred IoU vs GT: {compute_iou(pred_mask, gt_mask):.3f}")

# Check train mIoU
train_miou_overfit = evaluate_miou(model, overfit_loader)
print(f"Train mIoU on 20 images after 5 epochs: {train_miou_overfit:.3f}")

axes[1].bar(["Train mIoU (20 imgs)"], [train_miou_overfit], color="steelblue")
axes[1].axhline(0.80, color="green", linestyle="--", label="Pass threshold: 0.80")
axes[1].set_ylim(0, 1)
axes[1].set_title("Overfit probe — mIoU on training images")
axes[1].legend()

plt.tight_layout()
plt.savefig("viz/phase3_overfit_probe.png", dpi=150, bbox_inches="tight")
plt.show()
```
✅ Pass: training mIoU on those 20 images exceeds 80% after 5 epochs AND loss curve shows a clear downward trend. If loss doesn't drop, there is a bug in the loss function or optimiser setup — do not proceed to full training.

### Phase 3 pass threshold
All five problems pass. Estimated time: **3–5 days**

---

## Phase 4 — Full Training Run

**What you build:** A fully trained PEFT adapter model with training curves, evaluated on the test set.

### Training loop (complete)
```python
best_val_miou = 0
patience = 10
epochs_without_improvement = 0

for epoch in range(50):
    # --- Training ---
    model.train()
    train_losses = []
    for images, masks, prompts in train_loader:
        optimizer.zero_grad()
        preds = model(images, prompts)
        loss  = combined_loss(preds, masks)
        # L2 regularisation on adapter weights
        l2 = sum(p.norm(2) for p in adapter_params)
        loss = loss + 0.01 * l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validation ---
    model.eval()
    val_miou = evaluate_miou(model, val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1:3d} | Train loss: {sum(train_losses)/len(train_losses):.4f} | Val mIoU: {val_miou:.4f}")

    # --- Early stopping ---
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        torch.save(model.state_dict(), "best_model.pt")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
```

### Chapter problems

**Problem 4.1 — Training curve visualisation**
After training completes, plot all four signals together:
```python
import matplotlib.pyplot as plt

# training_log: list of dicts with keys epoch, train_loss, val_loss, train_miou, val_miou
epochs      = [r["epoch"]      for r in training_log]
train_loss  = [r["train_loss"] for r in training_log]
val_loss    = [r["val_loss"]   for r in training_log]
train_miou  = [r["train_miou"] for r in training_log]
val_miou    = [r["val_miou"]   for r in training_log]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(epochs, train_loss, label="Train loss", color="steelblue")
axes[0].plot(epochs, val_loss,   label="Val loss",   color="coral", linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & validation loss")
axes[0].legend()
axes[0].axvline(best_epoch, color="green", linestyle=":", alpha=0.6, label=f"Best epoch: {best_epoch}")

# mIoU plot
axes[1].plot(epochs, train_miou, label="Train mIoU", color="steelblue")
axes[1].plot(epochs, val_miou,   label="Val mIoU",   color="coral", linestyle="--")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mIoU")
axes[1].set_title("Train vs val mIoU")
axes[1].legend()
axes[1].axvline(best_epoch, color="green", linestyle=":", alpha=0.6)
axes[1].set_ylim(0, 1)

# Compute and annotate the train-val gap at best epoch
gap = train_miou[best_epoch - 1] - val_miou[best_epoch - 1]
axes[1].annotate(f"Gap: {gap:.2%}", xy=(best_epoch, val_miou[best_epoch - 1]),
                 xytext=(best_epoch + 2, val_miou[best_epoch - 1] - 0.05),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

plt.suptitle("Phase 4 — PEFT adapter training curves", fontsize=12)
plt.tight_layout()
plt.savefig("viz/phase4_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Best val mIoU: {max(val_miou):.4f} at epoch {best_epoch}")
print(f"Train-val gap at best epoch: {gap:.2%}")
```
✅ Pass: val mIoU plateau is visible, gap between train and val mIoU < 10 percentage points at the point of early stopping. If you see train mIoU at 90% and val at 60%, you are overfitting — reduce epochs or increase augmentation.

**Problem 4.2 — Test set evaluation**
Load `best_model.pt` and run on the held-out test set (never used during training or validation).
```python
model.load_state_dict(torch.load("best_model.pt"))
test_miou = evaluate_miou(model, test_loader)
print(f"PEFT adapter test mIoU: {test_miou:.4f}")
```
✅ Pass: test mIoU > val mIoU × 0.95 (test performance should not be dramatically worse than val — if it is, you have data leakage or distribution issues)

**Problem 4.3 — Per-class breakdown with bar chart**
```python
import matplotlib.pyplot as plt
import numpy as np

per_class_iou = evaluate_per_class_miou(model, test_loader)
class_names   = list(per_class_iou.keys())
iou_values    = list(per_class_iou.values())

fig, ax = plt.subplots(figsize=(10, 5))
bar_colours = ["green" if v >= 0.60 else "orange" if v >= 0.30 else "red" for v in iou_values]
bars = ax.barh(class_names, iou_values, color=bar_colours, edgecolor="white")

ax.axvline(0.30, color="red",    linestyle="--", linewidth=1, label="Fail threshold (0.30)")
ax.axvline(0.60, color="orange", linestyle="--", linewidth=1, label="OK threshold (0.60)")
ax.axvline(np.mean(iou_values), color="black", linestyle="-",  linewidth=1.5,
           label=f"Mean mIoU: {np.mean(iou_values):.3f}")

for bar, val in zip(bars, iou_values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax.set_xlim(0, 1.0)
ax.set_xlabel("IoU")
ax.set_title("Phase 4 — PEFT adapter per-class IoU on test set")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("viz/phase4_per_class_iou.png", dpi=150, bbox_inches="tight")
plt.show()

for cls_name, iou in zip(class_names, iou_values):
    status = "✅" if iou >= 0.60 else "⚠️" if iou >= 0.30 else "❌"
    print(f"  {status} {cls_name:20s}: {iou:.3f}")
```
✅ Pass: no class has IoU < 0.30 (red bar). If one class is red, check how many training examples it has — likely needs more data.

**Problem 4.4 — Qualitative prediction grid**
Pick one test image per class. Show input | GT mask | PEFT prediction in a single figure.
```python
import matplotlib.pyplot as plt

# one_per_class: list of (image, gt_mask, class_name) tuples — one per class
fig, axes = plt.subplots(len(one_per_class), 3, figsize=(12, 4 * len(one_per_class)))

for row, (image, gt_mask, class_name) in enumerate(one_per_class):
    pred_mask = run_inference(model, image)
    iou       = compute_iou(pred_mask, gt_mask)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"Class: {class_name}", fontsize=9)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(image)
    axes[row, 1].imshow(gt_mask, alpha=0.5, cmap="Greens")
    axes[row, 1].set_title("Ground truth", fontsize=9)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(image)
    axes[row, 2].imshow(pred_mask, alpha=0.5, cmap="Reds")
    axes[row, 2].set_title(f"PEFT prediction (IoU: {iou:.2f})", fontsize=9)
    axes[row, 2].axis("off")

plt.suptitle("Phase 4 — Qualitative results: GT (green) vs PEFT prediction (red)", fontsize=11)
plt.tight_layout()
plt.savefig("viz/phase4_qualitative_grid.png", dpi=150, bbox_inches="tight")
plt.show()
```
✅ Pass: predicted masks (red) substantially overlap with GT masks (green) in at least 4 out of 5 rows. You should be able to look at each pair and say "yes, that's a plausible segmentation of a robot arm / leg / head."

### Phase 4 pass threshold
Test mIoU number in hand, training curves plotted, per-class breakdown documented. Estimated time: **2–3 days**

---

## Phase 5 — Full Fine-tune Comparison

**What you build:** A fully fine-tuned SAM2 (all weights unfrozen) trained on the same data, for the upper-bound comparison.

> Note: this is compute-heavy. Run it once, save the checkpoint, never retrain.

### Key difference from Phase 4
```python
# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# Use lower LR for the encoder (protect pre-trained knowledge somewhat)
optimizer = torch.optim.AdamW([
    {"params": model.image_encoder.parameters(), "lr": 1e-5},
    {"params": model.prompt_encoder.parameters(), "lr": 1e-5},
    {"params": model.mask_decoder.parameters(),   "lr": 5e-5},
], weight_decay=1e-4)
```

### Chapter problems

**Problem 5.1 — Full fine-tune test mIoU**
Same evaluation as Problem 4.2, but with the full fine-tune model.
```python
full_ft_test_miou = evaluate_miou(full_ft_model, test_loader)
print(f"Full fine-tune test mIoU: {full_ft_test_miou:.4f}")
```
✅ Pass: full fine-tune mIoU ≥ PEFT adapter mIoU (if PEFT beats full fine-tune, something is wrong — check LR config)

**Problem 5.2 — PEFT recovery ratio**
```python
zero_shot_miou  = 0.XX   # from Phase 2
peft_miou       = 0.XX   # from Phase 4
full_ft_miou    = 0.XX   # from Phase 5

recovery = (peft_miou - zero_shot_miou) / (full_ft_miou - zero_shot_miou)
print(f"PEFT recovery ratio: {recovery:.2%}")
```
✅ Pass: recovery ratio ≥ 85%. This is the core claim of the project — that PEFT recovers most of the full fine-tune gain with 1–2% of the parameters.

**Problem 5.3 — Out-of-domain comparison grid**
Take 5 images of everyday objects (chairs, cups, hands). Run both PEFT and full fine-tune models. Show side by side.
```python
import matplotlib.pyplot as plt

ood_images  = [...]   # 5 out-of-domain images
ood_classes = ["chair", "cup", "hand", "bottle", "shoe"]

fig, axes = plt.subplots(5, 3, figsize=(12, 20))

for row, (image, label) in enumerate(zip(ood_images, ood_classes)):
    peft_mask  = run_inference(peft_model,  image)
    full_mask  = run_inference(full_ft_model, image)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"Input: {label}", fontsize=9)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(image)
    axes[row, 1].imshow(peft_mask, alpha=0.5, cmap="Blues")
    axes[row, 1].set_title("PEFT model", fontsize=9)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(image)
    axes[row, 2].imshow(full_mask, alpha=0.5, cmap="Reds")
    axes[row, 2].set_title("Full fine-tune model", fontsize=9)
    axes[row, 2].axis("off")

plt.suptitle("Phase 5 — Out-of-domain check: PEFT (blue) vs full fine-tune (red)\n"
             "Better result = more coherent mask on non-robot object", fontsize=10)
plt.tight_layout()
plt.savefig("viz/phase5_ood_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```
✅ Pass: PEFT model produces more coherent masks on out-of-domain objects than the full fine-tune model, or at worst equal. Document any case where full fine-tune produces a clearly broken mask — that is your catastrophic forgetting evidence.

### Visualisation 5.4 — Final summary chart (all four methods)
After Problem 5.2 gives you all numbers, produce the summary chart that goes in your README:
```python
import matplotlib.pyplot as plt
import numpy as np

methods    = ["Zero-shot\nSAM2", "ViT\nbaseline", "SAM2 PEFT\n(ours)", "Full\nfine-tune"]
miou       = [zero_shot_miou, vit_miou, peft_miou, full_ft_miou]
params_m   = [0, 86, 5.5, 304]       # trainable params in millions
latency_ms = [zs_lat, vit_lat, peft_lat, fft_lat]

x     = np.arange(len(methods))
width = 0.5
colours = ["#aaaaaa", "#5599cc", "#22aa66", "#dd4444"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# mIoU bars
bars = axes[0].bar(x, miou, width, color=colours, edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(methods, fontsize=9)
axes[0].set_ylabel("mIoU"); axes[0].set_ylim(0, 1)
axes[0].set_title("Test mIoU")
for bar, val in zip(bars, miou):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

# Params (log scale)
bars2 = axes[1].bar(x, [max(p, 0.01) for p in params_m], width, color=colours, edgecolor="white")
axes[1].set_yscale("log"); axes[1].set_xticks(x); axes[1].set_xticklabels(methods, fontsize=9)
axes[1].set_ylabel("Trainable parameters (M, log scale)")
axes[1].set_title("Parameter efficiency")
for bar, val in zip(bars2, params_m):
    axes[1].text(bar.get_x() + bar.get_width()/2, max(val, 0.01) * 1.5,
                 f"{val}M" if val > 0 else "0", ha="center", va="bottom", fontsize=9)

# Latency
bars3 = axes[2].bar(x, latency_ms, width, color=colours, edgecolor="white")
axes[2].set_xticks(x); axes[2].set_xticklabels(methods, fontsize=9)
axes[2].set_ylabel("Inference latency (ms/image)")
axes[2].set_title("Inference speed")
for bar, val in zip(bars3, latency_ms):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.0f}ms",
                 ha="center", va="bottom", fontsize=9)

plt.suptitle("SAM2 PEFT Robotics — Full comparison across all methods", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("viz/phase5_final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to viz/phase5_final_comparison.png — use this in your README")
```
✅ Pass: all three subplots have real numbers, the PEFT bar is green and sits close to the full fine-tune bar in mIoU while being dramatically shorter in the params chart. This is the figure that tells your project's story at a glance.

### Phase 5 pass threshold
Recovery ratio ≥ 85%. Out-of-domain comparison documented. Estimated time: **1–2 days**

---

## Phase 6 — Results Table & Write-up

**What you build:** The final comparison table and a concise project description you can put on your resume and GitHub README.

### Final results table

| Method | Test mIoU | Params Trained | Latency (ms) | Notes |
|---|---|---|---|---|
| Zero-shot SAM2 | XX% | 0 | XX ms | No training |
| ViT baseline | XX% | ~86M | XX ms | Your original project |
| **SAM2 PEFT (ours)** | **XX%** | **~5.5M** | **XX ms** | **This project** |
| Full fine-tune SAM2 | XX% | ~304M | XX ms | Upper bound |

### Chapter problems

**Problem 6.1 — All four rows filled**
Every cell in the table above has a real number from your experiments.
✅ Pass: no cells say "TBD"

**Problem 6.2 — Resume bullet passes the "so what" test**
Write a single bullet point for your resume:
```
Adapted SAM2 (300M params) to robot component segmentation via PEFT adapters — 
trained only 5.5M parameters (<2%) while recovering [X]% of full fine-tune mIoU 
([Y]% vs [Z]%); compared four methods across mIoU, latency, and parameter efficiency.
```
Fill in X, Y, Z from your numbers. Read it aloud. Does it communicate a result, not just a process?
✅ Pass: the bullet has three real numbers in it

**Problem 6.3 — GitHub README contains**
- [ ] Problem statement (2–3 sentences)
- [ ] Architecture diagram (you can screenshot the diagram from this conversation)
- [ ] Results table
- [ ] Instructions to reproduce (dataset download, training command, evaluation command)
- [ ] Requirements file (`pip install -r requirements.txt` works)

✅ Pass: someone else could clone the repo and reproduce your numbers

### Phase 6 pass threshold
Table complete, resume bullet written, README done. Estimated time: **1–2 days**

---

## Summary timeline

| Phase | Deliverable | Key metric | Estimated time |
|---|---|---|---|
| 0 | Working Colab environment | SAM2 runs on one image | 2–3 hours |
| 1 | COCO dataset, annotated | ≥280 train images, masks correct | 3–5 days |
| 2 | Baseline numbers | Zero-shot mIoU + ViT mIoU recorded | 1–2 days |
| 3 | Adapter implementation | ≤2% params trainable, overfit probe passes | 3–5 days |
| 4 | Full training run | Test mIoU in hand | 2–3 days |
| 5 | Full fine-tune comparison | Recovery ratio ≥ 85% | 1–2 days |
| 6 | Write-up | Table + resume bullet + README | 1–2 days |

**Total: ~2.5–3.5 weeks of focused work**

---

## Working with Claude Code

When you open this file in Claude Code, each phase maps to a conversation. Suggested prompts:

- **Phase 0:** `"Help me set up the Colab environment for SAM2 and run Problem 0.1–0.3"`
- **Phase 1:** `"I have images from Roboflow. Help me write the DataLoader for COCO-format segmentation masks and run Problem 1.4"`
- **Phase 3:** `"Help me implement AdapterBlock and insert it into SAM2's image encoder. Then run Problem 3.1–3.5"`
- **Phase 4:** `"My Phase 3 checks all pass. Help me write the full training loop with early stopping and L2 regularisation"`
- **Debugging:** `"Problem 3.5 failed — training mIoU on 20 images is only 40% after 5 epochs. Here is my loss function code: [paste]"`

Paste your problem outputs (the print statements) directly into the conversation — they are your debugging signal.
