from __future__ import annotations

"""Phase 5 — full SAM2 fine-tune with all training optimizations:
  - RAM image cache (preload all images as tensors before training)
  - Async prefetch (CPU loads next image while GPU processes current)
  - Mixed precision BF16 (2x faster tensor core utilisation)
  - torch.compile (kernel fusion, optional)
  - Batched decoder (stack annotations per image into one GPU call)
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.checkpoint as grad_ckpt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam2_peft.phase3 import (
    CocoPromptInstanceDataset,
    PromptSample,
    binary_iou,
    combined_loss,
    load_sam2_model,
    low_res_target,
    make_transforms,
    predict_instance_mask,
    resolve_device,
)


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model) -> int:
    trunk = model.image_encoder.trunk
    wrapped = 0
    for block in trunk.blocks:
        original_forward = block.forward

        def make_checkpointed(fwd):
            def checkpointed_forward(x):
                return grad_ckpt.checkpoint(fwd, x, use_reentrant=False)
            return checkpointed_forward

        block.forward = make_checkpointed(original_forward)
        wrapped += 1
    return wrapped


# ---------------------------------------------------------------------------
# Unfreeze all parameters
# ---------------------------------------------------------------------------

def configure_full_finetune(model) -> tuple[list, int]:
    for p in model.parameters():
        p.requires_grad_(True)
    all_params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in all_params)
    return all_params, total


# ---------------------------------------------------------------------------
# Optimization 1: RAM image cache
# ---------------------------------------------------------------------------

def build_ram_cache(
    dataset: CocoPromptInstanceDataset,
    transforms,
    image_groups: list[list[int]],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Pre-load and preprocess all training images into pinned CPU RAM.

    On second and subsequent epochs this eliminates all disk I/O and JPEG
    decode overhead from the hot loop — the tensor is already in RAM waiting
    for a non-blocking .to(device) call.
    """
    print("Building RAM image cache...", flush=True)
    t0 = time.time()
    cache: dict[int, torch.Tensor] = {}
    for ann_indices in image_groups:
        sample0 = dataset[ann_indices[0]]
        img_id  = sample0.image_id
        if img_id not in cache:
            tensor = transforms(sample0.image).unsqueeze(0)
            if device.type == "cuda":
                tensor = tensor.pin_memory()
            cache[img_id] = tensor
    elapsed = time.time() - t0
    size_gb = sum(t.nbytes for t in cache.values()) / 1e9
    print(f"RAM cache ready: {len(cache)} images, {size_gb:.2f} GB, {elapsed:.1f}s", flush=True)
    return cache


# ---------------------------------------------------------------------------
# Optimization 2: Async prefetch
# ---------------------------------------------------------------------------

class ImagePrefetcher:
    """Prefetch the next image tensor onto GPU while the current one is being processed.

    Uses a single background thread to call .to(device, non_blocking=True)
    on the next tensor, overlapping PCIe transfer with GPU compute.
    """

    def __init__(self, image_groups: list[list[int]], dataset, transforms,
                 ram_cache: dict | None, device: torch.device):
        self.groups    = image_groups
        self.dataset   = dataset
        self.transforms = transforms
        self.cache     = ram_cache
        self.device    = device
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future   = None
        self._idx      = 0

    def _load(self, ann_indices: list[int]) -> torch.Tensor:
        sample0 = self.dataset[ann_indices[0]]
        if self.cache is not None:
            cpu_tensor = self.cache[sample0.image_id]
        else:
            cpu_tensor = self.transforms(sample0.image).unsqueeze(0)
            if self.device.type == "cuda":
                cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.to(self.device, non_blocking=True)

    def prime(self):
        """Start prefetching the first image."""
        if self._idx < len(self.groups):
            self._future = self._executor.submit(self._load, self.groups[self._idx])

    def next(self) -> torch.Tensor:
        """Return current prefetched tensor and start fetching the next one."""
        tensor = self._future.result()
        self._idx += 1
        if self._idx < len(self.groups):
            self._future = self._executor.submit(self._load, self.groups[self._idx])
        return tensor

    def reset(self, shuffled_groups: list[list[int]]):
        self.groups = shuffled_groups
        self._idx   = 0
        self.prime()

    def shutdown(self):
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Optimization 3: Batched decoder
# ---------------------------------------------------------------------------

def decode_batch_from_features(
    model, transforms,
    backbone_out: dict,
    samples: list[PromptSample],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> list[torch.Tensor]:
    """Run prompt encoder + mask decoder for all annotations of one image in one call.

    Batches all N annotation point prompts together so the mask decoder runs
    once instead of N times, removing N-1 kernel launch round-trips.
    """
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *fs)
        for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]

    # Build batch of point prompts — one set per annotation
    all_coords, all_labels = [], []
    orig_hw = samples[0].image.shape[:2]
    for s in samples:
        coords = torch.as_tensor(s.point_coords, dtype=torch.float32, device=device)
        coords = transforms.transform_coords(coords, normalize=True, orig_hw=orig_hw)
        all_coords.append(coords)
        all_labels.append(torch.ones(coords.shape[0], dtype=torch.int64, device=device))

    # Pad to same length so we can stack
    max_pts = max(c.shape[0] for c in all_coords)
    padded_coords  = torch.zeros(len(samples), max_pts, 2, device=device)
    padded_labels  = torch.full((len(samples), max_pts), -1, dtype=torch.int64, device=device)
    for i, (c, l) in enumerate(zip(all_coords, all_labels)):
        padded_coords[i, :c.shape[0]] = c
        padded_labels[i, :l.shape[0]] = l

    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else _nullctx()
    with ctx:
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=(padded_coords, padded_labels), boxes=None, masks=None,
        )
        # repeat image features N times for the batch
        img_emb = feats[-1].expand(len(samples), -1, -1, -1)
        hf      = [f.expand(len(samples), -1, -1, -1) for f in feats[:-1]]
        low_res_masks, _, _, _ = model.sam_mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hf,
        )
    # unsqueeze to (1, H, W) to match low_res_target output shape
    return [low_res_masks[i, 0].unsqueeze(0) for i in range(len(samples))]


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Grouped dataset helper
# ---------------------------------------------------------------------------

def grouped_by_image(dataset: CocoPromptInstanceDataset) -> list[list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(dataset)):
        groups[dataset[i].image_id].append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Evaluation (always fp32, no_grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, transforms, dataset, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    per_class: dict[str, list[float]] = {}
    for ann_indices in grouped_by_image(dataset):
        sample0      = dataset[ann_indices[0]]
        input_tensor = transforms(sample0.image).unsqueeze(0).to(device)
        backbone_out = model.forward_image(input_tensor)
        _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *fs)
            for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        for idx in ann_indices:
            sample = dataset[idx]
            coords = torch.as_tensor(sample.point_coords, dtype=torch.float32, device=device)
            coords = transforms.transform_coords(coords, normalize=True, orig_hw=sample.image.shape[:2])
            labels = torch.ones(coords.shape[0], dtype=torch.int64, device=device)
            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=(coords.unsqueeze(0), labels.unsqueeze(0)), boxes=None, masks=None,
            )
            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=feats[-1],
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False, repeat_image=False, high_res_features=feats[:-1],
            )
            logits   = low_res_masks[:, 0].unsqueeze(1)
            upsampled = transforms.postprocess_masks(logits, sample.image.shape[:2])
            pred = (upsampled[0, 0] > 0).detach().cpu().numpy()
            iou  = binary_iou(pred, sample.mask)
            if not math.isnan(iou):
                per_class.setdefault(sample.class_name, []).append(iou)
    class_means = {cls: float(np.mean(ious)) for cls, ious in per_class.items()}
    miou = float(np.mean(list(class_means.values()))) if class_means else 0.0
    model.train()
    return miou, class_means


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_training_curves(path: Path, log: list[dict]) -> None:
    epochs     = [r["epoch"] for r in log]
    train_loss = [r["train_loss"] for r in log]
    val_miou   = [r["val_miou"] for r in log]
    best_epoch = max(log, key=lambda r: r["val_miou"])["epoch"]
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_loss, color="steelblue")
    axes[0].axvline(best_epoch, color="green", linestyle=":", alpha=0.7, label=f"Best epoch {best_epoch}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training loss"); axes[0].legend()
    axes[1].plot(epochs, val_miou, color="coral")
    axes[1].axvline(best_epoch, color="green", linestyle=":", alpha=0.7)
    axes[1].axhline(max(val_miou), color="coral", linestyle="--", alpha=0.4, label=f"Best: {max(val_miou):.3f}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU"); axes[1].set_title("Val mIoU")
    axes[1].set_ylim(0, 1); axes[1].legend()
    plt.suptitle("Phase 5 — full fine-tune training curves", fontsize=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {path}")


def save_per_class_chart(path: Path, test_per_class: dict, peft_per_class: dict | None = None) -> None:
    names  = sorted(test_per_class.keys())
    values = [test_per_class[n] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    x, w    = np.arange(len(names)), 0.35
    bars1   = ax.bar(x, values, w, label="Full fine-tune", color="steelblue", edgecolor="white")
    if peft_per_class:
        ax.bar(x + w, [peft_per_class.get(n, 0) for n in names], w,
               label="PEFT-SAM2", color="coral", edgecolor="white")
    for bar, v in zip(bars1, values):
        ax.text(bar.get_x() + bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + w/2); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1); ax.set_ylabel("IoU")
    ax.set_title("Phase 5 — Full fine-tune vs PEFT per-class IoU"); ax.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {path}")


def save_comparison_chart(path: Path, summary: dict) -> None:
    methods  = ["Zero-shot SAM2", "ViT-B/16", "PEFT-SAM2", "Full fine-tune"]
    mious    = [0.2178, 0.4223, summary["peft_test_miou"], summary["test_miou"]]
    colours  = ["#aaa", "#5588cc", "#ee8855", "#44aa77"]
    fig, ax  = plt.subplots(figsize=(10, 5))
    bars     = ax.bar(methods, mious, color=colours, edgecolor="white", width=0.5)
    for bar, v in zip(bars, mious):
        ax.text(bar.get_x() + bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("mIoU"); ax.set_title("All-method comparison — test mIoU")
    recovery = (summary["peft_test_miou"] - 0.2178) / max(summary["test_miou"] - 0.2178, 1e-6) * 100
    ax.text(0.99, 0.02, f"PEFT recovery ratio: {recovery:.1f}%",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {path}")


def save_qualitative_grid(path: Path, model, transforms, dataset, device, n_per_class=1) -> None:
    by_class: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        by_class.setdefault(dataset[i].class_name, []).append(i)
    rows = [(cls, random.choice(idxs)) for cls, idxs in sorted(by_class.items())]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4*len(rows)))
    if len(rows) == 1: axes = [axes]
    model.eval()
    with torch.no_grad():
        for row, (cls, idx) in enumerate(rows):
            sample = dataset[idx]
            pred   = predict_instance_mask(model, transforms, sample, device)
            iou    = binary_iou(pred, sample.mask)
            axes[row][0].imshow(sample.image); axes[row][0].set_title(f"Class: {cls}", fontsize=9)
            axes[row][1].imshow(sample.image); axes[row][1].imshow(sample.mask, alpha=0.5, cmap="Greens")
            axes[row][1].set_title("GT mask", fontsize=9)
            axes[row][2].imshow(sample.image); axes[row][2].imshow(pred, alpha=0.5, cmap="Blues")
            axes[row][2].set_title(f"Full FT pred (IoU {iou:.2f})", fontsize=9)
            for ax in axes[row]: ax.axis("off")
    model.train()
    plt.suptitle("Phase 5 — GT (green) vs full fine-tune prediction (blue)", fontsize=11)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 — optimised full SAM2 fine-tune.")
    parser.add_argument("--dataset-root",    default=Path("dataset"),                type=Path)
    parser.add_argument("--model-id",        default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device",          default="auto", choices=("auto","cuda","mps","cpu"))
    parser.add_argument("--lr",              default=1e-5,  type=float)
    parser.add_argument("--weight-decay",    default=1e-4,  type=float)
    parser.add_argument("--max-grad-norm",   default=1.0,   type=float)
    parser.add_argument("--epochs",          default=30,    type=int)
    parser.add_argument("--patience",        default=8,     type=int)
    parser.add_argument("--val-every",       default=1,     type=int)
    parser.add_argument("--seed",            default=42,    type=int)
    parser.add_argument("--output-dir",      default=Path("outputs/phase5"), type=Path)
    parser.add_argument("--checkpoint-path", default=Path("outputs/phase5/best_model.pt"), type=Path)
    parser.add_argument("--peft-summary",    default=Path("outputs/phase4/summary.json"), type=Path)
    # Optimization flags
    parser.add_argument("--no-grad-ckpt",    action="store_true",  help="Disable gradient checkpointing (safe on 5090)")
    parser.add_argument("--no-ram-cache",    action="store_true",  help="Disable RAM image cache")
    parser.add_argument("--no-prefetch",     action="store_true",  help="Disable async prefetch")
    parser.add_argument("--no-amp",          action="store_true",  help="Disable BF16 mixed precision")
    parser.add_argument("--no-compile",      action="store_true",  help="Disable torch.compile")
    parser.add_argument("--no-batch-decode", action="store_true",  help="Disable batched decoder")
    parser.add_argument("--resume",          action="store_true")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device_name = resolve_device(args.device)
    device      = torch.device(device_name)
    use_amp     = not args.no_amp and device.type == "cuda"
    amp_dtype   = torch.bfloat16 if use_amp else None

    print(f"Device: {device_name}", flush=True)
    print(f"Optimizations: grad_ckpt={not args.no_grad_ckpt} | ram_cache={not args.no_ram_cache} | "
          f"prefetch={not args.no_prefetch} | amp_bf16={use_amp} | "
          f"compile={not args.no_compile} | batch_decode={not args.no_batch_decode}", flush=True)

    train_dataset = CocoPromptInstanceDataset(args.dataset_root, split="train")
    val_dataset   = CocoPromptInstanceDataset(args.dataset_root, split="val")
    test_dataset  = CocoPromptInstanceDataset(args.dataset_root, split="test")
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}", flush=True)

    model = load_sam2_model(args.model_id, device_name, mode="train")
    all_params, total_count = configure_full_finetune(model)
    print(f"Trainable: {total_count/1e6:.2f}M (100%)", flush=True)

    # Gradient checkpointing (skip on 5090 with enough VRAM)
    if not args.no_grad_ckpt and device.type == "cuda":
        n = enable_gradient_checkpointing(model)
        print(f"Gradient checkpointing enabled on {n} blocks", flush=True)

    # torch.compile — try it, warn if it fails
    if not args.no_compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile: enabled (reduce-overhead)", flush=True)
        except Exception as e:
            print(f"torch.compile: skipped ({e})", flush=True)

    optimizer  = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
    transforms = make_transforms(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build image groups once
    image_groups = grouped_by_image(train_dataset)
    n_images     = len(image_groups)

    # RAM cache — preload all training images as pinned tensors
    ram_cache = None
    if not args.no_ram_cache:
        ram_cache = build_ram_cache(train_dataset, transforms, image_groups, device)

    # Resume
    start_epoch = 1; best_val_miou = 0.0; epochs_no_improve = 0
    training_log: list[dict] = []
    if args.resume and args.checkpoint_path.exists():
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1; best_val_miou = ckpt["best_val_miou"]
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        training_log = ckpt.get("training_log", [])
        print(f"Resumed from epoch {ckpt['epoch']}  best={best_val_miou:.4f}", flush=True)

    all_step_losses: list[float] = []

    # Prefetcher — initialised once, reset each epoch
    prefetcher = None
    if not args.no_prefetch:
        prefetcher = ImagePrefetcher(image_groups, train_dataset, transforms, ram_cache, device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        random.shuffle(image_groups)
        t0 = time.time()
        ann_step = 0

        if prefetcher is not None:
            prefetcher.reset(image_groups)

        for img_idx, ann_indices in enumerate(image_groups, start=1):

            # --- get image tensor (from prefetcher or direct) ---
            if prefetcher is not None:
                input_tensor = prefetcher.next()
            elif ram_cache is not None:
                sample0 = train_dataset[ann_indices[0]]
                input_tensor = ram_cache[sample0.image_id].to(device, non_blocking=True)
            else:
                sample0 = train_dataset[ann_indices[0]]
                input_tensor = transforms(sample0.image).unsqueeze(0).to(device)

            # --- encoder forward with optional BF16 ---
            amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else _nullctx()
            with amp_ctx:
                backbone_out = model.forward_image(input_tensor)

            optimizer.zero_grad(set_to_none=True)
            samples = [train_dataset[i] for i in ann_indices]

            # --- batched or sequential decoder ---
            if not args.no_batch_decode:
                with amp_ctx:
                    logits_list = decode_batch_from_features(
                        model, transforms, backbone_out, samples, device, amp_dtype,
                    )
                image_loss = torch.tensor(0.0, device=device)
                for logits, sample in zip(logits_list, samples):
                    target = low_res_target(sample, logits.shape[-2:], device)
                    image_loss = image_loss + combined_loss(logits, target) / len(samples)
                if not torch.isnan(image_loss):
                    scaler.scale(image_loss).backward(retain_graph=False)
                    epoch_losses.append(float(image_loss.detach().cpu()))
                    all_step_losses.append(float(image_loss.detach().cpu()))
            else:
                image_losses: list[float] = []
                for sample in samples:
                    with amp_ctx:
                        coords = torch.as_tensor(sample.point_coords, dtype=torch.float32, device=device)
                        coords = transforms.transform_coords(coords, normalize=True, orig_hw=sample.image.shape[:2])
                        labels = torch.ones(coords.shape[0], dtype=torch.int64, device=device)
                        _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
                        if model.directly_add_no_mem_embed:
                            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
                        feats = [
                            feat.permute(1, 2, 0).view(1, -1, *fs)
                            for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])
                        ][::-1]
                        sparse_emb, dense_emb = model.sam_prompt_encoder(
                            points=(coords.unsqueeze(0), labels.unsqueeze(0)), boxes=None, masks=None,
                        )
                        low_res_masks, _, _, _ = model.sam_mask_decoder(
                            image_embeddings=feats[-1],
                            image_pe=model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False, repeat_image=False, high_res_features=feats[:-1],
                        )
                        logits = low_res_masks[:, 0]
                        target = low_res_target(sample, logits.shape[-2:], device)
                        loss   = combined_loss(logits, target) / len(samples)
                    if not torch.isnan(loss):
                        scaler.scale(loss).backward(retain_graph=True)
                        image_losses.append(float(loss.detach().cpu()) * len(samples))
                if image_losses:
                    mean_l = float(np.mean(image_losses))
                    epoch_losses.append(mean_l); all_step_losses.append(mean_l)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
            scaler.step(optimizer); scaler.update()

            ann_step += len(ann_indices)
            if img_idx % 50 == 0 or img_idx == n_images:
                last = epoch_losses[-1] if epoch_losses else float("nan")
                print(f"epoch={epoch}/{args.epochs} img={img_idx}/{n_images} ann={ann_step} loss={last:.6f}", flush=True)

        train_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        scheduler.step()
        elapsed = time.time() - t0

        val_miou, val_per_class = 0.0, {}
        if epoch % args.val_every == 0:
            val_miou, val_per_class = evaluate(model, transforms, val_dataset, device)

        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        print(f"Epoch {epoch:3d} | loss={train_loss_mean:.4f} | val_mIoU={val_miou:.4f} | "
              f"gpu={gpu_mem:.1f}GB | t={elapsed:.0f}s", flush=True)

        log_entry = {"epoch": epoch, "train_loss": train_loss_mean, "val_miou": val_miou,
                     "val_per_class": val_per_class, "gpu_mem_gb": gpu_mem, "elapsed_s": elapsed}
        training_log.append(log_entry)

        if val_miou > best_val_miou:
            best_val_miou = val_miou; epochs_no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "best_val_miou": best_val_miou, "epochs_no_improve": 0,
                        "training_log": training_log}, args.checkpoint_path)
            print(f"  -> New best val mIoU {best_val_miou:.4f} — saved", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

        with open(args.output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)
        with open(args.output_dir / "step_losses.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss"])
            writer.writeheader()
            for s, l in enumerate(all_step_losses, start=1):
                writer.writerow({"step": s, "loss": l})
        save_training_curves(Path("viz/phase5_training_curves.png"), training_log)

    if prefetcher:
        prefetcher.shutdown()

    # Post-training test evaluation
    print("\n=== Loading best checkpoint for test evaluation ===", flush=True)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_miou, test_per_class = evaluate(model, transforms, test_dataset, device)
    print(f"Test mIoU: {test_miou:.4f}", flush=True)
    for cls, iou in sorted(test_per_class.items()):
        print(f"  {'PASS' if iou >= 0.30 else 'FAIL'}  {cls}: {iou:.4f}", flush=True)

    peft_test_miou = 0.8059
    peft_per_class = {"arm": 0.8450, "leg": 0.7881, "torso": 0.7422, "head": 0.8482}
    if args.peft_summary.exists():
        with open(args.peft_summary) as f:
            pd = json.load(f)
        peft_test_miou = pd.get("test_miou", peft_test_miou)
        peft_per_class = pd.get("test_per_class", peft_per_class)

    recovery = (peft_test_miou - 0.2178) / max(test_miou - 0.2178, 1e-6) * 100

    save_per_class_chart(Path("viz/phase5_per_class_iou.png"), test_per_class, peft_per_class)
    save_qualitative_grid(Path("viz/phase5_qualitative_grid.png"), model, transforms, test_dataset, device)
    save_comparison_chart(Path("viz/phase5_all_methods_comparison.png"),
                          {"test_miou": test_miou, "peft_test_miou": peft_test_miou})

    best_epoch = max(training_log, key=lambda r: r["val_miou"])["epoch"]
    summary = {
        "best_epoch": best_epoch, "best_val_miou": max(r["val_miou"] for r in training_log),
        "test_miou": test_miou, "test_per_class": test_per_class,
        "total_epochs_run": len(training_log),
        "total_parameters": total_count, "trainable_parameters": total_count, "trainable_percent": 100.0,
        "peft_test_miou": peft_test_miou, "peft_per_class": peft_per_class,
        "zero_shot_miou": 0.2178, "peft_recovery_ratio_pct": recovery,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"\nPEFT recovery ratio: {recovery:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
