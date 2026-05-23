from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam2_peft.models.sam2_peft import (
    build_peft_optimizer,
    configure_peft_trainable,
    count_parameters,
    insert_hiera_adapters,
)
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
# Cached forward: encode image once, decode once per annotation
# ---------------------------------------------------------------------------

def encode_image(model, transforms, image: np.ndarray, device: torch.device):
    """Run the full image encoder (trunk + neck + conv projections) once."""
    input_tensor = transforms(image).unsqueeze(0).to(device)
    return model.forward_image(input_tensor)


def decode_from_features(
    model, transforms,
    backbone_out: dict,
    sample: PromptSample,
    device: torch.device,
) -> torch.Tensor:
    """Run prompt encoder + mask decoder using pre-computed backbone features."""
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *fs)
        for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
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
        multimask_output=False,
        repeat_image=False,
        high_res_features=feats[:-1],
    )
    return low_res_masks[:, 0]


def grouped_by_image(dataset: CocoPromptInstanceDataset) -> list[list[int]]:
    """Return sample indices grouped by image_id, in shuffleable image order."""
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(dataset)):
        groups[dataset[i].image_id].append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, transforms, dataset, device: torch.device) -> tuple[float, dict[str, float]]:
    """Evaluate using image-grouped encoder caching for speed."""
    model.eval()
    per_class: dict[str, list[float]] = {}
    groups = grouped_by_image(dataset)
    for ann_indices in groups:
        sample0 = dataset[ann_indices[0]]
        backbone_out = encode_image(model, transforms, sample0.image, device)
        for idx in ann_indices:
            sample = dataset[idx]
            logits = decode_from_features(model, transforms, backbone_out, sample, device)
            logits = logits.unsqueeze(1)
            upsampled = transforms.postprocess_masks(logits, sample.image.shape[:2])
            pred = (upsampled[0, 0] > 0).detach().cpu().numpy()
            iou = binary_iou(pred, sample.mask)
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, color="steelblue", label="Train loss")
    axes[0].axvline(best_epoch, color="green", linestyle=":", alpha=0.7, label=f"Best epoch {best_epoch}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss"); axes[0].legend()

    axes[1].plot(epochs, val_miou, color="coral", label="Val mIoU")
    axes[1].axvline(best_epoch, color="green", linestyle=":", alpha=0.7)
    axes[1].axhline(max(val_miou), color="coral", linestyle="--", alpha=0.4,
                    label=f"Best: {max(val_miou):.3f}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU"); axes[1].set_ylim(0, 1); axes[1].legend()

    plt.suptitle("Phase 4 — PEFT adapter training curves", fontsize=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_per_class_chart(path: Path, per_class: dict[str, float]) -> None:
    names  = list(per_class.keys())
    values = list(per_class.values())
    colours = ["green" if v >= 0.60 else "orange" if v >= 0.30 else "red" for v in values]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, values, color=colours, edgecolor="white")
    ax.axvline(0.30, color="red",    linestyle="--", linewidth=1, label="Fail < 0.30")
    ax.axvline(0.60, color="orange", linestyle="--", linewidth=1, label="OK ≥ 0.60")
    mean = np.mean(values)
    ax.axvline(mean, color="black", linestyle="-", linewidth=1.5, label=f"Mean {mean:.3f}")
    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("IoU")
    ax.set_title("Phase 4 — per-class IoU on test set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_qualitative_grid(
    path: Path,
    model, transforms, dataset, device: torch.device,
    n_per_class: int = 1,
) -> None:
    by_class: dict[str, list[int]] = {}
    for i in range(len(dataset)):
        cls = dataset[i].class_name
        by_class.setdefault(cls, []).append(i)

    rows = []
    for cls, indices in sorted(by_class.items()):
        for idx in random.sample(indices, min(n_per_class, len(indices))):
            rows.append((cls, idx))

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    model.eval()
    with torch.no_grad():
        for row, (cls, idx) in enumerate(rows):
            sample = dataset[idx]
            pred   = predict_instance_mask(model, transforms, sample, device)
            iou    = binary_iou(pred, sample.mask)

            axes[row][0].imshow(sample.image); axes[row][0].set_title(f"Class: {cls}", fontsize=9)
            axes[row][1].imshow(sample.image)
            axes[row][1].imshow(sample.mask, alpha=0.5, cmap="Greens")
            axes[row][1].set_title("GT mask", fontsize=9)
            axes[row][2].imshow(sample.image)
            axes[row][2].imshow(pred, alpha=0.5, cmap="Reds")
            axes[row][2].set_title(f"PEFT pred (IoU {iou:.2f})", fontsize=9)
            for ax in axes[row]:
                ax.axis("off")

    model.train()
    plt.suptitle("Phase 4 — GT (green) vs PEFT prediction (red)", fontsize=11)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 — full PEFT training run.")
    parser.add_argument("--dataset-root",      default=Path("dataset"),               type=Path)
    parser.add_argument("--model-id",          default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device",            default="auto", choices=("auto","cuda","mps","cpu"))
    parser.add_argument("--adapter-bottleneck",default=64, type=int)
    parser.add_argument("--adapter-lr",        default=1e-4, type=float)
    parser.add_argument("--decoder-lr",        default=5e-5, type=float)
    parser.add_argument("--weight-decay",      default=1e-4, type=float)
    parser.add_argument("--l2-coeff",          default=0.01, type=float)
    parser.add_argument("--max-grad-norm",     default=1.0,  type=float)
    parser.add_argument("--epochs",            default=50,   type=int)
    parser.add_argument("--patience",          default=10,   type=int)
    parser.add_argument("--val-every",         default=1,    type=int)
    parser.add_argument("--seed",              default=42,   type=int)
    parser.add_argument("--output-dir",        default=Path("outputs/phase4"), type=Path)
    parser.add_argument("--checkpoint-path",   default=Path("outputs/phase4/best_model.pt"), type=Path)
    parser.add_argument("--resume",            action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_name = resolve_device(args.device)
    device      = torch.device(device_name)
    print(f"Device: {device_name}", flush=True)

    # Datasets
    train_dataset = CocoPromptInstanceDataset(args.dataset_root, split="train")
    val_dataset   = CocoPromptInstanceDataset(args.dataset_root, split="val")
    test_dataset  = CocoPromptInstanceDataset(args.dataset_root, split="test")
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}  Test samples: {len(test_dataset)}", flush=True)

    # Model + PEFT setup
    model      = load_sam2_model(args.model_id, device_name, mode="train")
    adapters   = insert_hiera_adapters(model, bottleneck=args.adapter_bottleneck)
    adapter_params, decoder_params = configure_peft_trainable(model)
    optimizer  = build_peft_optimizer(adapter_params, decoder_params,
                                      adapter_lr=args.adapter_lr,
                                      decoder_lr=args.decoder_lr,
                                      weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    transforms = make_transforms(model)
    counts     = count_parameters(model)
    print(f"Adapters: {len(adapters)}  Trainable: {counts.trainable/1e6:.2f}M / {counts.total/1e6:.2f}M ({counts.trainable_percent:.2f}%)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch        = 1
    best_val_miou      = 0.0
    epochs_no_improve  = 0
    training_log: list[dict] = []

    if args.resume and args.checkpoint_path.exists():
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch       = ckpt["epoch"] + 1
        best_val_miou     = ckpt["best_val_miou"]
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        training_log      = ckpt.get("training_log", [])
        print(f"Resumed from epoch {ckpt['epoch']}  best_val_miou={best_val_miou:.4f}", flush=True)

    # Training loop — image-grouped for 6× encoder cache speedup
    all_step_losses: list[float] = []
    trainable_params = adapter_params + decoder_params
    image_groups = grouped_by_image(train_dataset)  # list of lists, built once
    n_images = len(image_groups)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        random.shuffle(image_groups)   # shuffle image order each epoch
        t0 = time.time()
        ann_step = 0  # global annotation counter for logging

        for img_idx, ann_indices in enumerate(image_groups, start=1):
            # --- encode once per image ---
            sample0 = train_dataset[ann_indices[0]]
            backbone_out = encode_image(model, transforms, sample0.image, device)

            # --- accumulate gradients across all annotations of this image ---
            optimizer.zero_grad(set_to_none=True)
            image_losses: list[float] = []

            for idx in ann_indices:
                sample = train_dataset[idx]
                logits = decode_from_features(model, transforms, backbone_out, sample, device)
                target = low_res_target(sample, logits.shape[-2:], device)
                loss   = combined_loss(logits, target) / len(ann_indices)  # normalise

                # L2 on adapter weights
                l2 = torch.stack([p.norm(2) for p in adapter_params]).mean()
                loss = loss + (args.l2_coeff * l2) / len(ann_indices)

                if torch.isnan(loss):
                    print(f"NaN loss at epoch {epoch} img {img_idx} — skipping annotation", flush=True)
                    continue

                loss.backward()
                image_losses.append(float(loss.detach().cpu()) * len(ann_indices))

            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()

            ann_step += len(ann_indices)
            if image_losses:
                mean_img_loss = float(np.mean(image_losses))
                epoch_losses.append(mean_img_loss)
                all_step_losses.append(mean_img_loss)

            if img_idx % 50 == 0 or img_idx == n_images:
                last_loss = epoch_losses[-1] if epoch_losses else float("nan")
                print(f"epoch={epoch}/{args.epochs} img={img_idx}/{n_images} ann={ann_step} loss={last_loss:.6f}", flush=True)

        train_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        scheduler.step()
        elapsed = time.time() - t0

        # Validation
        val_miou, val_per_class = 0.0, {}
        if epoch % args.val_every == 0:
            val_miou, val_per_class = evaluate(model, transforms, val_dataset, device)

        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        print(f"Epoch {epoch:3d} | loss={train_loss_mean:.4f} | val_mIoU={val_miou:.4f} | "
              f"gpu={gpu_mem:.1f}GB | t={elapsed:.0f}s", flush=True)

        log_entry = {
            "epoch": epoch, "train_loss": train_loss_mean,
            "val_miou": val_miou, "val_per_class": val_per_class,
            "gpu_mem_gb": gpu_mem, "elapsed_s": elapsed,
        }
        training_log.append(log_entry)

        # Checkpoint + early stopping
        if val_miou > best_val_miou:
            best_val_miou     = val_miou
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_val_miou": best_val_miou, "epochs_no_improve": 0,
                "training_log": training_log,
            }, args.checkpoint_path)
            print(f"  -> New best val mIoU {best_val_miou:.4f} — checkpoint saved", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)", flush=True)
                break

        # Save log + curves after every epoch
        with open(args.output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)
        with open(args.output_dir / "step_losses.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss"])
            writer.writeheader()
            for s, l in enumerate(all_step_losses, start=1):
                writer.writerow({"step": s, "loss": l})
        save_training_curves(Path("viz/phase4_training_curves.png"), training_log)

    # -----------------------------------------------------------------------
    # Post-training: load best checkpoint, run test set
    # -----------------------------------------------------------------------
    print("\n=== Loading best checkpoint for test evaluation ===", flush=True)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_miou, test_per_class = evaluate(model, transforms, test_dataset, device)
    print(f"Test mIoU: {test_miou:.4f}", flush=True)
    for cls, iou in sorted(test_per_class.items()):
        status = "PASS" if iou >= 0.30 else "FAIL"
        print(f"  {status}  {cls}: {iou:.4f}", flush=True)

    # Final figures
    save_per_class_chart(Path("viz/phase4_per_class_iou.png"), test_per_class)
    save_qualitative_grid(Path("viz/phase4_qualitative_grid.png"),
                          model, transforms, test_dataset, device)
    save_training_curves(Path("viz/phase4_training_curves.png"), training_log)

    # Final summary
    best_epoch   = max(training_log, key=lambda r: r["val_miou"])["epoch"]
    best_val     = max(r["val_miou"] for r in training_log)
    summary = {
        "best_epoch": best_epoch, "best_val_miou": best_val,
        "test_miou": test_miou, "test_per_class": test_per_class,
        "total_epochs_run": len(training_log),
        "trainable_parameters": counts.trainable,
        "total_parameters": counts.total,
        "trainable_percent": counts.trainable_percent,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
