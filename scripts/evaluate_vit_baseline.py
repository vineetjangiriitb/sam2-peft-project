"""Evaluate trained ViT baseline on the test split. Outputs match zero-shot SAM2 format."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from sam2_peft.data import CocoSegmentationDataset
from sam2_peft.metrics import compute_iou, compute_miou
from sam2_peft.models.vit_segmenter import ViTSegmenter

NUM_CLASSES = 5
CLASS_NAMES = ["background", "arm", "leg", "torso", "head"]
FOREGROUND_NAMES = CLASS_NAMES[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--checkpoint", default="outputs/phase2_vit_baseline/best.pt")
    parser.add_argument("--output-dir", default="outputs/phase2_vit_baseline")
    parser.add_argument("--viz-dir", default="viz")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    viz_dir = Path(args.viz_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ViTSegmenter(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_mIoU={ckpt['val_miou']:.4f})")

    test_ds = CocoSegmentationDataset(args.dataset_root, split="test", image_size=224)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test images: {len(test_ds)}")

    rows: list[dict] = []
    latencies: list[float] = []
    worst: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []  # (miou, img, pred, gt)

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images_dev = images.to(device)
            t0 = time.perf_counter()
            logits = model(images_dev)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            pred = logits.argmax(dim=1).squeeze(0).cpu()
            gt = masks.squeeze(0)

            miou = compute_miou(pred, gt, num_classes=NUM_CLASSES, include_background=False)
            row: dict = {"image_index": i, "miou": miou, "latency_ms": latency_ms}
            for ci, name in enumerate(FOREGROUND_NAMES, start=1):
                iou = compute_iou(pred, gt, class_index=ci)
                row[f"iou_{name}"] = iou if iou is not None else float("nan")
            rows.append(row)

            # collect worst predictions for the failure grid
            img_np = images.squeeze(0).permute(1, 2, 0).numpy()
            worst.append((miou, img_np, pred.numpy(), gt.numpy()))

    # CSV — same schema as zero-shot results
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results CSV: {csv_path}")

    # Summary JSON
    all_miou = [r["miou"] for r in rows]
    mean_miou = float(np.mean(all_miou))
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    per_class_mean = {
        name: float(np.nanmean([r[f"iou_{name}"] for r in rows]))
        for name in FOREGROUND_NAMES
    }
    summary = {
        "images": len(rows),
        "mean_miou": mean_miou,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "per_class_iou": per_class_mean,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_val_miou": float(ckpt["val_miou"]),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")
    print(f"\nmean mIoU:     {mean_miou:.4f}")
    print(f"mean latency:  {mean_latency:.1f} ms/image")
    print(f"p95  latency:  {p95_latency:.1f} ms/image")
    for name, iou in per_class_mean.items():
        print(f"  {name}: {iou:.4f}")

    # mIoU distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_miou, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    ax.axvline(mean_miou, color="red", linestyle="--", label=f"mean={mean_miou:.3f}")
    ax.set_xlabel("mIoU")
    ax.set_ylabel("Images")
    ax.set_title("ViT Baseline — Test mIoU Distribution")
    ax.legend()
    fig.tight_layout()
    dist_path = viz_dir / "phase2_vit_iou_distribution.png"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"Distribution plot: {dist_path}")

    # Failure mode grid — 8 worst predictions
    worst.sort(key=lambda t: t[0])
    n_show = min(8, len(worst))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap("tab10", NUM_CLASSES)
    for row_i, (miou, img, pred, gt) in enumerate(worst[:n_show]):
        axes[row_i][0].imshow(img)
        axes[row_i][0].set_title(f"Image (mIoU={miou:.3f})", fontsize=8)
        axes[row_i][1].imshow(gt, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
        axes[row_i][1].set_title("Ground Truth", fontsize=8)
        axes[row_i][2].imshow(pred, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
        axes[row_i][2].set_title("Prediction", fontsize=8)
        for ax in axes[row_i]:
            ax.axis("off")
    fig.suptitle("ViT Baseline — Worst Failure Cases", fontsize=12)
    fig.tight_layout()
    fail_path = viz_dir / "phase2_vit_failure_modes.png"
    fig.savefig(fail_path, dpi=150)
    plt.close(fig)
    print(f"Failure grid:  {fail_path}")


if __name__ == "__main__":
    main()
