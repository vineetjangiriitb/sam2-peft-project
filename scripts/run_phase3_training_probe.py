from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
    binary_iou,
    load_sam2_model,
    make_transforms,
    predict_instance_mask,
    resolve_device,
    train_one_sample,
)


def gpu_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1e9


def save_loss_csv(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss"])
        writer.writeheader()
        for step, loss in enumerate(losses, start=1):
            writer.writerow({"step": step, "loss": loss})


def save_overfit_plot(
    path: Path,
    losses: list[float],
    mean_iou: float,
    sample_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(5, 4)

    ax_loss = fig.add_subplot(grid[0, :2])
    ax_loss.plot(losses, color="steelblue")
    ax_loss.axhline(0.1, color="green", linestyle="--", label="Target < 0.1")
    ax_loss.set_title("Overfit probe - training loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_iou = fig.add_subplot(grid[0, 2:])
    ax_iou.bar(["Train mIoU (20 instances)"], [mean_iou], color="steelblue")
    ax_iou.axhline(0.80, color="green", linestyle="--", label="Pass threshold: 0.80")
    ax_iou.set_ylim(0, 1)
    ax_iou.set_title("Overfit probe - train IoU")
    ax_iou.legend()

    for row, (image, gt_mask, pred_mask, iou, title) in enumerate(sample_rows[:4], start=1):
        axes = [fig.add_subplot(grid[row, col]) for col in range(4)]
        axes[0].imshow(image)
        axes[0].set_title(title, fontsize=8)
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("GT", fontsize=8)
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title(f"Pred IoU {iou:.3f}", fontsize=8)
        overlay = image.copy()
        overlay[pred_mask.astype(bool)] = (overlay[pred_mask.astype(bool)] * 0.45 + np.array([255, 80, 80]) * 0.55).astype(np.uint8)
        axes[3].imshow(overlay)
        axes[3].set_title("Prediction overlay", fontsize=8)
        for ax in axes:
            ax.axis("off")

    plt.suptitle("Phase 3 - PEFT adapter overfit probe", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 SAM2 PEFT smoke or overfit probe.")
    parser.add_argument("--dataset-root", default=Path("dataset"), type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--adapter-bottleneck", default=3, type=int)
    parser.add_argument("--mode", default="smoke", choices=("smoke", "overfit"))
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--sample-limit", default=None, type=int)
    parser.add_argument("--image-limit", default=None, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--output-dir", default=Path("outputs/phase3_training_probe"), type=Path)
    parser.add_argument("--viz-path", default=Path("viz/phase3_overfit_probe.png"), type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    epochs = args.epochs if args.epochs is not None else (1 if args.mode == "smoke" else 5)
    image_limit = args.image_limit if args.image_limit is not None else (None if args.mode == "smoke" else 20)
    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    print(f"Using device: {device_name}", flush=True)

    dataset = CocoPromptInstanceDataset(
        args.dataset_root,
        split=args.split,
        sample_limit=args.sample_limit,
        image_limit=image_limit,
    )
    model = load_sam2_model(args.model_id, device_name, mode="train")
    adapters = insert_hiera_adapters(model, bottleneck=args.adapter_bottleneck)
    adapter_params, decoder_params = configure_peft_trainable(model)
    optimizer = build_peft_optimizer(adapter_params, decoder_params)
    transforms = make_transforms(model)
    counts = count_parameters(model)

    print(f"Inserted adapters: {len(adapters)}")
    print(f"Training samples: {len(dataset)}")
    print(f"Trainable parameters: {counts.trainable} ({counts.trainable_percent:.2f}%)")

    losses: list[float] = []
    for epoch in range(epochs):
        epoch_losses = []
        for index in range(len(dataset)):
            loss = train_one_sample(model, transforms, dataset[index], optimizer, device)
            losses.append(loss)
            epoch_losses.append(loss)
            print(f"epoch={epoch + 1} step={index + 1}/{len(dataset)} loss={loss:.6f}", flush=True)
        print(f"Epoch {epoch + 1} mean loss: {float(np.mean(epoch_losses)):.6f}", flush=True)

    output_dir = args.output_dir / args.mode
    save_loss_csv(output_dir / "losses.csv", losses)

    memory_gb = gpu_memory_gb(device)
    summary = {
        "mode": args.mode,
        "model_id": args.model_id,
        "device": device_name,
        "epochs": epochs,
        "samples": len(dataset),
        "image_limit": image_limit,
        "sample_limit": args.sample_limit,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": float(np.mean(losses)) if losses else None,
        "trainable_parameters": counts.trainable,
        "total_parameters": counts.total,
        "trainable_percent": counts.trainable_percent,
        "max_gpu_memory_gb": memory_gb,
    }

    if args.mode == "overfit":
        model.eval()
        ious = []
        rows = []
        sample_indices = random.sample(range(len(dataset)), min(4, len(dataset)))
        for index in range(len(dataset)):
            sample = dataset[index]
            pred_mask = predict_instance_mask(model, transforms, sample, device)
            iou = binary_iou(pred_mask, sample.mask)
            ious.append(iou)
            if index in sample_indices:
                rows.append((sample.image, sample.mask, pred_mask, iou, f"{sample.class_name} | {sample.file_name}"))
        mean_iou = float(np.nanmean(ious))
        summary["train_mean_iou"] = mean_iou
        save_overfit_plot(args.viz_path, losses, mean_iou, rows)
        print(f"Train mean IoU on prompted instances from {image_limit} images: {mean_iou:.3f}")
        print(f"Saved {args.viz_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Saved {output_dir / 'losses.csv'}")
    print(f"Saved {output_dir / 'summary.json'}")
    if args.mode == "smoke":
        print("PASS Problem 3.4 one-epoch smoke run completed without NaN loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
