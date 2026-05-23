"""Train ViT-B/16 segmentation baseline on the robot-parts dataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from sam2_peft.data import CocoSegmentationDataset
from sam2_peft.metrics import compute_miou, per_class_iou
from sam2_peft.models.vit_segmenter import ViTSegmenter

NUM_CLASSES = 5  # background + arm, leg, torso, head


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    all_miou: list[float] = []
    class_iou_accum: dict[str, list[float]] = {}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()

            for pred, gt in zip(preds, masks):
                miou = compute_miou(pred, gt, num_classes=NUM_CLASSES, include_background=False)
                all_miou.append(miou)
                class_names = loader.dataset.class_names
                per_class = per_class_iou(pred, gt, class_names=class_names, include_background=False)
                for name, iou in per_class.items():
                    class_iou_accum.setdefault(name, []).append(iou)

    mean_miou = sum(all_miou) / len(all_miou)
    mean_per_class = {name: sum(v) / len(v) for name, v in class_iou_accum.items()}
    return mean_miou, mean_per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default="outputs/phase2_vit_baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    train_ds = CocoSegmentationDataset(args.dataset_root, split="train", image_size=224)
    val_ds = CocoSegmentationDataset(args.dataset_root, split="val", image_size=224)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = ViTSegmenter(num_classes=NUM_CLASSES, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_miou = 0.0
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = epoch_loss / len(train_ds)
        val_miou, val_per_class = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - t0

        per_class_str = "  ".join(f"{k}: {v:.4f}" for k, v in val_per_class.items())
        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  loss {train_loss:.4f}"
            f"  val_mIoU {val_miou:.4f}"
            f"  [{per_class_str}]"
            f"  {elapsed:.0f}s"
        )

        row = {"epoch": epoch, "train_loss": train_loss, "val_miou": val_miou, **val_per_class}
        history.append(row)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_miou": val_miou}, output_dir / "best.pt")
            print(f"  -> new best checkpoint saved (val_mIoU={val_miou:.4f})")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to: {output_dir}/best.pt")


if __name__ == "__main__":
    main()
