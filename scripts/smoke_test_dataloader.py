from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam2_peft.data import CocoSegmentationDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 dataloader smoke test.")
    parser.add_argument("--dataset-root", default="dataset", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--image-size", default=1024, type=int)
    args = parser.parse_args()

    try:
        dataset = CocoSegmentationDataset(
            dataset_root=args.dataset_root,
            split=args.split,
            image_size=args.image_size,
        )
    except FileNotFoundError as exc:
        print(f"FAIL dataloader smoke test: {exc}")
        print("Export the COCO dataset first, then rerun this script.")
        return 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, masks = next(iter(loader))

    print(f"Split: {args.split}")
    print(f"Images: {len(dataset)}")
    print(f"Foreground classes: {dataset.foreground_class_names}")
    print("Background label: 0")
    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Mask batch shape:  {tuple(masks.shape)}")
    print(f"Image dtype/range: {images.dtype}, {images.min().item():.3f}-{images.max().item():.3f}")
    print(f"Mask dtype/label ids in batch: {masks.dtype}, {torch.unique(masks).tolist()}")

    expected_image_shape = (args.batch_size, 3, args.image_size, args.image_size)
    expected_mask_shape = (args.batch_size, args.image_size, args.image_size)
    if tuple(images.shape) != expected_image_shape:
        raise AssertionError(f"Expected image shape {expected_image_shape}, got {tuple(images.shape)}")
    if tuple(masks.shape) != expected_mask_shape:
        raise AssertionError(f"Expected mask shape {expected_mask_shape}, got {tuple(masks.shape)}")

    print("PASS dataloader smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
