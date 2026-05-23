"""Quick 2-image sanity check for ViTSegmenter. Runs on CPU or MPS — no GPU required."""
from __future__ import annotations

import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from sam2_peft.data import CocoSegmentationDataset
from sam2_peft.models.vit_segmenter import ViTSegmenter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    args = parser.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    ds = CocoSegmentationDataset(args.dataset_root, split="val", image_size=224)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    images, masks = next(iter(loader))
    print(f"batch images: {images.shape}")
    print(f"batch masks:  {masks.shape}")
    assert images.shape == (2, 3, 224, 224), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (2, 224, 224), f"Unexpected mask shape: {masks.shape}"

    model = ViTSegmenter(num_classes=ds.num_classes, pretrained=False).to(device)
    model.eval()
    images = images.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(images)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"logits:       {logits.shape}")
    assert logits.shape == (2, ds.num_classes, 224, 224), f"Unexpected logit shape: {logits.shape}"
    assert logits.dtype == torch.float32
    print(f"Forward pass: {elapsed:.1f} ms for 2 images")
    print(f"num_classes:  {ds.num_classes} (background + {ds.num_foreground_classes} foreground)")
    print("Smoke test PASSED")


if __name__ == "__main__":
    main()
