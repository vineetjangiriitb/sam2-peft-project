from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam2_peft.models.sam2_peft import (
    assert_frozen_image_encoder,
    configure_peft_trainable,
    count_parameters,
    insert_hiera_adapters,
)
from sam2_peft.phase3 import (
    CocoPromptInstanceDataset,
    combined_loss,
    forward_prompt_sample,
    load_sam2_model,
    low_res_target,
    make_transforms,
    resolve_device,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 SAM2 PEFT setup checks.")
    parser.add_argument("--dataset-root", default=Path("dataset"), type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--adapter-bottleneck", default=3, type=int)
    parser.add_argument("--max-trainable-params", default=6_000_000, type=int)
    parser.add_argument("--max-trainable-percent", default=2.0, type=float)
    parser.add_argument("--skip-gradient-check", action="store_true")
    args = parser.parse_args()

    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    print(f"Using device: {device_name}", flush=True)
    model = load_sam2_model(args.model_id, device_name, mode="train")
    adapters = insert_hiera_adapters(model, bottleneck=args.adapter_bottleneck)
    configure_peft_trainable(model)
    counts = count_parameters(model)

    print(f"Inserted adapters: {len(adapters)}")
    print(
        f"Trainable: {counts.trainable / 1e6:.2f}M / "
        f"{counts.total / 1e6:.2f}M ({counts.trainable_percent:.2f}%)"
    )
    if counts.trainable > args.max_trainable_params:
        raise AssertionError(f"Trainable parameter count exceeds {args.max_trainable_params}.")
    if counts.trainable_percent > args.max_trainable_percent:
        raise AssertionError(f"Trainable parameter percent exceeds {args.max_trainable_percent}%.")
    print("PASS Problem 3.1 parameter-count check")

    if not args.skip_gradient_check:
        dataset = CocoPromptInstanceDataset(args.dataset_root, split=args.split, sample_limit=1)
        transforms = make_transforms(model)
        sample = dataset[0]
        logits = forward_prompt_sample(model, transforms, sample, device)
        target = low_res_target(sample, logits.shape[-2:], device)
        loss = combined_loss(logits, target)
        loss.backward()
        assert_frozen_image_encoder(model)
        print("PASS Problem 3.2 frozen image-encoder gradient check")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
