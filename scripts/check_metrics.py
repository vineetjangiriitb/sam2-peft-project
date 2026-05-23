from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam2_peft.metrics import compute_iou, compute_miou, per_class_iou


def main() -> int:
    gt = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [3, 3, 2, 2],
            [4, 4, 0, 0],
        ]
    )
    pred = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 1, 2, 0],
            [3, 0, 2, 2],
            [4, 0, 0, 0],
        ]
    )

    arm_iou = compute_iou(pred, gt, class_index=1)
    leg_iou = compute_iou(pred, gt, class_index=2)
    torso_iou = compute_iou(pred, gt, class_index=3)
    head_iou = compute_iou(pred, gt, class_index=4)
    miou = compute_miou(pred, gt, num_classes=5)
    by_class = per_class_iou(pred, gt, ["background", "arm", "leg", "torso", "head"])

    print(f"arm IoU:     {arm_iou:.4f}")
    print(f"leg IoU:     {leg_iou:.4f}")
    print(f"torso IoU:   {torso_iou:.4f}")
    print(f"head IoU:    {head_iou:.4f}")
    print(f"mIoU:        {miou:.4f}")
    print(f"per class:   {by_class}")

    assert abs(arm_iou - (2 / 3)) < 1e-6
    assert abs(leg_iou - (3 / 4)) < 1e-6
    assert abs(torso_iou - (1 / 2)) < 1e-6
    assert abs(head_iou - (1 / 2)) < 1e-6
    assert abs(miou - ((2 / 3 + 3 / 4 + 1 / 2 + 1 / 2) / 4)) < 1e-6
    print("PASS metric checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
