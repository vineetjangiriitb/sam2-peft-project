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
            [0, 0, 2, 2],
        ]
    )
    pred = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 1, 2, 0],
            [0, 0, 2, 2],
        ]
    )

    arm_iou = compute_iou(pred, gt, class_index=1)
    leg_iou = compute_iou(pred, gt, class_index=2)
    miou = compute_miou(pred, gt, num_classes=3)
    by_class = per_class_iou(pred, gt, ["background", "arm", "leg"])

    print(f"class 1 IoU: {arm_iou:.4f}")
    print(f"class 2 IoU: {leg_iou:.4f}")
    print(f"mIoU:        {miou:.4f}")
    print(f"per class:   {by_class}")

    assert abs(arm_iou - (2 / 3)) < 1e-6
    assert abs(leg_iou - (3 / 4)) < 1e-6
    assert abs(miou - ((2 / 3 + 3 / 4) / 2)) < 1e-6
    print("PASS metric checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

