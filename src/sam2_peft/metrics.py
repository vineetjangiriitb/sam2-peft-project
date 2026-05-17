from __future__ import annotations

import numpy as np
import torch


def _as_numpy(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy()
    return mask


def compute_iou(
    pred_mask: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    class_index: int,
) -> float | None:
    pred = _as_numpy(pred_mask) == class_index
    gt = _as_numpy(gt_mask) == class_index
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return None
    intersection = np.logical_and(pred, gt).sum()
    return float(intersection / union)


def compute_miou(
    pred_mask: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    num_classes: int,
    include_background: bool = False,
) -> float:
    start_class = 0 if include_background else 1
    ious = [
        iou
        for class_index in range(start_class, num_classes)
        if (iou := compute_iou(pred_mask, gt_mask, class_index)) is not None
    ]
    if not ious:
        raise ValueError("Cannot compute mIoU because no classes have non-empty union.")
    return float(sum(ious) / len(ious))


def per_class_iou(
    pred_mask: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    class_names: list[str],
    include_background: bool = False,
) -> dict[str, float]:
    start_class = 0 if include_background else 1
    scores: dict[str, float] = {}
    for class_index in range(start_class, len(class_names)):
        iou = compute_iou(pred_mask, gt_mask, class_index)
        if iou is not None:
            scores[class_names[class_index]] = iou
    return scores

