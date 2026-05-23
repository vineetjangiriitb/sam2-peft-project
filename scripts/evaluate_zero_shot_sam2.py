from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam2_peft.metrics import compute_iou, compute_miou


CLASS_COLORS = {
    1: np.array([255, 80, 80], dtype=np.uint8),
    2: np.array([80, 160, 255], dtype=np.uint8),
    3: np.array([80, 220, 120], dtype=np.uint8),
    4: np.array([255, 200, 50], dtype=np.uint8),
}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_predictor(model_id: str, device: str):
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise SystemExit(
            "SAM2 is not installed. Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "or run this script in the Phase 0 RunPod environment."
        ) from exc
    return SAM2ImagePredictor.from_pretrained(model_id, device=device)


def class_maps(coco: COCO) -> tuple[dict[int, int], dict[int, str], list[str]]:
    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda cat: cat["id"])
    category_id_to_index = {category["id"]: index + 1 for index, category in enumerate(categories)}
    category_id_to_name = {category["id"]: category["name"] for category in categories}
    class_names = ["background"] + [category["name"] for category in categories]
    return category_id_to_index, category_id_to_name, class_names


def semantic_mask(coco: COCO, image_id: int, height: int, width: int, category_id_to_index: dict[int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    for ann in anns:
        class_index = category_id_to_index[ann["category_id"]]
        mask[coco.annToMask(ann).astype(bool)] = class_index
    return mask


def bbox_center(bbox: list[float]) -> np.ndarray:
    x, y, width, height = [float(value) for value in bbox]
    return np.array([[x + width / 2.0, y + height / 2.0]], dtype=np.float32)


def predict_semantic_mask(
    predictor,
    image: np.ndarray,
    anns: list[dict],
    category_id_to_index: dict[int, int],
) -> tuple[np.ndarray, list[dict]]:
    predictor.set_image(image)
    pred = np.zeros(image.shape[:2], dtype=np.uint8)
    rows = []

    for ann in anns:
        class_index = category_id_to_index[ann["category_id"]]
        start = time.perf_counter()
        masks, scores, _ = predictor.predict(
            point_coords=bbox_center(ann["bbox"]),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        mask = masks[0].astype(bool)
        pred[mask] = class_index
        rows.append(
            {
                "annotation_id": ann["id"],
                "class_index": class_index,
                "score": float(scores[0]),
                "inference_time_ms": elapsed_ms,
                "pred_mask": mask,
            }
        )
    return pred, rows


def overlay_semantic(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = image.copy()
    for class_index, color in CLASS_COLORS.items():
        pixels = mask == class_index
        overlay[pixels] = (overlay[pixels] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def save_failure_grid(records: list[dict], output_path: Path) -> None:
    worst = sorted(records, key=lambda record: record["miou"])[:10]
    if not worst:
        return
    fig, axes = plt.subplots(len(worst), 3, figsize=(12, 4 * len(worst)))
    if len(worst) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, record in enumerate(worst):
        image = np.array(Image.open(record["image_path"]).convert("RGB"))
        gt_overlay = overlay_semantic(image, record["gt_mask"])
        pred_overlay = overlay_semantic(image, record["pred_mask"])

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{record['file_name']} | mIoU {record['miou']:.3f}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title("GT mask", fontsize=8)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title("SAM2 prediction", fontsize=8)
        axes[row, 2].axis("off")

    plt.suptitle("Phase 2 - 10 worst zero-shot SAM2 predictions", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_iou_distribution(image_records: list[dict], output_path: Path) -> None:
    values = [record["miou"] for record in image_records]
    if not values:
        return
    mean_iou = statistics.mean(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(mean_iou, color="red", linestyle="--", label=f"Mean mIoU: {mean_iou:.3f}")
    ax.set_xlabel("mIoU per image")
    ax.set_ylabel("Count")
    ax.set_title("Zero-shot SAM2 - mIoU distribution on test set")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "file_name",
        "annotation_id",
        "class_name",
        "iou",
        "sam2_score",
        "inference_time_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot SAM2 on the Phase 2 test split.")
    parser.add_argument("--dataset-root", default=Path("dataset"), type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--output-dir", default=Path("outputs/phase2_zero_shot_sam2"), type=Path)
    parser.add_argument("--viz-dir", default=Path("viz"), type=Path)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    annotation_path = args.dataset_root / "annotations" / f"{args.split}.json"
    image_dir = args.dataset_root / "images" / args.split
    coco = COCO(str(annotation_path))
    category_id_to_index, category_id_to_name, class_names = class_maps(coco)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    predictor = load_predictor(args.model_id, device)

    image_ids = sorted(coco.getImgIds())
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    csv_rows = []
    image_records = []
    per_class_ious: defaultdict[str, list[float]] = defaultdict(list)
    image_latencies = []
    annotation_latencies = []

    with torch.inference_mode():
        for position, image_id in enumerate(image_ids, start=1):
            image_info = coco.loadImgs(image_id)[0]
            image_path = image_dir / image_info["file_name"]
            image = np.array(Image.open(image_path).convert("RGB"))
            anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
            gt_mask = semantic_mask(coco, image_id, image_info["height"], image_info["width"], category_id_to_index)
            image_start = time.perf_counter()
            pred_mask, prediction_rows = predict_semantic_mask(predictor, image, anns, category_id_to_index)
            image_latency_ms = (time.perf_counter() - image_start) * 1000.0
            image_latencies.append(image_latency_ms)
            image_miou = compute_miou(pred_mask, gt_mask, num_classes=len(class_names))

            present_class_indices = sorted(set(np.unique(gt_mask).tolist()) - {0})
            for class_index in present_class_indices:
                class_name = class_names[class_index]
                per_class_ious[class_name].append(compute_iou(pred_mask, gt_mask, class_index) or 0.0)

            for prediction in prediction_rows:
                ann = next(ann for ann in anns if ann["id"] == prediction["annotation_id"])
                class_name = category_id_to_name[ann["category_id"]]
                gt_instance_mask = coco.annToMask(ann).astype(bool)
                pred_instance_mask = prediction["pred_mask"]
                union = np.logical_or(pred_instance_mask, gt_instance_mask).sum()
                instance_iou = 0.0 if union == 0 else float(np.logical_and(pred_instance_mask, gt_instance_mask).sum() / union)
                annotation_latencies.append(prediction["inference_time_ms"])
                csv_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": image_info["file_name"],
                        "annotation_id": prediction["annotation_id"],
                        "class_name": class_name,
                        "iou": instance_iou,
                        "sam2_score": prediction["score"],
                        "inference_time_ms": prediction["inference_time_ms"],
                    }
                )

            image_records.append(
                {
                    "image_id": image_id,
                    "file_name": image_info["file_name"],
                    "image_path": image_path,
                    "miou": image_miou,
                    "inference_time_ms": image_latency_ms,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                }
            )
            print(f"[{position}/{len(image_ids)}] {image_info['file_name']} mIoU={image_miou:.3f}", flush=True)

    image_mious = [record["miou"] for record in image_records]
    summary = {
        "model_id": args.model_id,
        "device": device,
        "split": args.split,
        "images": len(image_records),
        "annotations": len(csv_rows),
        "foreground_class_names": class_names[1:],
        "semantic_labels": {"0": "background", **{str(index): name for index, name in enumerate(class_names[1:], start=1)}},
        "mean_miou": statistics.mean(image_mious) if image_mious else None,
        "mean_latency_ms": statistics.mean(image_latencies) if image_latencies else None,
        "p95_latency_ms": p95(image_latencies),
        "mean_annotation_prompt_latency_ms": statistics.mean(annotation_latencies) if annotation_latencies else None,
        "p95_annotation_prompt_latency_ms": p95(annotation_latencies),
        "per_class_iou": {
            class_name: statistics.mean(values)
            for class_name, values in sorted(per_class_ious.items())
            if values
        },
        "annotation_counts": dict(Counter(row["class_name"] for row in csv_rows)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "results.csv", csv_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    save_failure_grid(image_records, args.viz_dir / "phase2_failure_modes.png")
    save_iou_distribution(image_records, args.viz_dir / "phase2_iou_distribution.png")

    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output_dir / 'results.csv'}")
    print(f"Saved {args.output_dir / 'summary.json'}")
    print(f"Saved {args.viz_dir / 'phase2_failure_modes.png'}")
    print(f"Saved {args.viz_dir / 'phase2_iou_distribution.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
