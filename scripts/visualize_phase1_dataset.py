from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


CLASS_COLORS = {
    "arm": [255, 80, 80],
    "leg": [80, 160, 255],
    "torso": [80, 220, 120],
    "head": [255, 200, 50],
}


def overlay_masks(coco: COCO, image: np.ndarray, anns: list[dict]) -> np.ndarray:
    overlay = image.copy()
    for ann in anns:
        mask = coco.annToMask(ann)
        cat_name = coco.loadCats(ann["category_id"])[0]["name"]
        color = np.array(CLASS_COLORS.get(cat_name, [128, 128, 128]))
        overlay[mask == 1] = (overlay[mask == 1] * 0.4 + color * 0.6).astype(np.uint8)
    return overlay


def save_alignment_grid(dataset_root: Path, output_dir: Path, seed: int) -> None:
    random.seed(seed)
    coco_val = COCO(str(dataset_root / "annotations" / "val.json"))
    img_ids = coco_val.getImgIds()
    if len(img_ids) < 5:
        raise ValueError(f"Need at least 5 validation images for mask spot check, found {len(img_ids)}")

    sample_ids = random.sample(img_ids, 5)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for col, img_id in enumerate(sample_ids):
        img_info = coco_val.loadImgs(img_id)[0]
        image_path = dataset_root / "images" / "val" / img_info["file_name"]
        image = np.array(Image.open(image_path).convert("RGB"))
        ann_ids = coco_val.getAnnIds(imgIds=img_id)
        anns = coco_val.loadAnns(ann_ids)
        overlay = overlay_masks(coco_val, image, anns)

        axes[0, col].imshow(image)
        axes[0, col].set_title(f"Image {col + 1}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"{len(anns)} masks", fontsize=9)
        axes[1, col].axis("off")

    legend_patches = [
        mpatches.Patch(color=[c / 255 for c in color], label=name)
        for name, color in CLASS_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9)
    plt.suptitle("Phase 1 - GT mask alignment check (top: raw, bottom: masked)", fontsize=11)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "phase1_mask_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_class_balance(dataset_root: Path, output_dir: Path) -> None:
    coco_train = COCO(str(dataset_root / "annotations" / "train.json"))
    categories = coco_train.loadCats(coco_train.getCatIds())
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    counts = Counter(ann["category_id"] for ann in coco_train.dataset["annotations"])

    names = [cat_id_to_name[cat_id] for cat_id in sorted(counts)]
    values = [counts[cat_id] for cat_id in sorted(counts)]
    total = sum(values)
    if total == 0:
        raise ValueError("No train annotations found.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar(names, values, color="steelblue", edgecolor="white")
    axes[0].set_title("Annotation count per class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", labelrotation=30)
    for bar, count in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].pie(values, labels=names, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Class distribution")

    dominant = max(values) / total * 100
    print(f"Most dominant class: {dominant:.1f}% of all train annotations")
    if dominant > 60:
        print("WARNING: class imbalance detected; collect more minority-class examples.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "phase1_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Phase 1 dataset visualizations.")
    parser.add_argument("--dataset-root", default="dataset", type=Path)
    parser.add_argument("--output-dir", default="viz", type=Path)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args()

    save_alignment_grid(args.dataset_root, args.output_dir, args.seed)
    save_class_balance(args.dataset_root, args.output_dir)
    print(f"Saved {args.output_dir / 'phase1_mask_alignment.png'}")
    print(f"Saved {args.output_dir / 'phase1_class_balance.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
