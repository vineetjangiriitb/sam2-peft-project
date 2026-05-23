from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_SPLITS = ("train", "val", "test")
REQUIRED_KEYS = ("images", "annotations", "categories")
TARGET_CLASSES = ("arm", "leg", "torso", "head")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_split(dataset_root: Path, split: str) -> tuple[bool, Counter]:
    ann_path = dataset_root / "annotations" / f"{split}.json"
    image_dir = dataset_root / "images" / split
    ok = True

    if not ann_path.exists():
        print(f"FAIL {split}: missing {ann_path}")
        return False, Counter()
    if not image_dir.exists():
        print(f"FAIL {split}: missing {image_dir}")
        return False, Counter()

    coco = load_json(ann_path)
    for key in REQUIRED_KEYS:
        if key not in coco:
            print(f"FAIL {split}: missing COCO key '{key}'")
            ok = False

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    category_names = [cat.get("name") for cat in categories]
    category_ids = {cat["id"] for cat in categories if "id" in cat}
    image_ids = {img["id"] for img in images if "id" in img}
    image_names = [img.get("file_name") for img in images]

    if sorted(category_names) != sorted(TARGET_CLASSES):
        print(f"FAIL {split}: expected exactly foreground classes {list(TARGET_CLASSES)}, found {category_names}")
        ok = False

    missing_files = [
        name for name in image_names
        if not name or not (image_dir / name).exists()
    ]
    if missing_files:
        print(f"FAIL {split}: {len(missing_files)} referenced image files are missing")
        for name in missing_files[:10]:
            print(f"  missing: {name}")
        ok = False

    bad_annotations = []
    empty_segmentations = []
    category_counts: Counter = Counter()
    for ann in annotations:
        ann_id = ann.get("id", "<missing id>")
        if ann.get("image_id") not in image_ids:
            bad_annotations.append((ann_id, "unknown image_id"))
        if ann.get("category_id") not in category_ids:
            bad_annotations.append((ann_id, "unknown category_id"))
        if "segmentation" not in ann:
            bad_annotations.append((ann_id, "missing segmentation"))
        elif not ann["segmentation"]:
            empty_segmentations.append(ann_id)
        if ann.get("category_id") in category_ids:
            category_counts[ann["category_id"]] += 1

    if bad_annotations:
        print(f"FAIL {split}: {len(bad_annotations)} invalid annotations")
        for ann_id, reason in bad_annotations[:10]:
            print(f"  annotation {ann_id}: {reason}")
        ok = False

    if empty_segmentations:
        print(f"FAIL {split}: {len(empty_segmentations)} annotations have empty segmentation")
        for ann_id in empty_segmentations[:10]:
            print(f"  annotation {ann_id}: empty segmentation")
        ok = False

    print(f"{split}: images={len(images)} annotations={len(annotations)} categories={len(categories)}")
    print(f"{split}: foreground categories={category_names}")
    return ok, category_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1 COCO segmentation dataset.")
    parser.add_argument("--dataset-root", default="dataset", type=Path)
    args = parser.parse_args()

    all_ok = True
    total_counts: Counter = Counter()

    for split in REQUIRED_SPLITS:
        split_ok, split_counts = validate_split(args.dataset_root, split)
        all_ok = all_ok and split_ok
        total_counts.update(split_counts)

    train_path = args.dataset_root / "annotations" / "train.json"
    if train_path.exists():
        train = load_json(train_path)
        train_count = len(train.get("images", []))
        if train_count < 280:
            print(f"FAIL train: expected at least 280 train images, found {train_count}")
            all_ok = False

    if total_counts:
        total = sum(total_counts.values())
        dominant = max(total_counts.values()) / total * 100
        print(f"Most dominant class share: {dominant:.1f}%")
        if dominant > 60:
            print("FAIL class balance: one class exceeds 60% of all annotations")
            all_ok = False
    else:
        print("FAIL class balance: no annotations found")
        all_ok = False

    if all_ok:
        print("PASS Phase 1 COCO structure checks")
        return 0

    print("FAIL Phase 1 COCO structure checks")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
