from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path


TARGET_CLASSES = ("arm", "leg", "torso", "head")
SPLITS = ("train", "val", "test")
ALIASES = {
    "arm": "arm",
    "arms": "arm",
    "robot arm": "arm",
    "robotic arm": "arm",
    "upperarm": "arm",
    "lowerarm": "arm",
    "so101 upperarm": "arm",
    "so101 lowerarm": "arm",
    "leg": "leg",
    "legs": "leg",
    "robot leg": "leg",
    "body": "torso",
    "base": "torso",
    "torso": "torso",
    "chest": "torso",
    "robot torso": "torso",
    "so101 base": "torso",
    "head": "head",
    "face": "head",
    "neck": "head",
    "robot head": "head",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def parse_class_map(values: list[str]) -> dict[str, str]:
    aliases = dict(ALIASES)
    for value in values:
        source, sep, target = value.partition("=")
        if not sep or target not in TARGET_CLASSES:
            raise ValueError(f"Expected SOURCE=TARGET where TARGET is one of {TARGET_CLASSES}: {value}")
        aliases[normalize(source)] = target
    return aliases


def source_root(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Expected a Roboflow export directory or zip: {path}")
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tmp.name)
    return Path(tmp.name), tmp


def split_name(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "train"
    if {"valid", "validation", "val"} & parts:
        return "val"
    if "test" in parts:
        return "test"
    return None


def find_coco_jsons(root: Path) -> list[tuple[str | None, Path]]:
    found = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            continue
        if {"images", "annotations", "categories"}.issubset(data):
            found.append((split_name(path), path))
    if not found:
        raise FileNotFoundError(f"No COCO JSON found under {root}")
    return found


def resplit(image_ids: list[int], seed: int) -> dict[int, str]:
    ids = list(image_ids)
    random.Random(seed).shuffle(ids)
    train_end = math.ceil(len(ids) * 0.70)
    if len(ids) >= 280:
        train_end = max(280, train_end)
    val_end = train_end + round(len(ids) * 0.15)
    return {
        image_id: "train" if index < train_end else "val" if index < val_end else "test"
        for index, image_id in enumerate(ids)
    }


def image_path(annotation_path: Path, file_name: str) -> Path | None:
    basename = Path(file_name).name
    for root in (annotation_path.parent, annotation_path.parent.parent):
        direct = root / file_name
        if direct.exists():
            return direct
        matches = list(root.rglob(basename))
        if matches:
            return matches[0]
    return None


def unique_name(file_name: str, used_names: set[str]) -> str:
    path = Path(file_name)
    candidate = path.name
    counter = 1
    while candidate in used_names:
        candidate = f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def prepare(output_root: Path) -> None:
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "annotations").mkdir(parents=True, exist_ok=True)


def empty_outputs() -> dict[str, dict]:
    categories = [
        {"id": index + 1, "name": name, "supercategory": "robot"}
        for index, name in enumerate(TARGET_CLASSES)
    ]
    return {
        split: {"images": [], "annotations": [], "categories": categories}
        for split in SPLITS
    }


def import_coco(root: Path, output_root: Path, aliases: dict[str, str], seed: int) -> dict[str, int]:
    coco_jsons = find_coco_jsons(root)
    declared_splits = {split for split, _ in coco_jsons if split in SPLITS}
    keep_declared_splits = len(declared_splits) > 1
    outputs = empty_outputs()
    next_image_id = {split: 1 for split in SPLITS}
    next_annotation_id = {split: 1 for split in SPLITS}
    used_image_names = {split: set() for split in SPLITS}
    target_id = {name: index + 1 for index, name in enumerate(TARGET_CLASSES)}
    stats: defaultdict[str, int] = defaultdict(int)

    for index, (known_split, annotation_path) in enumerate(coco_jsons):
        coco = load_json(annotation_path)
        categories = {category["id"]: category["name"] for category in coco["categories"]}
        images = {image["id"]: image for image in coco["images"]}
        annotations_by_image: defaultdict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            annotations_by_image[ann.get("image_id")].append(ann)

        split_by_image = (
            {image_id: known_split for image_id in images}
            if keep_declared_splits and known_split in SPLITS
            else resplit(list(images), seed + index)
        )

        for old_image_id, old_image in images.items():
            kept = []
            for ann in annotations_by_image.get(old_image_id, []):
                target = aliases.get(normalize(categories.get(ann.get("category_id"), "")))
                if target is None:
                    stats["annotations_skipped_by_class"] += 1
                    continue
                if not ann.get("segmentation"):
                    stats["annotations_skipped_without_segmentation"] += 1
                    continue
                kept.append((ann, target))

            src_image = image_path(annotation_path, old_image["file_name"])
            if not kept:
                stats["images_skipped_without_target_annotations"] += 1
                continue
            if src_image is None:
                stats["images_skipped_missing_file"] += 1
                continue

            split = split_by_image[old_image_id]
            out_dir = output_root / "images" / split
            new_file_name = unique_name(old_image["file_name"], used_image_names[split])
            shutil.copy2(src_image, out_dir / new_file_name)

            new_image_id = next_image_id[split]
            next_image_id[split] += 1
            new_image = dict(old_image, id=new_image_id, file_name=new_file_name)
            outputs[split]["images"].append(new_image)

            for ann, target in kept:
                new_ann = dict(
                    ann,
                    id=next_annotation_id[split],
                    image_id=new_image_id,
                    category_id=target_id[target],
                )
                next_annotation_id[split] += 1
                outputs[split]["annotations"].append(new_ann)
                stats[f"annotations_kept_{target}"] += 1

    for split, data in outputs.items():
        with (output_root / "annotations" / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    return dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a Roboflow COCO segmentation export into the Phase 1 layout.")
    parser.add_argument("--source", required=True, type=Path, help="Roboflow COCO export zip or extracted folder.")
    parser.add_argument("--output-root", default=Path("dataset"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--class-map", action="append", default=[], metavar="SOURCE=TARGET")
    args = parser.parse_args()

    root, tmp = source_root(args.source)
    try:
        prepare(args.output_root)
        stats = import_coco(root, args.output_root, parse_class_map(args.class_map), args.seed)
    finally:
        if tmp is not None:
            tmp.cleanup()

    print(f"Wrote COCO dataset to {args.output_root}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")
    print("Run: python scripts/validate_coco_dataset.py")
    print("Run: python scripts/visualize_phase1_dataset.py")
    print("Run: python scripts/smoke_test_dataloader.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
