# Dataset Construction Guide

Phase 1 builds the dataset required by `SAM2_PEFT_Project_Plan.md`.

## Target

- 350-500 total humanoid robot images.
- COCO instance segmentation format with polygon masks.
- Split: 70% train, 15% val, 15% test.
- Minimum 3 usable component classes.
- Target classes:
  - `arm`
  - `leg`
  - `torso`
  - `joint`
  - `camera_sensor`
  - `gripper`

If one of the target classes is too rare to annotate consistently, document it before merging or dropping it.

## Directory Layout

Place exported data here:

```text
dataset/
  images/
    train/
    val/
    test/
  annotations/
    train.json
    val.json
    test.json
```

Each COCO JSON must reference images by filename only, not absolute local paths.

## Recommended Build Path

1. Collect candidate humanoid robot images.
2. Remove duplicates, near-duplicates, tiny images, blurry images, and images where components are not visible.
3. Annotate segmentation polygons for visible robot components.
4. Export to COCO segmentation format.
5. Put images and JSON files into the directory layout above.
6. Run:

```bash
python scripts/validate_coco_dataset.py
python scripts/visualize_phase1_dataset.py
python scripts/smoke_test_dataloader.py
```

## Annotation Rules

- Annotate only visible pixels.
- Do not hallucinate occluded parts.
- Use one polygon per visible component instance when possible.
- Keep labels consistent across train, val, and test.
- If left/right components are both labeled, use the same class name, for example both arms are `arm`.
- Background is not a COCO category.
- Avoid bounding-box-only exports; every annotation must include `segmentation`.

## Phase 1 Pass Checks

Phase 1 is not complete until:

- `dataset/annotations/train.json` has at least 280 images.
- All annotations include segmentation masks.
- At least 3 categories are present.
- `viz/phase1_mask_alignment.png` shows masks aligned correctly in at least 4 out of 5 validation images.
- `viz/phase1_class_balance.png` shows no class above 60% of all annotations.
- A dataloader smoke test can load a batch without errors.

Expected dataloader smoke output includes:

```text
Image batch shape: (2, 3, 1024, 1024)
Mask batch shape:  (2, 1024, 1024)
PASS dataloader smoke test
```
