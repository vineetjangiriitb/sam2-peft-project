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
  - `head`

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

1. Search Roboflow Universe for instance-segmentation projects with robot-part labels. Prefer datasets with a clear open license and polygon masks, not bounding boxes.
2. Candidate sources to inspect first:
   - [`so101-segmentation-v4`](https://universe.roboflow.com/adityas-workspace-kukhm/so101-segmentation-v4) / [`so101-segementation-v3`](https://universe.roboflow.com/adityas-workspace-kukhm/so101-segementation-v3): useful robot-arm labels. These are small and may need to be combined with manual annotation.
   - [`Components`](https://universe.roboflow.com/robot-pose-annotation/components-zx3kz) by Robot Pose Annotation: large CC BY 4.0 instance-segmentation dataset with component labels. It is not humanoid-specific, so use only if the visual content is acceptable after spot-checking.
   - `AGV_kami`: larger instance-segmentation source with component labels. Use only if the images match the project scope closely enough after visual inspection.
   - Additional Roboflow searches: `class:"robot arm" instance segmentation`, `humanoid robot head segmentation`, `humanoid robot segmentation`.
3. Download the chosen project version as COCO segmentation. Do not download YOLO/object-detection exports for Phase 1.
4. Import the downloaded zip or extracted folder:

```bash
python scripts/import_roboflow_coco.py --source path/to/roboflow-export.zip
```

Add explicit label mappings when a source uses custom names:

```bash
python scripts/import_roboflow_coco.py \
  --source path/to/roboflow-export.zip \
  --class-map so101_upperarm=arm \
  --class-map robot_head=head
```

5. Remove duplicates, near-duplicates, tiny images, blurry images, and images where components are not visible.
6. Manually annotate the missing classes if the imported dataset has fewer than 350 images, fewer than 3 classes, or poor class balance.
7. Run:

```bash
python scripts/validate_coco_dataset.py
python scripts/visualize_phase1_dataset.py
python scripts/smoke_test_dataloader.py
```

The importer normalizes common robot-arm labels into the project classes and writes:

```text
dataset/images/{train,val,test}/
dataset/annotations/{train,val,test}.json
```

It does not prove Phase 1 is complete. The validation and visualization checks above are still mandatory.

## Annotation Rules

- Annotate only visible pixels.
- Do not hallucinate occluded parts.
- Use one polygon per visible component instance when possible.
- Keep labels consistent across train, val, and test.
- If left/right components are both labeled, use the same class name, for example both arms are `arm`.
- Background is not a COCO category.
- Avoid bounding-box-only exports; every annotation must include `segmentation`.

## AI-Assisted Labeling

Use Roboflow Smart Polygon, SAM-assisted labeling, or Auto Label to accelerate annotation, but still review masks before accepting them as training ground truth. In particular, check that `arm`, `leg`, `torso`, and `head` are separated consistently, because automatic masks often find object boundaries without knowing the project-specific part taxonomy.

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
