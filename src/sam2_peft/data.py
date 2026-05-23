from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoSegmentationDataset(Dataset):
    """COCO instance annotations converted to one semantic mask per image."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int | None = 1024,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.image_dir = self.dataset_root / "images" / split
        self.annotation_path = self.dataset_root / "annotations" / f"{split}.json"

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {self.annotation_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")

        self.coco = COCO(str(self.annotation_path))
        self.image_ids = sorted(self.coco.getImgIds())
        self.categories = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c["id"])
        self.category_id_to_index = {
            category["id"]: index + 1
            for index, category in enumerate(self.categories)
        }
        self.index_to_category = {
            index + 1: category["name"]
            for index, category in enumerate(self.categories)
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def num_classes(self) -> int:
        return len(self.categories) + 1

    @property
    def num_foreground_classes(self) -> int:
        return len(self.categories)

    @property
    def class_names(self) -> list[str]:
        return ["background"] + [category["name"] for category in self.categories]

    @property
    def foreground_class_names(self) -> list[str]:
        return [category["name"] for category in self.categories]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file referenced by COCO JSON: {image_path}")

        image = Image.open(image_path).convert("RGB")
        mask = self._semantic_mask(image_id, image_info["height"], image_info["width"])

        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask_image = Image.fromarray(mask)
            mask = np.array(mask_image.resize((self.image_size, self.image_size), Image.NEAREST))

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor

    def _semantic_mask(self, image_id: int, height: int, width: int) -> np.ndarray:
        semantic = np.zeros((height, width), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            class_index = self.category_id_to_index[ann["category_id"]]
            instance_mask = self.coco.annToMask(ann).astype(bool)
            semantic[instance_mask] = class_index

        return semantic
