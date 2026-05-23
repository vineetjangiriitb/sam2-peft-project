from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO


@dataclass(frozen=True)
class PromptSample:
    image: np.ndarray
    mask: np.ndarray
    point_coords: np.ndarray
    image_id: int
    annotation_id: int
    class_name: str
    file_name: str


class CocoPromptInstanceDataset:
    """One SAM2 point-prompt training sample per COCO instance annotation."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        sample_limit: int | None = None,
        image_limit: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_dir = self.dataset_root / "images" / split
        self.annotation_path = self.dataset_root / "annotations" / f"{split}.json"
        self.coco = COCO(str(self.annotation_path))
        self.category_id_to_name = {
            category["id"]: category["name"]
            for category in self.coco.loadCats(self.coco.getCatIds())
        }

        samples: list[tuple[int, int]] = []
        image_ids = sorted(self.coco.getImgIds())
        if image_limit is not None:
            image_ids = image_ids[:image_limit]

        for image_id in image_ids:
            ann_ids = sorted(self.coco.getAnnIds(imgIds=image_id))
            samples.extend((image_id, ann_id) for ann_id in ann_ids)
        self.samples = samples[:sample_limit] if sample_limit is not None else samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PromptSample:
        image_id, annotation_id = self.samples[index]
        image_info = self.coco.loadImgs(image_id)[0]
        ann = self.coco.loadAnns([annotation_id])[0]
        image_path = self.image_dir / image_info["file_name"]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self.coco.annToMask(ann).astype(np.float32)
        point_coords = bbox_center(ann["bbox"])

        return PromptSample(
            image=image,
            mask=mask,
            point_coords=point_coords,
            image_id=image_id,
            annotation_id=annotation_id,
            class_name=self.category_id_to_name[ann["category_id"]],
            file_name=image_info["file_name"],
        )


def bbox_center(bbox: list[float]) -> np.ndarray:
    x, y, width, height = [float(value) for value in bbox]
    return np.array([[x + width / 2.0, y + height / 2.0]], dtype=np.float32)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(logits).flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return focal_loss(logits, target) + dice_loss(logits, target)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sam2_model(model_id: str, device: str, mode: str = "train"):
    try:
        from sam2.build_sam import build_sam2_hf
    except ImportError as exc:
        raise SystemExit(
            "SAM2 is not installed. Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        ) from exc
    return build_sam2_hf(model_id, device=device, mode=mode)


def make_transforms(model):
    from sam2.utils.transforms import SAM2Transforms

    return SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)


def forward_prompt_sample(model, transforms, sample: PromptSample, device: torch.device) -> torch.Tensor:
    input_image = transforms(sample.image).unsqueeze(0).to(device)
    backbone_out = model.forward_image(input_image)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]

    coords = torch.as_tensor(sample.point_coords, dtype=torch.float32, device=device)
    coords = transforms.transform_coords(coords, normalize=True, orig_hw=sample.image.shape[:2])
    labels = torch.ones(coords.shape[0], dtype=torch.int64, device=device)
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(coords.unsqueeze(0), labels.unsqueeze(0)),
        boxes=None,
        masks=None,
    )

    low_res_masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=feats[-1],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=feats[:-1],
    )
    return low_res_masks[:, 0]


def low_res_target(sample: PromptSample, output_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    target = torch.as_tensor(sample.mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    target = F.interpolate(target, size=output_hw, mode="nearest")
    return target[:, 0]


def train_one_sample(model, transforms, sample: PromptSample, optimizer, device: torch.device) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = forward_prompt_sample(model, transforms, sample, device)
    target = low_res_target(sample, logits.shape[-2:], device)
    loss = combined_loss(logits, target)
    if torch.isnan(loss):
        raise AssertionError("NaN loss detected.")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def predict_instance_mask(model, transforms, sample: PromptSample, device: torch.device) -> np.ndarray:
    logits = forward_prompt_sample(model, transforms, sample, device)
    logits = logits.unsqueeze(1)
    upsampled = transforms.postprocess_masks(logits, sample.image.shape[:2])
    return (upsampled[0, 0] > 0).detach().cpu().numpy()


def binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return math.nan
    return float(np.logical_and(pred, gt).sum() / union)
