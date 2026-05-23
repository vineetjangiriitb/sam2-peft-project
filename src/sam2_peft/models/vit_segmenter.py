from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ViTSegmenter(nn.Module):
    """ViT-B/16 backbone with a ConvTranspose2d decoder for semantic segmentation.

    Input:  (N, 3, 224, 224) float32, values in [0, 1]
    Output: (N, num_classes, 224, 224) raw logits
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = tv_models.vit_b_16(weights=weights)

        # Keep everything except the classification head.
        # vit.encoder outputs (N, 197, 768): [CLS] + 196 patch tokens.
        self.patch_embedding = vit.conv_proj          # (N, 768, 14, 14) after reshape
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        self.patch_size = 16
        self.hidden_dim = 768
        self.grid_size = 14  # 224 / 16

        # Upsample 14x14 → 224x224 in three stages (×2 each then ×2 again = ×8)
        # 14 → 28 → 56 → 112 → 224 using 4 transpose convs
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),  # 14→28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28→56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 56→112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 112→224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),                         # channel projection
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]

        # Patch embedding: (N, 768, 14, 14) → flatten to (N, 196, 768)
        patch_tokens = self.patch_embedding(x)  # (N, 768, 14, 14)
        patch_tokens_flat = patch_tokens.flatten(2).transpose(1, 2)  # (N, 196, 768)

        # Prepend CLS token
        cls = self.class_token.expand(n, -1, -1)  # (N, 1, 768)
        tokens = torch.cat([cls, patch_tokens_flat], dim=1)  # (N, 197, 768)

        # Add positional embedding from the encoder
        tokens = tokens + self.encoder.pos_embedding
        tokens = self.encoder.dropout(tokens)
        tokens = self.encoder.layers(tokens)
        tokens = self.encoder.ln(tokens)

        # Drop CLS, reshape patch tokens back to spatial grid
        patch_out = tokens[:, 1:, :]  # (N, 196, 768)
        spatial = patch_out.transpose(1, 2).reshape(n, self.hidden_dim, self.grid_size, self.grid_size)

        return self.decoder(spatial)  # (N, num_classes, 224, 224)
