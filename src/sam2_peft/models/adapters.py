from __future__ import annotations

import torch
from torch import nn


class AdapterBlock(nn.Module):
    """Residual bottleneck adapter initialized as an identity mapping."""

    def __init__(self, dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x))))

