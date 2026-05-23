from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sam2_peft.models.adapters import AdapterBlock


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int

    @property
    def trainable_percent(self) -> float:
        return 100.0 * self.trainable / self.total


class AdapterWrappedBlock(nn.Module):
    """SAM2 Hiera block followed by a residual adapter."""

    def __init__(self, block: nn.Module, bottleneck: int = 64) -> None:
        super().__init__()
        self.block = block
        self.adapter = AdapterBlock(dim=block.dim_out, bottleneck=bottleneck)
        self.dim = block.dim
        self.dim_out = block.dim_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.block(x))


def insert_hiera_adapters(model: nn.Module, bottleneck: int = 3) -> list[AdapterBlock]:
    """Insert adapters after every SAM2 Hiera image-encoder block."""

    trunk = model.image_encoder.trunk
    if not hasattr(trunk, "blocks"):
        raise TypeError("Expected SAM2 image_encoder.trunk to expose a blocks ModuleList.")

    adapters: list[AdapterBlock] = []
    for index, block in enumerate(trunk.blocks):
        if isinstance(block, AdapterWrappedBlock):
            adapters.append(block.adapter)
            continue
        device = next(block.parameters()).device
        wrapped = AdapterWrappedBlock(block, bottleneck=bottleneck).to(device)
        trunk.blocks[index] = wrapped
        adapters.append(wrapped.adapter)
    return adapters


def configure_peft_trainable(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Freeze SAM2 except inserted adapters and the SAM mask decoder."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    adapter_params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, AdapterBlock):
            for parameter in module.parameters():
                parameter.requires_grad = True
                adapter_params.append(parameter)

    decoder_params: list[nn.Parameter] = []
    for parameter in model.sam_mask_decoder.parameters():
        parameter.requires_grad = True
        decoder_params.append(parameter)

    if not adapter_params:
        raise ValueError("No adapter parameters found. Call insert_hiera_adapters first.")

    return adapter_params, decoder_params


def build_peft_optimizer(
    adapter_params: list[nn.Parameter],
    decoder_params: list[nn.Parameter],
    adapter_lr: float = 1e-4,
    decoder_lr: float = 5e-5,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": adapter_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )


def count_parameters(model: nn.Module) -> ParameterCounts:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return ParameterCounts(total=total, trainable=trainable)


def assert_frozen_image_encoder(model: nn.Module) -> None:
    for name, parameter in model.image_encoder.named_parameters():
        if ".adapter." not in name and parameter.grad is not None:
            raise AssertionError(f"Encoder weight {name} has a gradient.")
