from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sam2_peft.models.adapters import AdapterBlock


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic AdapterBlock checks.")
    parser.add_argument("--dim", default=256, type=int)
    parser.add_argument("--bottleneck", default=64, type=int)
    args = parser.parse_args()

    torch.manual_seed(7)
    adapter = AdapterBlock(dim=args.dim, bottleneck=args.bottleneck)
    x = torch.randn(1, 100, args.dim)
    out = adapter(x)
    max_diff = (out - x).abs().max().item()
    total_params = count_parameters(adapter)

    print(f"Adapter dim: {args.dim}")
    print(f"Bottleneck: {args.bottleneck}")
    print(f"Parameters: {total_params}")
    print(f"Max identity diff: {max_diff:.8f}")

    if max_diff >= 1e-6:
        raise AssertionError(f"Adapter should start as identity, got max diff {max_diff}")

    loss = out.sum()
    loss.backward()
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")
    print("PASS adapter identity check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

