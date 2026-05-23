"""
Generate viz/phase5_final_comparison.png — 3-panel summary chart across all 4 methods.

Known values (from experiment logs):
  Zero-shot SAM2: mIoU=21.78%, params=0, latency=236.8 ms (measured Phase 2)
  ViT-B/16:       mIoU=42.23%, params=89.6M, latency=19.4 ms (measured Phase 2)
  SAM2 PEFT:      mIoU=80.59%, params=7.8M, latency≈full FT (adapters add <0.5% compute)
  Full fine-tune: mIoU=83.43%, params=224.4M, latency=236.8 ms (same encoder as zero-shot)

PEFT latency note: Phase 4 checkpoint was lost before latency could be measured. Since PEFT
and full fine-tune share the identical SAM2 encoder (only 48 small bottleneck adapters added,
<0.5% extra FLOPs), PEFT latency ≈ full FT latency. Both shown as 237 ms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = PROJECT_ROOT / "viz"
VIZ_DIR.mkdir(exist_ok=True)

# ── All measured / derived numbers ──────────────────────────────────────────
methods    = ["Zero-shot\nSAM2", "ViT-B/16\nbaseline", "SAM2 PEFT\n(ours)", "Full\nfine-tune"]
miou       = [0.2178, 0.4223, 0.8059, 0.8343]
params_m   = [0.0,   89.6,    7.8,   224.4]      # trainable params in millions
# Latency: zero-shot & ViT measured in Phase 2; PEFT & full FT approximated as equal
# (same SAM2 encoder; adapters add <0.5% compute; checkpoint unavailable to re-measure)
latency_ms = [236.8, 19.4, 237.0, 237.0]

colours = ["#aaaaaa", "#5599cc", "#22aa66", "#dd4444"]
x       = np.arange(len(methods))
width   = 0.55

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#f8f9fa")
for ax in axes:
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)

# ── Panel 1: Test mIoU ───────────────────────────────────────────────────────
bars = axes[0].bar(x, miou, width, color=colours, edgecolor="white", linewidth=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, fontsize=9)
axes[0].set_ylabel("Test mIoU", fontsize=10)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Segmentation Accuracy", fontsize=11, fontweight="bold", pad=8)
axes[0].axhline(miou[3], color="#dd4444", linestyle="--", linewidth=0.8, alpha=0.4)
for bar, val in zip(bars, miou):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, val + 0.015,
        f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
    )
# Annotate PEFT recovery ratio
axes[0].annotate(
    "95.4% recovery\nof full FT gain",
    xy=(2, miou[2]), xytext=(2.35, 0.65),
    fontsize=7.5, color="#22aa66",
    arrowprops=dict(arrowstyle="->", color="#22aa66", lw=0.9),
    ha="left"
)

# ── Panel 2: Trainable parameters (log scale) ────────────────────────────────
params_plot = [max(p, 0.05) for p in params_m]  # avoid log(0)
bars2 = axes[1].bar(x, params_plot, width, color=colours, edgecolor="white", linewidth=0.8)
axes[1].set_yscale("log")
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, fontsize=9)
axes[1].set_ylabel("Trainable parameters (M, log scale)", fontsize=10)
axes[1].set_title("Parameter Efficiency", fontsize=11, fontweight="bold", pad=8)
labels = ["0 M", "89.6 M", "7.8 M\n(3.42%)", "224.4 M\n(100%)"]
for bar, val, label in zip(bars2, params_plot, labels):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2, val * 2.2,
        label, ha="center", va="bottom", fontsize=8, fontweight="bold"
    )
axes[1].set_ylim(0.02, 2000)

# ── Panel 3: Inference latency ───────────────────────────────────────────────
bars3 = axes[2].bar(x, latency_ms, width, color=colours, edgecolor="white", linewidth=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods, fontsize=9)
axes[2].set_ylabel("Inference latency (ms / image)", fontsize=10)
axes[2].set_title("Inference Speed", fontsize=11, fontweight="bold", pad=8)
for bar, val in zip(bars3, latency_ms):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2, val + 3,
        f"{val:.0f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
    )
axes[2].set_ylim(0, 280)
# Note about PEFT latency approximation
axes[2].text(
    0.5, -0.18, "* PEFT latency ≈ full FT (same encoder, adapters add <0.5% compute;\n"
    "  Phase 4 checkpoint unavailable for direct measurement)",
    transform=axes[2].transAxes, ha="center", va="top", fontsize=6.5, color="#888888"
)

plt.suptitle(
    "SAM2 PEFT for Humanoid Robot Segmentation — All Methods Comparison",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(VIZ_DIR / "phase5_final_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved to {VIZ_DIR / 'phase5_final_comparison.png'}")
print("\nKey numbers:")
for m, iou, p, lat in zip(methods, miou, params_m, latency_ms):
    name = m.replace("\n", " ")
    print(f"  {name:25s}  mIoU={iou*100:.1f}%  params={p:.1f}M  latency={lat:.0f}ms")
