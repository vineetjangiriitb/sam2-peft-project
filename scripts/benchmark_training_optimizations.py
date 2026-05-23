from __future__ import annotations

"""Benchmark training optimizations by running one epoch with each combination.

Measures real wall-clock time and GPU memory for:
  baseline  : no optimizations (sequential, fp32, no cache)
  +ram_cache: preload all images to pinned RAM
  +prefetch : async CPU→GPU overlap
  +amp_bf16 : mixed precision BF16
  +compile  : torch.compile reduce-overhead
  +batch_dec: batched decoder
  all       : all optimizations combined

Reports a table of epoch time and GPU memory for each configuration.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint as grad_ckpt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sam2_peft.phase3 import (
    CocoPromptInstanceDataset,
    combined_loss,
    load_sam2_model,
    low_res_target,
    make_transforms,
    resolve_device,
)
from train_full_finetune import (
    ImagePrefetcher,
    _nullctx,
    build_ram_cache,
    configure_full_finetune,
    decode_batch_from_features,
    enable_gradient_checkpointing,
    grouped_by_image,
)


# ---------------------------------------------------------------------------
# One-epoch benchmark runner
# ---------------------------------------------------------------------------

def run_one_epoch(
    model, transforms, optimizer, scaler,
    train_dataset, image_groups, device,
    use_amp: bool,
    ram_cache: dict | None,
    use_prefetch: bool,
    use_batch_decode: bool,
    max_images: int | None = None,
) -> tuple[float, float]:
    """Run one training epoch. Returns (elapsed_seconds, peak_gpu_gb)."""
    amp_dtype = torch.bfloat16 if use_amp else None
    all_params = [p for p in model.parameters() if p.requires_grad]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prefetcher = None
    if use_prefetch:
        prefetcher = ImagePrefetcher(image_groups, train_dataset, transforms, ram_cache, device)
        prefetcher.prime()

    model.train()
    t0 = time.time()
    groups = image_groups[:max_images] if max_images else image_groups

    for img_idx, ann_indices in enumerate(groups, start=1):
        # --- get input tensor ---
        if prefetcher is not None:
            input_tensor = prefetcher.next()
        elif ram_cache is not None:
            s0 = train_dataset[ann_indices[0]]
            input_tensor = ram_cache[s0.image_id].to(device, non_blocking=True)
        else:
            s0 = train_dataset[ann_indices[0]]
            input_tensor = transforms(s0.image).unsqueeze(0)
            if device.type == "cuda":
                input_tensor = input_tensor.pin_memory()
            input_tensor = input_tensor.to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else _nullctx()

        with amp_ctx:
            backbone_out = model.forward_image(input_tensor)

        optimizer.zero_grad(set_to_none=True)
        samples = [train_dataset[i] for i in ann_indices]

        if use_batch_decode:
            with amp_ctx:
                logits_list = decode_batch_from_features(
                    model, transforms, backbone_out, samples, device, amp_dtype,
                )
            loss = torch.tensor(0.0, device=device)
            for logits, sample in zip(logits_list, samples):
                target = low_res_target(sample, logits.shape[-2:], device)
                loss   = loss + combined_loss(logits, target) / len(samples)
            if not torch.isnan(loss):
                scaler.scale(loss).backward(retain_graph=False)
        else:
            for sample in samples:
                with amp_ctx:
                    coords = torch.as_tensor(sample.point_coords, dtype=torch.float32, device=device)
                    coords = transforms.transform_coords(coords, normalize=True, orig_hw=sample.image.shape[:2])
                    labels = torch.ones(coords.shape[0], dtype=torch.int64, device=device)
                    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
                    if model.directly_add_no_mem_embed:
                        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
                    feats = [
                        feat.permute(1, 2, 0).view(1, -1, *fs)
                        for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])
                    ][::-1]
                    sparse_emb, dense_emb = model.sam_prompt_encoder(
                        points=(coords.unsqueeze(0), labels.unsqueeze(0)), boxes=None, masks=None,
                    )
                    low_res_masks, _, _, _ = model.sam_mask_decoder(
                        image_embeddings=feats[-1],
                        image_pe=model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False, repeat_image=False, high_res_features=feats[:-1],
                    )
                    logits = low_res_masks[:, 0]
                    target = low_res_target(sample, logits.shape[-2:], device)
                    loss   = combined_loss(logits, target) / len(samples)
                if not torch.isnan(loss):
                    scaler.scale(loss).backward(retain_graph=True)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer)
        scaler.update()

    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    if prefetcher:
        prefetcher.shutdown()

    return elapsed, peak_gb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark training optimizations.")
    parser.add_argument("--dataset-root", default=Path("dataset"), type=Path)
    parser.add_argument("--model-id",     default="facebook/sam2.1-hiera-large")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--max-images",   default=50, type=int,
                        help="Images per epoch (use subset for faster benchmark)")
    parser.add_argument("--output",       default=Path("outputs/benchmark_optimizations.json"), type=Path)
    parser.add_argument("--no-grad-ckpt", action="store_true")
    args = parser.parse_args()

    device_name = resolve_device(args.device)
    device      = torch.device(device_name)
    print(f"Device: {device_name}  |  images per run: {args.max_images}", flush=True)

    train_dataset = CocoPromptInstanceDataset(args.dataset_root, split="train")
    image_groups  = grouped_by_image(train_dataset)[:args.max_images]

    # Configurations to benchmark — each is a dict of flags
    configs = [
        {"name": "baseline",              "amp": False, "ram": False, "prefetch": False, "batch": False},
        {"name": "+ram_cache",            "amp": False, "ram": True,  "prefetch": False, "batch": False},
        {"name": "+ram+prefetch",         "amp": False, "ram": True,  "prefetch": True,  "batch": False},
        {"name": "+ram+prefetch+amp_bf16","amp": True,  "ram": True,  "prefetch": True,  "batch": False},
        {"name": "+all (incl batch_dec)", "amp": True,  "ram": True,  "prefetch": True,  "batch": True},
    ]

    results = []

    for cfg in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {cfg['name']}", flush=True)

        # Fresh model for each config to avoid state leakage
        model = load_sam2_model(args.model_id, device_name, mode="train")
        all_params, _ = configure_full_finetune(model)

        if not args.no_grad_ckpt and device.type == "cuda":
            enable_gradient_checkpointing(model)

        optimizer = torch.optim.AdamW(all_params, lr=1e-5, weight_decay=1e-4)
        scaler    = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
        transforms = make_transforms(model)

        ram_cache = None
        if cfg["ram"]:
            ram_cache = build_ram_cache(train_dataset, transforms, image_groups, device)

        # Warmup pass — first forward is slow due to CUDA context init
        print("Warming up...", flush=True)
        s0   = train_dataset[image_groups[0][0]]
        inp  = transforms(s0.image).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model.forward_image(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()

        print("Running benchmark epoch...", flush=True)
        elapsed, peak_gb = run_one_epoch(
            model, transforms, optimizer, scaler,
            train_dataset, image_groups, device,
            use_amp=cfg["amp"],
            ram_cache=ram_cache,
            use_prefetch=cfg["prefetch"],
            use_batch_decode=cfg["batch"],
            max_images=args.max_images,
        )

        imgs_per_sec = args.max_images / elapsed
        # Extrapolate to full 280-image epoch
        full_epoch_est = elapsed * (280 / args.max_images)

        print(f"  elapsed:        {elapsed:.1f}s  ({imgs_per_sec:.2f} img/s)", flush=True)
        print(f"  peak GPU mem:   {peak_gb:.2f} GB", flush=True)
        print(f"  full epoch est: {full_epoch_est:.0f}s  ({full_epoch_est/60:.1f} min)", flush=True)

        results.append({
            "config":           cfg["name"],
            "elapsed_s":        round(elapsed, 1),
            "imgs_per_sec":     round(imgs_per_sec, 2),
            "peak_gpu_gb":      round(peak_gb, 2),
            "full_epoch_est_s": round(full_epoch_est, 0),
        })

        # Cleanup
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*60}", flush=True)
    print(f"{'Config':<30} {'Time(s)':>8} {'img/s':>7} {'GPU GB':>8} {'Full epoch':>12}", flush=True)
    print("-" * 70, flush=True)
    baseline_t = results[0]["elapsed_s"]
    for r in results:
        speedup = baseline_t / r["elapsed_s"]
        print(f"{r['config']:<30} {r['elapsed_s']:>8.1f} {r['imgs_per_sec']:>7.2f} "
              f"{r['peak_gpu_gb']:>8.2f} {r['full_epoch_est_s']:>8.0f}s "
              f"  ({speedup:.2f}x)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
