#!/usr/bin/env python3
"""
model_cost.py — quantify size & inference cost for timm Swin/ViT models.

Reports:
- #parameters, #buffers
- Weight & buffer memory (by current dtype)
- Estimated on-disk size (weights + buffers)
- FLOPs/MACs for a single forward pass (best-effort via fvcore/ptflops/thop)
- Latency (ms) and throughput (items/s) from a timed forward pass
- Peak runtime memory (proxy for activations + workspaces) on the chosen device

Usage examples:
  python model_cost.py --model swin_base_patch4_window7_224 --img-size 224 --batch 1 --dtype fp16 --device cuda
  python model_cost.py --model vit_base_patch16_384 --img-size 384 --batch 8 --dtype bf16 --device cuda
  python model_cost.py --model swin_tiny_patch4_window7_224 --device cpu
"""

import argparse
import time
import math
import warnings
from contextlib import nullcontext

import torch
import timm

# ---------------------------
# Utilities
# ---------------------------

def human_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"

def resolve_dtype(dtype_str, device):
    s = dtype_str.lower()
    if s in ["fp32","float32"]:
        return torch.float32
    if s in ["fp16","float16","half"]:
        # bf16 is often safer on Ampere+; allow fp16 if requested
        return torch.float16
    if s in ["bf16","bfloat16"]:
        # CPU supports bf16 tensors but not all ops; cuda bf16 needs Ampere+
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype_str}")

def device_ctx(device, autocast_dtype):
    if device == "cuda":
        # Prefer autocast for non-fp32; it keeps numerics stable & speed kernels
        if autocast_dtype == torch.float32:
            return nullcontext()
        # autocast supports both bf16 and fp16 on CUDA (hardware permitting)
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    # CPU autocast only supports float32/bf16 at the moment
    if device == "cpu":
        if autocast_dtype == torch.float32:
            return nullcontext()
        if autocast_dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        # fp16 on CPU is not supported for most ops; ignore autocast
        return nullcontext()
    return nullcontext()

def count_params_buffers(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    return n_params, n_buffers

def bytes_of_tensors(tensors):
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total

def model_weights_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def model_buffers_bytes(model):
    return sum(b.numel() * b.element_size() for b in model.buffers())

# ---------------------------
# FLOPs/MACs estimators
# ---------------------------

def try_fvcore(model, inputs):
    """
    Returns (macs, flops) as integers if possible, else None.
    fvcore FlopCountAnalysis reports FLOPs; we derive MACs = FLOPs / 2 for conv/linear heavy nets.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fca = FlopCountAnalysis(model, inputs)
            flops = int(fca.total())
            macs = flops // 2
            return macs, flops
    except Exception as e:
        return None

def try_ptflops(model, inputs):
    """
    ptflops returns MACs (multiply-adds counted as a single op by default).
    """
    try:
        from ptflops import get_model_complexity_info

        # ptflops wants (C,H,W) without batch
        x = inputs
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() != 4:
            return None
        c, h, w = x.shape[1], x.shape[2], x.shape[3]

        # We need to run on CPU for analysis safety
        model_cpu = model.to("cpu")
        model_cpu.eval()
        with torch.no_grad():
            macs_str, params_str = get_model_complexity_info(
                model_cpu, (c, h, w), as_strings=True, print_per_layer_stat=False, verbose=False
            )
        # macs_str like '4.23 GMac'
        units = {"K":1e3, "M":1e6, "G":1e9, "T":1e12}
        val, unit = macs_str.split()[0], macs_str.split()[1]
        scale = 1.0
        for k in units:
            if unit.upper().startswith(k.upper()):
                scale = units[k]
                break
        macs = int(float(val) * scale)
        flops = macs * 2
        return macs, flops
    except Exception:
        return None

def try_thop(model, inputs):
    """
    thop.profile returns macs & params. macs = number of MAC operations.
    """
    try:
        from thop import profile
        model_cpu = model.to("cpu")
        model_cpu.eval()
        x = inputs
        if isinstance(x, (list, tuple)):
            x = x[0]
        with torch.no_grad():
            macs, params = profile(model_cpu, inputs=(x,), verbose=False)
        flops = int(macs * 2)
        return int(macs), flops
    except Exception:
        return None

def estimate_macs_flops(model, inputs):
    """
    Try multiple backends. Returns (macs, flops) or (None, None) if all fail.
    """
    for fn in (try_fvcore, try_ptflops, try_thop):
        out = fn(model, inputs)
        if out is not None:
            return out
    return None, None

# ---------------------------
# Timing & memory profiling
# ---------------------------

def benchmark(model, inputs, device, dtype_autocast, warmup=5, iters=20):
    """
    Returns (avg_latency_ms, throughput_items_per_s, peak_mem_bytes)
    Peak memory only available/meaningful on CUDA; for CPU we return None.
    """
    model.eval()
    no_sync = torch.no_grad()
    ctx = device_ctx(device, dtype_autocast)

    # Prepare memory stats
    peak_mem = None
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Warmup
    with no_sync, ctx:
        for _ in range(max(0, warmup)):
            _ = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    times = []
    with no_sync, ctx:
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    avg_ms = sum(times) / len(times)
    batch = inputs[0].shape[0] if isinstance(inputs, (list, tuple)) else inputs.shape[0]
    throughput = (batch / (avg_ms / 1000.0)) if avg_ms > 0 else float("nan")

    if device == "cuda" and torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated()

    return avg_ms, throughput, peak_mem

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="timm model name (e.g., swin_base_patch4_window7_224, vit_base_patch16_224)")
    parser.add_argument("--img-size", type=int, default=None, help="Square image size (H=W). If omitted, use model.default_cfg['input_size'] or 224.")
    parser.add_argument("--channels", type=int, default=None, help="Input channels. If omitted, use model.default_cfg['input_size'][0] or 3.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for analysis.")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32","bf16","fp16"], help="Precision for inference run and size calc.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda","cpu"], help="Device to test on.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations for latency/throughput.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing.")
    args = parser.parse_args()

    # Create model
    model = timm.create_model(args.model, pretrained=False)
    model.eval()

    # Infer default input shape from timm cfg if not provided
    in_cfg = getattr(model, "default_cfg", {}) or {}
    cfg_input = in_cfg.get("input_size", None)  # (C, H, W)
    C = args.channels or (cfg_input[0] if cfg_input else 3)
    HW = args.img_size or (cfg_input[1] if cfg_input else 224)
    H = W = HW

    # Resolve dtype & device
    device = args.device
    dtype = resolve_dtype(args.dtype, device)

    # Move model to device (weights dtype)
    # We keep parameters in the autocast-friendly dtype only if fp32 is not requested.
    # This keeps the weight memory estimate aligned with actual tensor dtypes.
    if device == "cuda":
        model = model.to(device)
        if dtype in (torch.float16, torch.bfloat16):
            model = model.to(dtype)
    else:
        model = model.to("cpu")
        # CPU: prefer float32 weights, even if bf16 autocast is used during forward
        # (bf16 params on CPU may not be supported for all ops)
        if dtype == torch.float16:
            warnings.warn("CPU + fp16 is not generally supported; falling back to fp32 weights.")

    # Build dummy input
    x = torch.randn(args.batch, C, H, W)
    if device == "cuda":
        x = x.to(device)
        # For fp16 weights, inputs can remain fp32; autocast will handle casts.
        # If no autocast (fp32), input stays fp32.
    else:
        x = x.to("cpu")

    # Count parameters & buffers
    n_params, n_buffers = count_params_buffers(model)
    weight_bytes = model_weights_bytes(model)
    buffer_bytes = model_buffers_bytes(model)
    est_ondisk = weight_bytes + buffer_bytes  # rough .pt size sans metadata

    # FLOPs / MACs (best effort on CPU copy to avoid CUDA graph capture issues)
    macs, flops = estimate_macs_flops(model, x)

    # Benchmark: latency, throughput, peak memory
    avg_ms, throughput, peak_mem = benchmark(model, x, device, dtype, warmup=args.warmup, iters=args.iters)

    # Print report
    print("\n=== Model Cost Report ===")
    print(f"Model:               {args.model}")
    print(f"Device / DType:      {device} / {args.dtype}")
    print(f"Input:               batch={args.batch}, shape=({C},{H},{W})")
    print()
    print(f"Parameters:          {n_params:,}")
    print(f"Buffers:             {n_buffers:,}")
    print(f"Weights memory:      {human_bytes(weight_bytes)}")
    print(f"Buffers memory:      {human_bytes(buffer_bytes)}")
    print(f"~On-disk (est.):     {human_bytes(est_ondisk)}  (weights+buffers; actual checkpoint may vary)")
    if macs is not None and flops is not None:
        print(f"Compute (per forward):")
        print(f"  MACs:              {macs:,}")
        print(f"  FLOPs (~2×MACs):   {flops:,}")
    else:
        print("Compute (per forward):  unavailable (fvcore/ptflops/thop not found or unsupported model)")
    print()
    print("Runtime (measured):")
    print(f"  Avg latency:       {avg_ms:.2f} ms  (over {args.iters} iters, warmup={args.warmup})")
    print(f"  Throughput:        {throughput:.2f} items/s")
    if peak_mem is not None:
        print(f"  Peak allocated:    {human_bytes(peak_mem)}  (includes activations + workspaces)")
    else:
        print("  Peak allocated:    N/A on CPU")
    print()
    print("Notes:")
    print("- Peak memory is a practical upper bound for activations/workspaces with this input/batch/precision.")
    print("- For more realistic numbers, run on your actual device and dtype, with your target batch/resolution.")
    print("- FLOPs/MACs are per forward pass of the specified input size; they scale with resolution and batch.")
    print("- If compute is 'unavailable', install one of: fvcore, ptflops, thop.")

if __name__ == "__main__":
    main()

