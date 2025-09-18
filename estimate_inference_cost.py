#!/usr/bin/env python3
"""
model_fitcheck.py — Measure size & inference cost for timm Swin/ViT, plus an iOS/edge "fit check".

What you get:
- #params, #buffers
- Weight & buffer memory at current param dtype (and what-if FP32/FP16/INT8/INT4)
- FLOPs/MACs per forward (best-effort via fvcore/ptflops/thop)
- Timed latency / throughput on your machine
- Peak runtime memory (proxy for activations + workspaces) on CUDA
- "Fit Check" vs a target memory budget, with configurable iOS headroom

Usage example:
  python model_fitcheck.py --model vit_base_patch16_224 --img-size 224 --batch 1 \
    --dtype fp16 --device cuda --budget-mb 4000 --ios-headroom 0.5
"""

import argparse
import time
import warnings
from contextlib import nullcontext

import torch
import timm

# ---------------------------
# Helpers
# ---------------------------

def human_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"

def resolve_dtype(s):
    s = s.lower()
    if s in ("fp32","float32"): return torch.float32
    if s in ("fp16","float16","half"): return torch.float16
    if s in ("bf16","bfloat16"): return torch.bfloat16
    raise ValueError(f"Unknown dtype {s}")

def device_ctx(device, autocast_dtype):
    if device == "cuda":
        if autocast_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    if device == "cpu":
        # CPU autocast supports bf16; fp16 mostly unsupported on CPU
        if autocast_dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()
    return nullcontext()

def count_params_buffers(model):
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    return n_params, n_buffers

def bytes_of_params(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def bytes_of_buffers(model):
    return sum(b.numel() * b.element_size() for b in model.buffers())

def bytes_if_quantized(n_params, dtype_bytes):
    return n_params * dtype_bytes

# ---------------------------
# FLOPs/MACs estimators (best-effort)
# ---------------------------

def try_fvcore(model, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fca = FlopCountAnalysis(model, inputs)
            flops = int(fca.total())
            macs = flops // 2
            return macs, flops
    except Exception:
        return None

def try_ptflops(model, inputs):
    try:
        from ptflops import get_model_complexity_info
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        if x.dim() != 4:
            return None
        c, h, w = x.shape[1], x.shape[2], x.shape[3]
        m_cpu = model.to("cpu").eval()
        with torch.no_grad():
            macs_str, _ = get_model_complexity_info(
                m_cpu, (c, h, w), as_strings=True, print_per_layer_stat=False, verbose=False
            )
        units = {"K":1e3, "M":1e6, "G":1e9, "T":1e12}
        val, unit = macs_str.split()[0], macs_str.split()[1]
        scale = next((v for k,v in units.items() if unit.upper().startswith(k)), 1.0)
        macs = int(float(val) * scale); flops = macs * 2
        return macs, flops
    except Exception:
        return None

def try_thop(model, inputs):
    try:
        from thop import profile
        m_cpu = model.to("cpu").eval()
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        with torch.no_grad():
            macs, _ = profile(m_cpu, inputs=(x,), verbose=False)
        return int(macs), int(macs)*2
    except Exception:
        return None

def estimate_macs_flops(model, inputs):
    for fn in (try_fvcore, try_ptflops, try_thop):
        out = fn(model, inputs)
        if out is not None:
            return out
    return None, None

# ---------------------------
# Benchmarking
# ---------------------------

def benchmark(model, inputs, device, dtype_autocast, warmup=5, iters=20):
    model.eval()
    ctx = device_ctx(device, dtype_autocast)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    with torch.no_grad(), ctx:
        for _ in range(max(0, warmup)):
            _ = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad(), ctx:
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    avg_ms = sum(times_ms)/len(times_ms)
    batch = (inputs[0].shape[0] if isinstance(inputs, (list, tuple)) else inputs.shape[0])
    throughput = batch / (avg_ms/1000.0) if avg_ms > 0 else float("nan")
    peak_mem = torch.cuda.max_memory_allocated() if (device=="cuda" and torch.cuda.is_available()) else None
    return avg_ms, throughput, peak_mem

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="timm model name (e.g., swin_tiny_patch4_window7_224, vit_base_patch16_224)")
    ap.add_argument("--img-size", type=int, default=None, help="Square size (H=W). Defaults to model cfg or 224.")
    ap.add_argument("--channels", type=int, default=None, help="Input channels. Defaults to cfg or 3.")
    ap.add_argument("--batch", type=int, default=1, help="Batch size.")
    ap.add_argument("--dtype", choices=["fp32","bf16","fp16"], default="fp32", help="Param dtype & autocast target.")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=20, help="Timing iterations.")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    ap.add_argument("--budget-mb", type=float, default=4000.0, help="Target device memory budget in MB to “fit check”.")
    ap.add_argument("--ios-headroom", type=float, default=0.50,
                    help="Extra fraction to add for iOS/edge overhead (e.g., 0.5 = +50%).")
    args = ap.parse_args()

    # Create model
    model = timm.create_model(args.model, num_clases=0, pretrained=True).eval()

    # Input shape
    cfg = getattr(model, "default_cfg", {}) or {}
    input_size = cfg.get("input_size", (3, 224, 224))
    C = args.channels or (input_size[0] if isinstance(input_size, (list, tuple)) else 3)
    HW = args.img_size or (input_size[1] if isinstance(input_size, (list, tuple)) else 224)
    H = W = HW

    # Dtype/device
    dtype = resolve_dtype(args.dtype)
    if args.device == "cuda":
        model = model.to("cuda")
        # Put weights in the chosen dtype for a realistic weight memory reading
        if dtype in (torch.float16, torch.bfloat16):
            model = model.to(dtype)
    else:
        model = model.to("cpu")
        if dtype == torch.float16:
            warnings.warn("CPU fp16 not generally supported; keeping fp32 weights.")

    # Dummy input (fp32; autocast will cast as needed)
    x = torch.randn(args.batch, C, H, W, device=("cuda" if args.device=="cuda" else "cpu"))

    # Counts & sizes
    n_params, n_buffers = count_params_buffers(model)
    weight_bytes = bytes_of_params(model)
    buffer_bytes = bytes_of_buffers(model)
    est_ondisk = weight_bytes + buffer_bytes  # rough

    # What-if weights memory by precision (for Core ML planning)
    dtype_bytes = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }
    weight_bytes_by_prec = {
        k: bytes_if_quantized(n_params, v) for k, v in dtype_bytes.items()
    }

    # FLOPs/MACs
    macs, flops = estimate_macs_flops(model, x)

    # Benchmark
    avg_ms, thr, peak_mem = benchmark(model, x, args.device, dtype, warmup=args.warmup, iters=args.iters)

    # Build “fit” estimates
    # Peak CUDA memory ≈ activations + workspaces + some buffers (excludes weights already resident).
    # On iOS unified memory & framework overhead can be higher; apply a headroom multiplier.
    peak_cuda = peak_mem or 0
    ios_est_runtime_bytes = int(peak_cuda * (1.0 + max(0.0, args.ios_headroom)))

    # Total at inference time must hold weights + runtime.
    total_bytes_current_dtype = weight_bytes + ios_est_runtime_bytes

    # If the user plans a different on-device weight precision, show totals:
    total_bytes_by_prec = {
        k: int(weight_bytes_by_prec[k] + ios_est_runtime_bytes) for k in weight_bytes_by_prec
    }

    # Print report
    print("\n=== Model Fit & Cost Report ===")
    print(f"Model:                  {args.model}")
    print(f"Input:                  batch={args.batch}, shape=({C},{H},{W})")
    print(f"Run env:                device={args.device}, dtype={args.dtype}")
    print()
    print(f"Parameters:             {n_params:,}")
    print(f"Buffers:                {n_buffers:,}")
    print(f"Weights memory (now):   {human_bytes(weight_bytes)}")
    print(f"Buffers memory (now):   {human_bytes(buffer_bytes)}")
    print(f"~On-disk (est.):        {human_bytes(est_ondisk)}")
    if macs is not None:
        print(f"Compute per forward:")
        print(f"  MACs:                 {macs:,}")
        print(f"  FLOPs (~2×MACs):      {flops:,}")
    else:
        print("Compute per forward:    unavailable (install fvcore / ptflops / thop)")
    print()
    print("Runtime (measured on your machine):")
    print(f"  Avg latency:          {avg_ms:.2f} ms  (iters={args.iters}, warmup={args.warmup})")
    print(f"  Throughput:           {thr:.2f} items/s")
    if peak_mem is not None:
        print(f"  Peak CUDA allocated:  {human_bytes(peak_cuda)}  (activations + workspaces proxy)")
    else:
        print("  Peak allocated:       N/A (CPU)")
    print()
    print("What-if weight precision (for deployment planning):")
    for k in ("fp32","fp16","bf16","int8","int4"):
        wb = weight_bytes_by_prec[k]
        tb = total_bytes_by_prec[k]
        print(f"  {k.upper():<5} weights: {human_bytes(wb):>10}  => Total(~weights + iOS runtime est): {human_bytes(tb)}")
    print()
    if args.budget_mb is not None:
        budget_bytes = int(args.budget_mb * 1024 * 1024)
        ok_now = total_bytes_current_dtype <= budget_bytes
        print(f"=== Fit Check vs Budget ===")
        print(f"Budget:                 {args.budget_mb:.0f} MB")
        print(f"iOS headroom factor:    +{int(args.ios_headroom*100)}% applied to runtime")
        print(f"Total (current dtype):  {human_bytes(total_bytes_current_dtype)}  =>  {'FIT ✅' if ok_now else 'NO FIT ❌'}")
        # Also print a couple of useful alternates:
        for k in ("fp16","int8","int4"):
            ok = total_bytes_by_prec[k] <= budget_bytes
            print(f"  If {k.upper():<4}:        {human_bytes(total_bytes_by_prec[k])}  =>  {'FIT ✅' if ok else 'NO FIT ❌'}")
        print()
    print("Notes:")
    print("- Peak CUDA memory is used as a practical proxy for activation/workspace needs; we add a configurable headroom")
    print("  (default +50%) to be conservative for iOS/edge runtimes (Core ML / Metal / unified memory).")
    print("- For deployment, you’ll typically convert to Core ML and quantize weights (e.g., FP16 or INT8) to reduce memory.")
    print("- If compute prints 'unavailable', install one of: fvcore, ptflops, or thop.")
    print("- Re-run with your target batch/resolution; memory and FLOPs scale with both.")

if __name__ == "__main__":
    main()

