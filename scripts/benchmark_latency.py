"""
Latency Benchmark for mimic-video Inference Pipeline
=====================================================

Measures wall-clock time (and optional CUDA event time) for each stage:
  1. Video DiT forward → hidden states h_video  (raw [B, T*H'*W', D])
  2. h_video pooling                             (spatial mean pool → [B, T, D])
  3. Action DiT denoising (N Euler steps)        (→ action chunk)

Usage:
    # Quick run with synthetic inputs (no model weights needed)
    python scripts/benchmark_latency.py --dry_run

    # Full run with real model weights
    python scripts/benchmark_latency.py --device cuda --warmup 3 --repeats 10

    # Compare pool modes (run separately to see the difference)
    python scripts/benchmark_latency.py --pool_mode mean --repeats 10
    python scripts/benchmark_latency.py --pool_mode none  --repeats 10

Arguments:
    --device          cuda / cpu                      [default: cuda]
    --dtype           bf16 / fp32                     [default: bf16]
    --batch_size      batch size B                    [default: 1]
    --warmup          warmup iterations (skipped)     [default: 3]
    --repeats         timed iterations                [default: 10]
    --action_steps    Euler steps for action ODE      [default: 10]
    --pool_mode       "mean" or "none"                [default: mean]
    --dry_run         use random tensors, skip model load
    --cosmos_model_id  HuggingFace model id for Cosmos [default: nvidia/Cosmos-Predict2-2B-Video2World]
    --stage1_checkpoint  path to Stage-1 LoRA checkpoint (optional)
"""

import argparse
import sys
import os
import time
import statistics

# Ensure project root (parent of scripts/) is on sys.path so that
# `configs` and `mimic_video` packages can be found regardless of CWD.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Context-manager timer that records elapsed seconds."""

    def __enter__(self):
        cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        cuda_sync()
        self.elapsed = time.perf_counter() - self._start


def stats(times: list) -> dict:
    return {
        "mean_ms":   statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "min_ms":    min(times) * 1000,
        "max_ms":    max(times) * 1000,
        "std_ms":    (statistics.stdev(times) * 1000) if len(times) > 1 else 0.0,
    }


def print_stats(label: str, times: list):
    s = stats(times)
    print(
        f"  {label:<40s}  "
        f"mean={s['mean_ms']:7.2f} ms  "
        f"median={s['median_ms']:7.2f} ms  "
        f"min={s['min_ms']:7.2f} ms  "
        f"max={s['max_ms']:7.2f} ms  "
        f"std={s['std_ms']:6.2f} ms"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run benchmark (no model loading — synthetic random tensors)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_dry_run(args):
    """Use random tensors that mimic real tensor shapes; no model weights."""
    print("\n" + "=" * 70)
    print("DRY-RUN MODE — using random tensors (no model loaded)")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    B               = args.batch_size
    C_lat           = 16          # Cosmos VAE latent channels
    T_lat           = 5           # latent frames (2 cond + 3 pred)
    H_lat, W_lat    = 16, 32      # spatial latent dims for 256×256 with 2-cam concat
    backbone_hidden = 2048        # Cosmos 2B hidden dim
    patch_h, patch_w = 1, 1      # effective patch after patchify (already factored into H_lat*W_lat)
    action_dim      = 7
    action_chunk    = 16
    proprio_dim     = 8
    decoder_hidden  = 512
    decoder_layers  = 8
    decoder_heads   = 8

    # Simulated raw hidden states [B, T*H'*W', backbone_hidden]
    THW  = T_lat * H_lat * W_lat  # ~12 000 for 5×30×80
    h_raw = torch.randn(B, THW, backbone_hidden, device=device, dtype=dtype)

    print(f"\n  Tensor shapes used:")
    print(f"    h_video (raw) :  {list(h_raw.shape)}  ({THW} tokens)")
    print(f"    Action chunk  :  [{B}, {action_chunk}, {action_dim}]")

    # ── 1. Simulate Video DiT forward (just a Linear + LayerNorm as proxy)
    proxy_linear = torch.nn.Linear(backbone_hidden, backbone_hidden).to(device=device, dtype=dtype)

    times_vdit = []
    for i in range(args.warmup + args.repeats):
        x_in = torch.randn_like(h_raw)
        with Timer() as t:
            _ = proxy_linear(x_in)
        if i >= args.warmup:
            times_vdit.append(t.elapsed)

    # ── 2. h_video pooling: spatial mean pool → [B, T_lat, D]
    times_pool_mean = []
    times_pool_none = []

    for i in range(args.warmup + args.repeats):
        h = torch.randn_like(h_raw)
        # mean pool
        with Timer() as t:
            h_bthw = h.view(B, T_lat, H_lat * W_lat, backbone_hidden)
            _ = h_bthw.mean(dim=2)  # [B, T, D]
        if i >= args.warmup:
            times_pool_mean.append(t.elapsed)

        # none (identity — just a view)
        with Timer() as t:
            _ = h.clone()  # simulate copy / no-op pass-through
        if i >= args.warmup:
            times_pool_none.append(t.elapsed)

    # ── 3. Action DiT forward (proxy: single MHA + linear)
    import torch.nn as nn
    proxy_attn  = nn.MultiheadAttention(decoder_hidden, decoder_heads, batch_first=True).to(device=device, dtype=dtype)
    proxy_proj  = nn.Linear(decoder_hidden, action_dim).to(device=device, dtype=dtype)

    times_action = []
    for i in range(args.warmup + args.repeats):
        a = torch.randn(B, action_chunk, decoder_hidden, device=device, dtype=dtype)
        with Timer() as t:
            for _ in range(args.action_steps):
                a, _ = proxy_attn(a, a, a)
            _ = proxy_proj(a)
        if i >= args.warmup:
            times_action.append(t.elapsed)

    # ── Results
    print(f"\n  Results (warmup={args.warmup}, repeats={args.repeats}):\n")
    print_stats("Video-DiT forward (proxy linear)",   times_vdit)
    print_stats("h_video pool → mean [B,T,D]",        times_pool_mean)
    print_stats("h_video pool → none (identity)",      times_pool_none)
    print_stats(f"Action-DiT ({args.action_steps} Euler steps, proxy)", times_action)


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark with real model weights
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_real(args):
    print("\n" + "=" * 70)
    print("REAL MODEL MODE — loading Cosmos + ActionDecoderDiT")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # ── Load backbone ──────────────────────────────────────────────────────
    print("\n[1/2] Loading CosmosVideoBackbone …")
    from configs.config import DataConfig, ModelConfig
    from mimic_video.models.video_backbone import CosmosVideoBackbone

    dcfg = DataConfig()
    mcfg = ModelConfig()

    backbone = CosmosVideoBackbone(
        model_id            = args.cosmos_model_id,
        lora_rank           = mcfg.lora_rank,
        lora_alpha          = mcfg.lora_alpha,
        lora_target_modules = mcfg.lora_target_modules,
        hidden_state_layer  = mcfg.hidden_state_layer,
        dtype               = dtype,
        device              = str(device),
    )

    if args.stage1_checkpoint:
        print(f"    → Loading LoRA from {args.stage1_checkpoint}")
        backbone.load_lora(args.stage1_checkpoint)

    backbone.to(device)
    backbone.eval()
    backbone.freeze_for_stage2()  # freeze for inference; keeps hooks active

    # Compute a single T5 embedding with the correct hidden size.
    # (Don't hardcode 4096; Cosmos' cross-attn expects a specific dimension.)
    B = args.batch_size
    with torch.no_grad():
        t5_emb = backbone.encode_text("a person doing an action")
        t5_emb = t5_emb.to(device=device, dtype=dtype)
        if B > 1 and t5_emb.shape[0] == 1:
            t5_emb = t5_emb.expand(B, -1, -1)

    # Offload VAE/text-encoder now (not needed for this latency test).
    # Keep the precomputed `t5_emb` on the target device.
    backbone.offload_vae_and_text_encoder("cpu")

    # ── Load action decoder ───────────────────────────────────────────────
    print("[2/2] Building ActionDecoderDiT …")
    from mimic_video.models.action_decoder import ActionDecoderDiT
    from mimic_video.models.flow_matching  import FlowMatchingScheduler

    action_decoder = ActionDecoderDiT(
        action_dim         = dcfg.action_dim,
        proprio_dim        = dcfg.proprio_dim,
        hidden_dim         = mcfg.decoder_hidden_dim,
        num_layers         = mcfg.decoder_num_layers,
        num_heads          = mcfg.decoder_num_heads,
        mlp_ratio          = mcfg.decoder_mlp_ratio,
        backbone_hidden_dim= mcfg.backbone_hidden_dim,
        action_chunk_size  = dcfg.action_chunk_size,
        proprio_mask_prob  = 0.0,       # deterministic at inference
    ).to(device=device, dtype=dtype)
    action_decoder.eval()

    fm = FlowMatchingScheduler()

    # ── Synthetic inputs ──────────────────────────────────────────────────
    C_lat   = backbone.vae.config.z_dim if hasattr(backbone.vae, 'config') else 16
    T_lat   = dcfg.num_latent_frames   # 5
    T_cond  = dcfg.num_cond_latent_frames   # 2
    T_pred  = dcfg.num_pred_latent_frames   # 3

    # We need H_lat, W_lat: approximate from VAE scale factors
    vae_sf_sp   = getattr(backbone, 'vae_scale_factor_spatial',  8)
    vae_sf_temp = getattr(backbone, 'vae_scale_factor_temporal', 4)
    H_lat = dcfg.camera_height // vae_sf_sp // 2  # /2 for 2×2 concat; approx
    W_lat = dcfg.camera_width  // vae_sf_sp // 2

    print(f"\n  Inferred latent spatial size: {H_lat}×{W_lat}  (from VAE scale_factor={vae_sf_sp})")

    z_cond  = torch.randn(B, C_lat, T_cond, H_lat, W_lat, device=device, dtype=dtype)
    z_noisy = torch.randn(B, C_lat, T_pred, H_lat, W_lat, device=device, dtype=dtype)

    tau_v   = torch.ones(B, device=device, dtype=dtype)   # pure noise

    # Proprio + action noise
    proprio = torch.randn(B, dcfg.proprio_dim, device=device, dtype=dtype)
    a_noise = torch.randn(B, dcfg.action_chunk_size, dcfg.action_dim, device=device, dtype=dtype)

    pool_mode = args.pool_mode

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"\n  Warming up ({args.warmup} iterations) …")
    with torch.no_grad():
        for _ in range(args.warmup):
            # video dit
            backbone.clear_hidden_states_cache()
            backbone.forward_transformer(
                z_noisy=z_noisy, z_cond=z_cond,
                tau_v=tau_v, encoder_hidden_states=t5_emb,
            )
            h_raw = backbone.get_captured_hidden_states()
            # pool
            backbone.pool_hidden_states(h_raw.float(), T_lat, mode=pool_mode)

    # ── BENCHMARK 1: Video DiT forward ──────────────────────────────────
    print(f"\n  Benchmarking ({args.repeats} repeats) …")
    times_vdit = []
    with torch.no_grad():
        for _ in range(args.repeats):
            backbone.clear_hidden_states_cache()
            with Timer() as t:
                backbone.forward_transformer(
                    z_noisy=z_noisy, z_cond=z_cond,
                    tau_v=tau_v, encoder_hidden_states=t5_emb,
                )
            times_vdit.append(t.elapsed)
    # Capture hidden states for pool benchmark
    h_raw_cached = backbone.get_captured_hidden_states().float().detach()

    # ── BENCHMARK 2: h_video pooling ────────────────────────────────────
    times_pool_mean = []
    times_pool_none = []

    with torch.no_grad():
        for _ in range(args.repeats):
            h = h_raw_cached.clone()
            with Timer() as t:
                _ = backbone.pool_hidden_states(h, T_lat, mode="mean")
            times_pool_mean.append(t.elapsed)

        for _ in range(args.repeats):
            h = h_raw_cached.clone()
            with Timer() as t:
                _ = backbone.pool_hidden_states(h, T_lat, mode="none")
            times_pool_none.append(t.elapsed)

    # ── BENCHMARK 3: Action DiT (full ODE denoising) ────────────────────
    # Get pooled h_video for action decoder input
    h_pooled_mean = backbone.pool_hidden_states(h_raw_cached.clone(), T_lat, mode="mean")
    h_pooled_none = backbone.pool_hidden_states(h_raw_cached.clone(), T_lat, mode="none")

    h_for_action = h_pooled_mean if pool_mode == "mean" else h_pooled_none
    h_for_action = h_for_action.to(device=device, dtype=dtype)

    # Single action decoder forward
    times_action_fwd = []
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = action_decoder(
                noisy_actions=a_noise, proprio=proprio,
                h_video=h_for_action, tau_a=tau_v, tau_v=tau_v, training=False,
            )
        for _ in range(args.repeats):
            with Timer() as t:
                _ = action_decoder(
                    noisy_actions=a_noise, proprio=proprio,
                    h_video=h_for_action, tau_a=tau_v, tau_v=tau_v, training=False,
                )
            times_action_fwd.append(t.elapsed)

    # Full ODE denoising (N Euler steps)
    times_action_ode = []
    dt = -1.0 / args.action_steps

    with torch.no_grad():
        for _ in range(args.repeats):
            a_t = a_noise.clone()
            tau  = 1.0
            tau_tensor = torch.ones(B, device=device, dtype=dtype)
            with Timer() as t:
                for _ in range(args.action_steps):
                    tau_tensor.fill_(tau)
                    v = action_decoder(
                        noisy_actions=a_t, proprio=proprio,
                        h_video=h_for_action, tau_a=tau_tensor, tau_v=tau_v, training=False,
                    )
                    a_t = a_t + v * dt
                    tau += dt
            times_action_ode.append(t.elapsed)

    # ── Print Results ─────────────────────────────────────────────────────
    THW = h_raw_cached.shape[1]
    print(f"\n  Config: B={B}, latent={T_lat}×{H_lat}×{W_lat}, h_tokens={THW}, "
          f"pool_mode={pool_mode}, action_steps={args.action_steps}")
    print(f"  dtype={args.dtype}, device={device}\n")

    print(f"  {'Stage':<45s}  {'mean':>8s}  {'median':>8s}  {'min':>8s}  {'max':>8s}  {'std':>7s}")
    print("  " + "-" * 90)

    def row(label, times):
        s = stats(times)
        print(f"  {label:<45s}  "
              f"{s['mean_ms']:>7.2f}ms  "
              f"{s['median_ms']:>7.2f}ms  "
              f"{s['min_ms']:>7.2f}ms  "
              f"{s['max_ms']:>7.2f}ms  "
              f"{s['std_ms']:>6.2f}ms")

    row("1. Video-DiT forward (h_video raw)",        times_vdit)
    # row("2. h_video pool → mean [B,T,D]",            times_pool_mean)
    # row("2. h_video pool → none (passthrough)",       times_pool_none)
    row("2. Action-DiT single forward",               times_action_fwd)
    row(f"3. Action-DiT ODE ({args.action_steps} Euler steps)", times_action_ode)

    total_mean = (
        statistics.mean(times_vdit)
        + (statistics.mean(times_pool_mean) if pool_mode == "mean" else statistics.mean(times_pool_none))
        + statistics.mean(times_action_ode)
    )
    print("  " + "-" * 90)
    print(f"  {'  → TOTAL (DiT + pool + ODE)':<45s}  {total_mean*1000:>7.2f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Latency benchmark for mimic-video pipeline")
    p.add_argument("--device",           default="cuda",  help="cuda / cpu")
    p.add_argument("--dtype",            default="bf16",  choices=["bf16", "fp32"])
    p.add_argument("--batch_size",       type=int, default=1)
    p.add_argument("--warmup",           type=int, default=3,  help="warmup iterations (not timed)")
    p.add_argument("--repeats",          type=int, default=10, help="timed iterations")
    p.add_argument("--action_steps",     type=int, default=10, help="Euler ODE steps for action")
    p.add_argument("--pool_mode",        default="mean",  choices=["mean", "none"],
                   help="h_video pooling mode for action decoder input")
    p.add_argument("--dry_run",          action="store_true",
                   help="Use random proxy tensors, skip model loading")
    p.add_argument("--cosmos_model_id",  default="nvidia/Cosmos-Predict2-2B-Video2World")
    p.add_argument("--stage1_checkpoint", default=None,
                   help="Optional path to Stage-1 LoRA checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\nmimic-video Latency Benchmark")
    print(f"  device={args.device}  dtype={args.dtype}  batch_size={args.batch_size}")
    print(f"  warmup={args.warmup}  repeats={args.repeats}  action_ode_steps={args.action_steps}")
    print(f"  pool_mode={args.pool_mode}  dry_run={args.dry_run}")

    if args.dry_run:
        benchmark_dry_run(args)
    else:
        benchmark_real(args)
