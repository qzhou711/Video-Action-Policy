"""Entry point for Stage 2: Action decoder training with frozen backbone.

Supports single-GPU and multi-GPU (DDP) training.

Usage:
    # Single GPU
    python scripts/train_stage2.py

    # Multi-GPU (e.g. 5x 4090)
    torchrun --nproc_per_node=5 scripts/train_stage2.py

    # Resume from checkpoint
    torchrun --nproc_per_node=5 scripts/train_stage2.py --resume checkpoints/stage2/step_1000

    # 4090 5x libero_object
    torchrun --nproc_per_node=5 scripts/train_stage2.py \
    --suite libero_object \
    --cosmos_model_id ./checkpoints/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
    --stage1_checkpoint checkpoints/libero_object/stage1/step_4000 \
    --wandb_project "dit4dit-stage2" \
    --resume checkpoints/libero_object/stage2/step_1000

"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig, Stage2Config, get_suite_data_config, LIBERO_SUITES, GPU_PRESETS, apply_gpu_preset
from mimic_video.data.dataset import MimicVideoDataset
from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.action_decoder import ActionDecoderDiT
from mimic_video.training.stage2_trainer import Stage2Trainer


def setup_distributed():
    """Initialize distributed training if launched via torchrun."""
    if "RANK" in os.environ:
        from datetime import timedelta
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Timeout of 2 h lets rank 0 download the dataset and compute
        # action_stats on a fresh machine without other ranks timing out.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1  # single GPU fallback


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Action decoder training")
    parser.add_argument("--suite", type=str, default=None,
                        choices=list(LIBERO_SUITES.keys()),
                        help="LIBERO suite name (auto-sets repo_id, episodes, dirs)")
    parser.add_argument(
        "--stage1_checkpoint", type=str, default=None,
        help="Path to Stage 1 LoRA checkpoint (default: from config)"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to Stage 2 checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precomputed_dir", type=str, default=None)
    parser.add_argument("--cosmos_model_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # GPU preset + per-field overrides
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(GPU_PRESETS.keys()),
                        help="GPU preset (4090 / a100_40g / a100_80g / v100 / b200)")
    parser.add_argument("--micro_batch_size", type=int, default=None,
                        help="Override micro_batch_size from preset or config")
    parser.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16"],
                        help="Override training dtype (bf16 or fp16)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (faster, uses more VRAM)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (e.g. scale with batch size: lr=1e-4*(batch/200))")
    parser.add_argument("--allow_partial_action_chunk", action="store_true", default=True,
                        help="Include episode tail samples with shorter future actions; padded tail is masked in loss. (default: on)")
    parser.add_argument("--disable_partial_action_chunk", action="store_true",
                        help="Disable partial action chunk training and require full action_chunk_size targets.")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)
    device = f"cuda:{local_rank}"

    # Enable performance optimizations for Ada/Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Config: use suite if specified, otherwise defaults
    if args.suite:
        data_config = get_suite_data_config(args.suite)
    else:
        data_config = DataConfig()
    if args.precomputed_dir:
        data_config.precomputed_dir = args.precomputed_dir

    model_config = ModelConfig()
    train_config = Stage2Config()

    # Auto-set per-suite output dir, wandb name, and stage1 checkpoint
    if args.suite:
        train_config.output_dir = f"checkpoints/{args.suite}/stage2"
        train_config.wandb_run_name = f"stage2-{args.suite}"
        train_config.stage1_checkpoint = f"checkpoints/{args.suite}/stage1/final"

    if args.cosmos_model_id:
        model_config.cosmos_model_id = args.cosmos_model_id
    if args.wandb_project:
        train_config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        train_config.wandb_run_name = args.wandb_run_name

    # Apply GPU preset, then individual overrides (overrides win over preset)
    if args.preset:
        apply_gpu_preset(train_config, args.preset)
    if args.micro_batch_size is not None:
        train_config.micro_batch_size = args.micro_batch_size
    if args.dtype is not None:
        train_config.dtype = args.dtype
    if args.no_gradient_checkpointing:
        train_config.gradient_checkpointing = False
    if args.lr is not None:
        train_config.lr = args.lr

    # Auto-compute gradient accumulation for multi-GPU
    effective_batch = train_config.batch_size
    per_step_samples = train_config.micro_batch_size * world_size
    gradient_accumulation_steps = max(1, effective_batch // per_step_samples)

    if is_main:
        print(f"\n{'='*60}")
        print(f"  DDP: world_size={world_size}, rank={rank}, device={device}")
        print(f"  micro_batch={train_config.micro_batch_size} × {world_size} GPUs"
              f" × {gradient_accumulation_steps} accum = {per_step_samples * gradient_accumulation_steps} effective")
        print(f"{'='*60}\n")

    stage1_path = args.stage1_checkpoint or train_config.stage1_checkpoint

    # Load precomputed T5 embedding(s)
    # Multi-task: t5_embeddings.pt (dict) → dataset handles per-sample, trainer gets None
    # Single-task: t5_embedding.pt (tensor) → trainer broadcasts to all samples
    multi_t5_path = os.path.join(data_config.precomputed_dir, "t5_embeddings.pt")
    single_t5_path = os.path.join(data_config.precomputed_dir, "t5_embedding.pt")

    t5_embedding = None  # What we pass to the trainer
    if os.path.exists(multi_t5_path):
        if is_main:
            print(f"Found multi-task T5 embeddings at {multi_t5_path}")
            print("  → Dataset will provide per-sample T5 embeddings via batch.")
        # Don't load here — dataset will handle it
    elif os.path.exists(single_t5_path):
        if is_main:
            print(f"Loading single-task T5 embedding from {single_t5_path}")
        t5_embedding = torch.load(single_t5_path, map_location="cpu", weights_only=True)
    else:
        if is_main:
            print("WARNING: No precomputed T5 embedding found. Run precompute_embeddings.py first.")

    # Create dataset
    if is_main:
        print("Loading dataset...")
    train_episodes = list(range(data_config.train_episodes))

    train_dataset = MimicVideoDataset(
        repo_id=data_config.repo_id,
        camera_names=data_config.camera_names,
        state_keys=data_config.state_keys,
        action_keys=data_config.action_keys,
        num_pixel_frames=data_config.num_pixel_frames,
        action_chunk_size=data_config.action_chunk_size,
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        target_height=data_config.camera_height,
        target_width=data_config.camera_width,
        episode_indices=train_episodes,
        precomputed_dir=data_config.precomputed_dir,
        action_norm_type=data_config.action_norm_type,
        fps=data_config.fps,
        require_action_chunk=True,
        allow_partial_action_chunk=(args.allow_partial_action_chunk and not args.disable_partial_action_chunk),
    )

    # Compute or load action stats
    stats_path = os.path.join(data_config.precomputed_dir, "action_stats.pt")
    if is_main and not os.path.exists(stats_path):
        print("Computing action statistics...")
        action_stats = train_dataset.compute_action_stats()
        os.makedirs(data_config.precomputed_dir, exist_ok=True)
        torch.save(action_stats, stats_path)
        print(f"Saved action stats to {stats_path}")
    
    if world_size > 1:
        dist.barrier()
        
    if os.path.exists(stats_path):
        if is_main:
            print(f"Loading action stats from {stats_path}")
        action_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        train_dataset.action_mean = action_stats["mean"]
        train_dataset.action_std = action_stats["std"]
        train_dataset.action_min = action_stats["min"]
        train_dataset.action_max = action_stats["max"]

    if is_main:
        print(f"Train dataset: {len(train_dataset)} samples")

    # Sampler: DistributedSampler for DDP, None for single GPU
    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Load backbone with Stage 1 LoRA weights
    if is_main:
        print("Loading Cosmos video backbone...")
    backbone = CosmosVideoBackbone(
        model_id=model_config.cosmos_model_id,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_target_modules=model_config.lora_target_modules,
        hidden_state_layer=model_config.hidden_state_layer,
        dtype=torch.bfloat16,
        device=device,
    )

    # Load Stage 1 LoRA weights
    if os.path.exists(stage1_path):
        if is_main:
            print(f"Loading Stage 1 LoRA weights from {stage1_path}")
        backbone.load_lora(stage1_path)
    else:
        if is_main:
            print(f"WARNING: Stage 1 checkpoint not found at {stage1_path}. Using base model.")

    backbone.transformer.to(device)
    backbone.offload_vae_and_text_encoder("cpu")

    # Create action decoder
    if is_main:
        print("Creating action decoder...")
    perceiver_slots = (
        model_config.perceiver_slots_per_frame
        if model_config.hidden_state_pool == "perceiver"
        else 0
    )
    action_decoder = ActionDecoderDiT(
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        hidden_dim=model_config.decoder_hidden_dim,
        num_layers=model_config.decoder_num_layers,
        num_heads=model_config.decoder_num_heads,
        mlp_ratio=model_config.decoder_mlp_ratio,
        backbone_hidden_dim=backbone.hidden_dim,
        action_chunk_size=data_config.action_chunk_size,
        proprio_mask_prob=data_config.proprio_mask_prob,
        perceiver_slots=perceiver_slots,
        num_latent_frames=data_config.num_latent_frames,
    )

    action_decoder.to(device)

    # Wrap action decoder in DDP
    if world_size > 1:
        action_decoder = DDP(
            action_decoder,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        if is_main:
            print(f"Wrapped action decoder in DDP across {world_size} GPUs")
    else:
        if is_main:
            print("Compiling action decoder with torch.compile...")
        action_decoder = torch.compile(action_decoder)

    # Print parameter counts
    if is_main:
        decoder_module = action_decoder.module if world_size > 1 else action_decoder
        decoder_params = sum(p.numel() for p in decoder_module.parameters())
        print(f"Action decoder parameters: {decoder_params:,} (~{decoder_params / 1e6:.1f}M)")

    # Create trainer
    trainer = Stage2Trainer(
        backbone=backbone,
        action_decoder=action_decoder,
        train_dataloader=train_dataloader,
        lr=train_config.lr,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        grad_clip=train_config.grad_clip,
        total_steps=train_config.total_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_schedule=train_config.lr_schedule,
        dtype=train_config.dtype,
        output_dir=train_config.output_dir,
        log_every=train_config.log_every,
        save_every=train_config.save_every,
        wandb_project=train_config.wandb_project if is_main else None,
        wandb_run_name=train_config.wandb_run_name,
        precomputed_t5_embedding=t5_embedding,
        num_cond_latent_frames=data_config.num_cond_latent_frames,
        hidden_state_pool=model_config.hidden_state_pool,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    # Resume if requested
    start_step = 0
    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        start_step = trainer._load_checkpoint(args.resume)
        if is_main:
            print(f"Resumed at step {start_step}")

    # Train
    if is_main:
        print("Starting Stage 2 training...")
    trainer.train(start_step=start_step)
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
