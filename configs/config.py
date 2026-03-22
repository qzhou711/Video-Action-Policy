"""Dataclass-based configs for mimic-video training."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    repo_id: str = "lerobot/libero_10"
    num_episodes: int = 379
    train_episodes: int = 340
    val_episodes: int = 39

    # Frame dimensions (per camera)
    camera_height: int = 256
    camera_width: int = 256
    camera_names: List[str] = field(
        default_factory=lambda: [
            "observation.images.image",
            "observation.images.wrist_image",
        ]
    )

    # Video frames: 17 pixel frames -> 5 latent frames (2 cond + 3 pred)
    num_pixel_frames: int = 17
    num_latent_frames: int = 5
    num_cond_latent_frames: int = 2
    num_pred_latent_frames: int = 3
    fps: int = 10

    # State/Action feature keys (will be concatenated)
    state_keys: List[str] = field(
        default_factory=lambda: [
            "observation.state",
        ]
    )
    action_keys: List[str] = field(
        default_factory=lambda: [
            "action",
        ]
    )

    # Actions
    action_chunk_size: int = 16
    action_dim: int = 7  # x, y, z, roll, pitch, yaw, gripper
    proprio_dim: int = 8  # x, y, z, rx, ry, rz, rw, gripper
    action_names: List[str] = field(
        default_factory=lambda: [
            "x", "y", "z", "roll", "pitch", "yaw", "gripper"
        ]
    )

    # Proprioception masking probability during training
    proprio_mask_prob: float = 0.1

    # Text prompt for the task
    task_prompt: str = "Complete the manipulation task."

    # Precomputed embeddings path
    precomputed_dir: str = "precomputed/"


@dataclass
class ModelConfig:
    # Cosmos model
    cosmos_model_id: str = "nvidia/Cosmos-Predict2-2B-Video2World"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    )

    # Hidden state extraction
    hidden_state_layer: int = 19  # Layer k=19
    hidden_state_pool: str = "none"  # "mean" (5 tokens) or "none" (all ~6000 tokens)

    # Action decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 8
    decoder_num_heads: int = 8
    decoder_mlp_ratio: int = 4
    backbone_hidden_dim: int = 2048  # Cosmos transformer hidden dim

    # VAE latent channels
    vae_latent_channels: int = 16


@dataclass
class Stage1Config:
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 10.0
    total_steps: int = 10000  # 1个epoch 400 steps, 25个epoch
    batch_size: int = 200  # effective batch size via accumulation
    micro_batch_size: int = 20
    gradient_accumulation_steps: int = 2  # auto-computed in train script: batch_size / (micro_batch * num_gpus)

    # LR schedule: constant after warmup
    lr_schedule: str = "constant"

    # Mixed precision
    dtype: str = "bf16"
    gradient_checkpointing: bool = True

    # Logging
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "checkpoints/stage1"
    wandb_project: str = "mimic-video"
    wandb_run_name: str = "stage1-lora"


@dataclass
class Stage2Config:
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 10.0
    total_steps: int = 10000  # 1个epoch 400 steps, 25个epoch
    batch_size: int = 200  # effective batch size via accumulation
    micro_batch_size: int = 20
    gradient_accumulation_steps: int = 2  # auto-computed in train script: batch_size / (micro_batch * num_gpus)

    # LR schedule: linear decay after warmup
    lr_schedule: str = "linear_decay"

    # Mixed precision
    dtype: str = "bf16"
    gradient_checkpointing: bool = True

    # Action flow matching tau sampling
    # pi0-style: U^(1/power) where U~Uniform(0,1)
    tau_power: float = 0.999

    # Logging
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "checkpoints/stage2"
    wandb_project: str = "mimic-video"
    wandb_run_name: str = "stage2-action-decoder"

    # Stage 1 checkpoint to load
    stage1_checkpoint: str = "checkpoints/stage1/final"
