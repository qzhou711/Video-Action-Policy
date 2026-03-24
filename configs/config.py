"""Dataclass-based configs for mimic-video training."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# LIBERO Suite Registry
# ============================================================
LIBERO_SUITES: Dict[str, Dict] = {
    "libero_spatial": {
        "repo_id": "lerobot/libero_spatial_image",
        "num_episodes": 432,
        "train_episodes": 432,
        "val_episodes": 0,
    },
    "libero_object": {
        "repo_id": "lerobot/libero_object_image",
        "num_episodes": 454,
        "train_episodes": 454,
        "val_episodes": 0,
    },
    "libero_goal": {
        "repo_id": "lerobot/libero_goal_image",
        "num_episodes": 428,
        "train_episodes": 428,
        "val_episodes": 0,
    },
    "libero_10": {
        "repo_id": "lerobot/libero_10_image",
        "num_episodes": 379,
        "train_episodes": 379,
        "val_episodes": 0,
    },
}


@dataclass
class DataConfig:
    # Dataset (overridden by get_suite_data_config)
    repo_id: str = ""
    num_episodes: int = 454
    train_episodes: int = 454  # 100% for training
    val_episodes: int = 0

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
    action_norm_type: str = "min-max"  # "min-max" (LIBERO, mimic) or "mean-std" (BridgeDataV2)
    action_chunk_size: int = 16
    action_dim: int = 7  # x, y, z, roll, pitch, yaw, gripper
    proprio_dim: int = 8  # x, y, z, rx, ry, rz, rw, gripper
    action_names: List[str] = field(
        default_factory=lambda: [
            "x", "y", "z", "roll", "pitch", "yaw", "gripper"
        ]
    )

    # Proprioception masking probability during training
    proprio_mask_prob: float = 0.5

    # Text prompts for tasks (auto-populated from dataset metadata if None)
    task_prompts: Optional[List[str]] = None

    # Precomputed embeddings path
    precomputed_dir: str = "precomputed/"


def get_suite_data_config(suite_name: str) -> DataConfig:
    """Create a DataConfig with suite-specific settings.

    Args:
        suite_name: One of 'libero_spatial', 'libero_object', 'libero_goal', 'libero_10'.

    Returns:
        DataConfig with repo_id, episodes, and precomputed_dir set for the suite.
    """
    if suite_name not in LIBERO_SUITES:
        available = ", ".join(LIBERO_SUITES.keys())
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {available}")

    suite = LIBERO_SUITES[suite_name]
    return DataConfig(
        repo_id=suite["repo_id"],
        num_episodes=suite["num_episodes"],
        train_episodes=suite["train_episodes"],
        val_episodes=suite["val_episodes"],
        precomputed_dir=f"precomputed/{suite_name}/",
    )


@dataclass
class ModelConfig:
    # Cosmos backbone (HF hub id; override with local snapshot path if needed)
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
    hidden_state_pool: str = "mean"  # "mean" (global mean pooling, paper default) or "none" (all ~6000 tokens)

    # Action decoder (paper: 12-layer transformer, 16 heads, hidden dim 1024)
    decoder_hidden_dim: int = 1024
    decoder_num_layers: int = 12
    decoder_num_heads: int = 16
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
    log_video_every: int = 500
    save_every: int = 1000
    output_dir: str = "stage1"
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


    # Logging
    log_every: int = 10
    log_video_every: int = 10  # Temp low interval for OOM stress testing
    save_every: int = 1000
    output_dir: str = "stage2"
    wandb_project: str = "mimic-video"
    wandb_run_name: str = "stage2-action-decoder"

    # Stage 1 checkpoint to load
    stage1_checkpoint: str = ""
