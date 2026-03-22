"""Evaluation script for mimic-video.

Loads trained model (backbone + LoRA + action decoder) and evaluates:
- Action prediction MSE on held-out episodes
- Per-joint error analysis
- Optionally generates denoised video for visualization

Usage:
    python scripts/evaluate.py --stage1_checkpoint PATH --stage2_checkpoint PATH
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Use cached models, avoid network errors
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig, Stage2Config
from mimic_video.data.dataset import MimicVideoDataset
from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.action_decoder import ActionDecoderDiT
from mimic_video.inference.policy import MimicVideoPolicy


def evaluate_action_prediction(
    policy: MimicVideoPolicy,
    val_dataset: MimicVideoDataset,
    joint_names: list,
    device: str = "cuda",
    max_samples: int = 200,
):
    """Evaluate action prediction on validation episodes.

    Args:
        policy: Trained inference policy.
        val_dataset: Validation dataset.
        device: Device to run on.
        max_samples: Maximum number of samples to evaluate.

    Returns:
        Dict with evaluation metrics.
    """
    policy.eval()

    all_mse = []
    all_per_joint_mse = []
    num_samples = min(len(val_dataset), max_samples)

    print(f"Evaluating on {num_samples} samples...")
    for idx in tqdm(range(num_samples)):
        sample = val_dataset[idx]

        video = sample["video"].unsqueeze(0)  # [1, T, C, H, W]
        proprio = sample["proprio"].unsqueeze(0)  # [1, proprio_dim]
        gt_actions = sample["actions"].unsqueeze(0)  # [1, T_action, action_dim]

        # Denormalize ground truth for comparison
        gt_actions_denorm = val_dataset.denormalize_actions(gt_actions.to(device))

        # Predict actions
        pred_actions = policy.predict_action(video, proprio)  # [1, T_action, action_dim]

        # Compute MSE
        mse = ((pred_actions - gt_actions_denorm) ** 2).mean().item()
        per_joint_mse = ((pred_actions - gt_actions_denorm) ** 2).mean(dim=(0, 1)).cpu().numpy()

        all_mse.append(mse)
        all_per_joint_mse.append(per_joint_mse)

    all_mse = np.array(all_mse)
    all_per_joint_mse = np.stack(all_per_joint_mse, axis=0)

    # Use passed joint_names, pad with generic names if not enough
    if len(joint_names) < all_per_joint_mse.shape[1]:
        joint_names = joint_names + [f"action_{i}" for i in range(len(joint_names), all_per_joint_mse.shape[1])]

    metrics = {
        "overall_mse": float(all_mse.mean()),
        "overall_mse_std": float(all_mse.std()),
        "per_joint_mse": {
            name: float(all_per_joint_mse[:, i].mean())
            for i, name in enumerate(joint_names[:all_per_joint_mse.shape[1]])
        },
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate mimic-video")
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="Path to Stage 1 LoRA checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="Path to Stage 2 action decoder checkpoint")
    parser.add_argument("--precomputed_dir", type=str, default="precomputed/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tau_v", type=float, default=1.0, help="Video noise level at inference (1.0 = no denoising)")
    parser.add_argument("--num_action_steps", type=int, default=10, help="Number of Euler steps for action denoising")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to evaluate")
    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()

    # Load precomputed T5 embedding
    t5_path = os.path.join(args.precomputed_dir, "t5_embedding.pt")
    t5_embedding = None
    if os.path.exists(t5_path):
        t5_embedding = torch.load(t5_path, map_location="cpu", weights_only=True)

    # Load action stats
    stats_path = os.path.join(args.precomputed_dir, "action_stats.pt")
    action_stats = None
    if os.path.exists(stats_path):
        action_stats = torch.load(stats_path, map_location="cpu", weights_only=True)

    # Create validation dataset
    print("Loading validation dataset...")
    val_episodes = list(range(data_config.train_episodes, data_config.num_episodes))

    val_dataset = MimicVideoDataset(
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
        episode_indices=val_episodes,
        precomputed_dir=args.precomputed_dir,
        action_stats=action_stats,
        fps=data_config.fps,
    )
    print(f"Validation dataset: {len(val_dataset)} samples")

    # Load backbone
    print("Loading Cosmos video backbone with Stage 1 LoRA...")
    backbone = CosmosVideoBackbone(
        model_id=model_config.cosmos_model_id,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_target_modules=model_config.lora_target_modules,
        hidden_state_layer=model_config.hidden_state_layer,
        dtype=torch.bfloat16,
        device=args.device,
    )
    backbone.load_lora(args.stage1_checkpoint)
    backbone.transformer.to(args.device)
    backbone.offload_vae_and_text_encoder("cpu")

    # Load action decoder
    print("Loading action decoder...")
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
    )

    decoder_path = os.path.join(args.stage2_checkpoint, "action_decoder.pt")
    state_dict = torch.load(decoder_path, map_location=args.device, weights_only=True)
    # Strip 'module.' prefix from DDP-saved checkpoints
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    action_decoder.load_state_dict(state_dict)
    action_decoder.to(args.device)
    action_decoder.eval()

    # Create policy
    print("Creating inference policy...")
    policy = MimicVideoPolicy(
        backbone=backbone,
        action_decoder=action_decoder,
        action_stats=action_stats,
        t5_embedding=t5_embedding,
        tau_v=args.tau_v,
        num_action_denoise_steps=args.num_action_steps,
        num_cond_latent_frames=data_config.num_cond_latent_frames,
        num_pred_latent_frames=data_config.num_pred_latent_frames,
        num_pixel_frames=data_config.num_pixel_frames,
        camera_names=data_config.camera_names,
        target_height=data_config.camera_height,
        target_width=data_config.camera_width,
        device=args.device,
    )

    # Action names for logging
    action_names = getattr(data_config, "action_names", [f"action_{i}" for i in range(data_config.action_dim)])

    # Evaluate
    metrics = evaluate_action_prediction(
        policy=policy,
        val_dataset=val_dataset,
        joint_names=action_names,
        device=args.device,
        max_samples=args.max_samples,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Action MSE: {metrics['overall_mse']:.6f} (+/- {metrics['overall_mse_std']:.6f})")
    print(f"\nPer-Joint MSE:")
    for joint_name, mse in metrics['per_joint_mse'].items():
        print(f"  {joint_name:20s}: {mse:.6f}")
    print("=" * 60)

    # Save metrics
    output_path = os.path.join(args.precomputed_dir, "eval_metrics.pt")
    torch.save(metrics, output_path)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
