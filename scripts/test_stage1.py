"""Simple Stage-1 test script (single sample or all tasks).

Usage:
    # Single sample
    python scripts/test_stage1.py \
      --suite libero_object \
      --checkpoint checkpoints/libero_object/stage1/step_8000 \
      --cosmos_model_id checkpoints/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/<id> \
      --device cuda --ode_steps 20 --sample_idx 0 --output_path stage1_test.gif

    # All 10 tasks in libero_object, one sample per task
    python scripts/test_stage1.py \
      --suite libero_object \
      --checkpoint checkpoints/libero_object/stage1/step_8000 \
      --cosmos_model_id checkpoints/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/<id> \
      --device cuda --all_tasks --samples_per_task 1 --output_dir stage1_all_tasks
      
python scripts/test_stage1.py \
  --suite libero_object \
  --checkpoint checkpoints/libero_object/stage1/step_5000 \
  --cosmos_model_id checkpoints/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
  --device cuda \
  --ode_steps 20 \
  --full_episode \
  --all_tasks \
  --samples_per_task 3 \
  --frame_pick last \
  --save_format mp4 \
  --output_dir stage1_full_all_tasks_3eps_5000_after

"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
import argparse
import contextlib
import json
import re

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import ModelConfig, get_suite_data_config, LIBERO_SUITES
from mimic_video.data.dataset import MimicVideoDataset
from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.flow_matching import FlowMatchingScheduler


def sanitize_filename(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:80] or "task"


def load_task_descriptions(precomputed_dir: str):
    desc_path = os.path.join(precomputed_dir, "t5_task_descriptions.json")
    if not os.path.exists(desc_path):
        return {}
    with open(desc_path, "r") as f:
        data = json.load(f)
    return {int(k): str(v) for k, v in data.items()}


def to_video_np(x: torch.Tensor):
    x = (x.squeeze(0).permute(1, 2, 3, 0).clamp(-1, 1) * 0.5 + 0.5) * 255
    return x.cpu().to(torch.uint8).numpy()


def save_side_by_side(gt_full: torch.Tensor, pred_full: torch.Tensor, save_path: str):
    gt_np = to_video_np(gt_full)
    pred_np = to_video_np(pred_full)
    side_by_side = np.concatenate([gt_np, pred_np], axis=2)  # [T, H, 2W, C]
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".mp4":
        imageio.mimsave(save_path, list(side_by_side), fps=10)
        return
    pil_frames = [Image.fromarray(f) for f in side_by_side]
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # 10 FPS
        loop=0,
    )


def save_frame_sequence(frames, save_path: str):
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".mp4":
        imageio.mimsave(save_path, frames, fps=10)
        return
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0,
    )


def predict_one_sample(
    backbone,
    fm,
    batch,
    num_cond_latent_frames: int,
    compute_dtype,
    device: str,
    ode_steps: int,
):
    video = batch["video"].unsqueeze(0)  # [1, T, C, H, W]
    video = video.permute(0, 2, 1, 3, 4).to(device)  # [1, C, T, H, W]

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=compute_dtype)
        if device.startswith("cuda")
        else contextlib.nullcontext()
    )

    with torch.inference_mode():
        backbone.move_vae_to(device)
        z_0 = backbone.encode_video(video)  # [1, C_lat, T_lat, H_lat, W_lat]
        z_cond = z_0[:, :, :num_cond_latent_frames]
        z_pred_gt = z_0[:, :, num_cond_latent_frames:]

        if "t5_embedding" not in batch:
            raise ValueError("Batch missing t5_embedding; run precompute_embeddings.py first.")
        t5_emb = batch["t5_embedding"].unsqueeze(0).to(device, dtype=compute_dtype)
        if t5_emb.ndim == 4 and t5_emb.shape[1] == 1:
            t5_emb = t5_emb.squeeze(1)

        z_noise = torch.randn_like(z_pred_gt)

        def model_fn(z_t, tau):
            tau_tensor = torch.tensor([tau], device=z_t.device, dtype=z_t.dtype)
            with autocast_ctx:
                _, full_out = backbone.forward_transformer(
                    z_noisy=z_t,
                    z_cond=z_cond,
                    tau_v=tau_tensor,
                    encoder_hidden_states=t5_emb,
                )
            x0_pred = full_out[:, :, num_cond_latent_frames:]
            return (z_t - x0_pred) / max(tau, 1e-6)

        z_pred_denoised = fm.ode_solve_euler(
            model_fn, z_noise, num_steps=ode_steps, tau_start=1.0, tau_end=0.0
        )

        backbone.move_vae_to(device)
        gt_full = backbone.decode_video(z_0)
        pred_latents = torch.cat([z_cond, z_pred_denoised], dim=2)
        pred_full = backbone.decode_video(pred_latents)

        future_mse = torch.mean((z_pred_denoised.float() - z_pred_gt.float()) ** 2).item()

    return gt_full, pred_full, future_mse


def pick_frame_idx(gt_np: np.ndarray, frame_pick: str, num_cond_latent_frames: int, pred_frame_offset: int) -> int:
    if frame_pick == "last":
        return gt_np.shape[0] - 1
    if frame_pick == "mid":
        return gt_np.shape[0] // 2
    idx = num_cond_latent_frames + pred_frame_offset
    return min(max(idx, 0), gt_np.shape[0] - 1)


def main():
    parser = argparse.ArgumentParser(description="Test Stage 1: future image prediction")
    parser.add_argument("--suite", type=str, required=True, choices=list(LIBERO_SUITES.keys()))
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 1 LoRA checkpoint")
    parser.add_argument("--cosmos_model_id", type=str, required=True, help="Path to Cosmos model")
    parser.add_argument("--output_path", type=str, default="stage1_test.mp4", help="Output path (single-sample mode)")
    parser.add_argument("--output_dir", type=str, default="stage1_test_outputs", help="Output dir (all-task mode)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ode_steps", type=int, default=20, help="Euler ODE steps for denoising")
    parser.add_argument("--sample_idx", type=int, default=0, help="Dataset sample index in single-sample mode")
    parser.add_argument("--all_tasks", action="store_true", help="Run all tasks in the selected suite")
    parser.add_argument("--samples_per_task", type=int, default=1, help="-1 means all samples per task")
    parser.add_argument("--save_format", type=str, default="mp4", choices=["mp4", "gif"], help="Output format")
    parser.add_argument("--full_episode", action="store_true", help="Run one complete episode (rolling prediction)")
    parser.add_argument("--task_id", type=int, default=0, help="Task id for --full_episode mode")
    parser.add_argument("--episode_rank_in_task", type=int, default=0,
                        help="Which episode in selected task for --full_episode mode (0-based)")
    parser.add_argument("--pred_frame_offset", type=int, default=0,
                        help="Future-frame offset in latent timeline for --full_episode mode")
    parser.add_argument("--frame_pick", type=str, default="last", choices=["early", "mid", "last"],
                        help="Which frame to pick from each window in --full_episode mode")
    args = parser.parse_args()

    device = args.device
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    print(f"Using device: {device}")

    data_config = get_suite_data_config(args.suite)
    model_config = ModelConfig()
    model_config.cosmos_model_id = args.cosmos_model_id

    print(f"Loading dataset for {args.suite}...")
    test_dataset = MimicVideoDataset(
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
        episode_indices=None,  # full suite
        precomputed_dir=data_config.precomputed_dir,
        action_norm_type=data_config.action_norm_type,
        fps=data_config.fps,
        require_action_chunk=False,
    )
    print(f"Dataset size: {len(test_dataset)}")

    print("Initializing Cosmos video backbone...")
    backbone = CosmosVideoBackbone(
        model_id=model_config.cosmos_model_id,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_target_modules=model_config.lora_target_modules,
        hidden_state_layer=model_config.hidden_state_layer,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device=device,
    )
    print(f"Loading LoRA from {args.checkpoint}...")
    backbone.load_lora(args.checkpoint, is_trainable=False)
    backbone.transformer.to(device)
    backbone.transformer.eval()
    backbone.move_vae_to(device)
    backbone.offload_vae_and_text_encoder("cpu")

    fm = FlowMatchingScheduler()
    num_cond_latent_frames = data_config.num_cond_latent_frames
    num_pred_latent_frames = data_config.num_pred_latent_frames
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def run_one_full_episode(task_id: int, episode_meta: dict, out_path: str):
        ep_idx = episode_meta["episode_index"]
        ep_start = episode_meta["dataset_from_index"]
        ep_end = episode_meta["dataset_to_index"]
        # Stage-1 full-episode visualization should only require a full video window.
        min_frames_needed = test_dataset.num_pixel_frames
        last_start = ep_end - min_frames_needed
        if last_start < ep_start:
            raise ValueError(
                f"Episode too short for Stage-1 windowed prediction. "
                f"task_id={task_id}, episode_index={ep_idx}"
            )

        window_global_indices = list(range(ep_start, last_start + 1))
        local_index_map = {g: i for i, g in enumerate(test_dataset.valid_indices)}
        window_local_indices = [local_index_map[g] for g in window_global_indices if g in local_index_map]
        if not window_local_indices:
            raise ValueError(
                f"No valid local indices for task_id={task_id}, episode_index={ep_idx}"
            )

        compare_frames = []
        mse_list = []
        print(
            f"  full episode | task_id={task_id} episode_index={ep_idx} "
            f"windows={len(window_local_indices)}"
        )
        for step_i, sample_idx in enumerate(window_local_indices):
            batch = test_dataset[sample_idx]
            gt_full, pred_full, future_mse = predict_one_sample(
                backbone, fm, batch, num_cond_latent_frames, compute_dtype, device, args.ode_steps
            )
            mse_list.append(future_mse)

            gt_np = to_video_np(gt_full)
            pred_np = to_video_np(pred_full)
            frame_idx = pick_frame_idx(
                gt_np, args.frame_pick, num_cond_latent_frames, args.pred_frame_offset
            )
            gt_f = gt_np[frame_idx]
            pred_f = pred_np[frame_idx]
            compare_frames.append(np.concatenate([gt_f, pred_f], axis=1))

            if (step_i + 1) % 50 == 0 or step_i == 0 or (step_i + 1) == len(window_local_indices):
                print(f"    progress: {step_i + 1}/{len(window_local_indices)} windows")

        save_frame_sequence(compare_frames, out_path)
        return float(np.mean(mse_list)), float(np.std(mse_list)), len(window_local_indices)

    if args.full_episode:
        if args.pred_frame_offset < 0 or args.pred_frame_offset >= num_pred_latent_frames:
            raise ValueError(
                f"--pred_frame_offset must be in [0, {num_pred_latent_frames - 1}]"
            )

        episodes = test_dataset.lerobot_dataset.meta.episodes
        ep_to_task = test_dataset.episode_to_task
        task_desc = load_task_descriptions(data_config.precomputed_dir)
        if args.all_tasks:
            os.makedirs(args.output_dir, exist_ok=True)
            task_episode_map = {}
            for ep in episodes:
                t_idx = ep_to_task.get(ep["episode_index"], -1)
                if t_idx >= 0:
                    task_episode_map.setdefault(t_idx, []).append(ep)

            all_means = []
            for task_idx in sorted(task_episode_map.keys()):
                task_name = task_desc.get(task_idx, f"task_{task_idx}")
                task_slug = sanitize_filename(task_name)
                task_dir = os.path.join(args.output_dir, f"task_{task_idx:02d}_{task_slug}")
                os.makedirs(task_dir, exist_ok=True)
                task_eps = task_episode_map[task_idx]
                chosen_eps = task_eps if args.samples_per_task < 0 else task_eps[:args.samples_per_task]
                print(f"\nTask {task_idx}: {task_name} | episodes={len(chosen_eps)}/{len(task_eps)}")
                for ep_rank, ep_meta in enumerate(chosen_eps):
                    out_path = os.path.join(
                        task_dir,
                        f"episode_{ep_rank:03d}_epidx_{ep_meta['episode_index']}.{args.save_format}",
                    )
                    mse_mean, mse_std, nwin = run_one_full_episode(task_idx, ep_meta, out_path)
                    all_means.append(mse_mean)
                    print(
                        f"    saved: {out_path} | windows={nwin} "
                        f"| mse_mean={mse_mean:.6f} mse_std={mse_std:.6f}"
                    )
            backbone.offload_vae_and_text_encoder("cpu")
            if all_means:
                print(f"\nAll full-episode runs done. Global mean MSE: {float(np.mean(all_means)):.6f}")
            print(f"Outputs saved to: {args.output_dir}")
            return

        if args.task_id < 0:
            raise ValueError("--task_id must be >= 0")
        if args.episode_rank_in_task < 0:
            raise ValueError("--episode_rank_in_task must be >= 0")
        task_episode_meta = [
            ep for ep in episodes if ep_to_task.get(ep["episode_index"], -1) == args.task_id
        ]
        if not task_episode_meta:
            raise ValueError(f"No episodes found for task_id={args.task_id}")
        if args.episode_rank_in_task >= len(task_episode_meta):
            raise ValueError(
                f"--episode_rank_in_task={args.episode_rank_in_task} out of range "
                f"(num_episodes_for_task={len(task_episode_meta)})"
            )

        episode_meta = task_episode_meta[args.episode_rank_in_task]
        task_name = task_desc.get(args.task_id, f"task_{args.task_id}")
        print(
            f"Running full episode | task_id={args.task_id} ({task_name}) "
            f"| episode_index={episode_meta['episode_index']}"
        )
        out_path = args.output_path
        root, ext = os.path.splitext(out_path)
        if ext.lower() not in [".mp4", ".gif"]:
            out_path = f"{root}.{args.save_format}"
        mse_mean, mse_std, _ = run_one_full_episode(args.task_id, episode_meta, out_path)
        backbone.offload_vae_and_text_encoder("cpu")
        print(f"Saved full-episode comparison to: {out_path}")
        print(f"Full-episode window MSE mean: {mse_mean:.6f}")
        print(f"Full-episode window MSE std : {mse_std:.6f}")
        return

    if not args.all_tasks:
        if args.sample_idx < 0 or args.sample_idx >= len(test_dataset):
            raise ValueError(f"--sample_idx out of range: {args.sample_idx}, dataset size={len(test_dataset)}")
        print(f"Running single sample: idx={args.sample_idx}, ode_steps={args.ode_steps}")
        batch = test_dataset[args.sample_idx]
        gt_full, pred_full, future_mse = predict_one_sample(
            backbone, fm, batch, num_cond_latent_frames, compute_dtype, device, args.ode_steps
        )
        print(f"Future latent MSE: {future_mse:.6f}")
        out_path = args.output_path
        root, ext = os.path.splitext(out_path)
        if ext.lower() not in [".mp4", ".gif"]:
            out_path = f"{root}.{args.save_format}"
        print(f"Saving to {out_path}...")
        save_side_by_side(gt_full, pred_full, out_path)
        backbone.offload_vae_and_text_encoder("cpu")
        print("Done!")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    task_desc = load_task_descriptions(data_config.precomputed_dir)
    task_to_indices = {}
    for local_idx, global_idx in enumerate(test_dataset.valid_indices):
        task_idx = test_dataset._get_task_index_for_global_idx(global_idx)
        task_to_indices.setdefault(task_idx, []).append(local_idx)

    print(f"Running all tasks with ode_steps={args.ode_steps}, samples_per_task={args.samples_per_task}")
    all_mse = []
    for task_idx in sorted(task_to_indices.keys()):
        indices = task_to_indices[task_idx]
        chosen = indices if args.samples_per_task < 0 else indices[:args.samples_per_task]
        desc = task_desc.get(task_idx, f"task_{task_idx}")
        task_slug = sanitize_filename(desc)
        task_dir = os.path.join(args.output_dir, f"task_{task_idx:02d}_{task_slug}")
        os.makedirs(task_dir, exist_ok=True)
        print(f"\nTask {task_idx}: {desc}")
        print(f"  samples selected: {len(chosen)} / total {len(indices)}")

        for rank_i, sample_idx in enumerate(chosen):
            batch = test_dataset[sample_idx]
            gt_full, pred_full, future_mse = predict_one_sample(
                backbone, fm, batch, num_cond_latent_frames, compute_dtype, device, args.ode_steps
            )
            all_mse.append(future_mse)
            save_name = f"sample_{rank_i:04d}_datasetidx_{sample_idx:07d}_mse_{future_mse:.6f}.{args.save_format}"
            save_path = os.path.join(task_dir, save_name)
            save_side_by_side(gt_full, pred_full, save_path)
            print(f"  [{rank_i + 1}/{len(chosen)}] mse={future_mse:.6f} -> {save_name}")

    backbone.offload_vae_and_text_encoder("cpu")

    if all_mse:
        print("\nAll-task run done.")
        print(f"Future latent MSE mean: {float(np.mean(all_mse)):.6f}")
        print(f"Future latent MSE std : {float(np.std(all_mse)):.6f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
