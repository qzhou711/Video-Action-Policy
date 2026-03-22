"""Stage 1 Trainer: LoRA finetuning of the Cosmos video backbone.

Trains the video backbone with LoRA to predict future video frames
using flow matching, conditioned on past frames and text embeddings.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.flow_matching import FlowMatchingScheduler


class Stage1Trainer:
    """Stage 1: LoRA finetuning of the video backbone for video prediction."""

    def __init__(
        self,
        backbone: CosmosVideoBackbone,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        grad_clip: float = 10.0,
        total_steps: int = 27000,
        gradient_accumulation_steps: int = 256,
        lr_schedule: str = "constant",
        dtype: str = "bf16",
        output_dir: str = "checkpoints/stage1",
        log_every: int = 10,
        save_every: int = 1000,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        precomputed_t5_embedding: Optional[torch.Tensor] = None,
        num_cond_latent_frames: int = 2,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.backbone = backbone
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.total_steps = total_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_schedule = lr_schedule
        self.output_dir = output_dir
        self.log_every = log_every
        self.save_every = save_every
        self.num_cond_latent_frames = num_cond_latent_frames
        self.device = device
        self.precomputed_t5_embedding = precomputed_t5_embedding
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.is_distributed = (world_size > 1)

        self.compute_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.fm = FlowMatchingScheduler()

        # Enable gradient checkpointing (unwrap DDP if needed)
        transformer_unwrapped = self._unwrap_transformer()
        if hasattr(transformer_unwrapped, 'base_model'):
            transformer_unwrapped.base_model.model.gradient_checkpointing = True
        else:
            transformer_unwrapped.gradient_checkpointing = True

        # Freeze VAE and text encoder
        for param in backbone.vae.parameters():
            param.requires_grad = False
        if backbone.text_encoder is not None:
            for param in backbone.text_encoder.parameters():
                param.requires_grad = False

        # Setup optimizer (only LoRA params)
        trainable_params = [p for p in self.backbone.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Setup LR scheduler
        self.lr_scheduler = self._build_lr_scheduler()

        # Wandb logging
        self.val_every = save_every  # visual validation at same cadence as checkpoints
        self.ode_steps = 20  # Euler steps for denoising during validation

        self.use_wandb = wandb_project is not None and self.is_main
        if self.use_wandb:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)

        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)

    def _unwrap_transformer(self):
        """Get the underlying transformer module (unwrap DDP if needed)."""
        t = self.backbone.transformer
        if hasattr(t, 'module'):
            return t.module
        return t

    def _build_lr_scheduler(self):
        """Build learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            # Constant LR after warmup for stage 1
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dict with "video" [B, T, C, H, W] frames in [-1, 1].

        Returns:
            Dict with loss metrics and timing info.
        """
        timings = {}

        t0 = time.time()
        video = batch["video"]  # [B, T, C, H, W]
        B = video.shape[0]

        # Rearrange to [B, C, T, H, W] for VAE
        video = video.permute(0, 2, 1, 3, 4).to(self.device)
        timings["data_to_gpu"] = time.time() - t0

        # Encode with VAE (frozen, no grad) — VAE stays on GPU
        t0 = time.time()
        with torch.no_grad():
            z_0 = self.backbone.encode_video(video)  # [B, C_lat, T_lat, H_lat, W_lat]
        timings["vae_encode"] = time.time() - t0

        # Split into conditioning and prediction
        z_cond = z_0[:, :, :self.num_cond_latent_frames]   # [B, C, T_cond, H, W]
        z_pred = z_0[:, :, self.num_cond_latent_frames:]   # [B, C, T_pred, H, W]

        # Sample noise and timesteps
        eps_v = torch.randn_like(z_pred)
        tau_v = self.fm.sample_tau_video(B, device=z_pred.device)  # [B]

        # Create noisy latents: z_tau = (1-tau)*z_pred + tau*eps
        z_noisy = self.fm.interpolate(z_pred, eps_v, tau_v)

        # Get T5 text embedding
        if self.precomputed_t5_embedding is not None:
            t5_emb = self.precomputed_t5_embedding.to(self.device, dtype=self.compute_dtype)
            if t5_emb.shape[0] == 1:
                t5_emb = t5_emb.expand(B, -1, -1)
        elif "t5_embedding" in batch:
            t5_emb = batch["t5_embedding"].to(self.device, dtype=self.compute_dtype)
        else:
            raise ValueError("No T5 embedding available. Either precompute or include in batch.")

        # Forward through transformer (LoRA active)
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=self.compute_dtype):
            raw_output, _ = self.backbone.forward_transformer(
                z_noisy=z_noisy,
                z_cond=z_cond,
                tau_v=tau_v,
                encoder_hidden_states=t5_emb,
            )
        timings["transformer_fwd"] = time.time() - t0

        # The raw output of the Cosmos network IS the velocity v = eps - x_0
        # We only compute loss on the prediction frames (not conditioning frames)
        T_cond = self.num_cond_latent_frames
        velocity_pred = raw_output[:, :, T_cond:]  # [B, C, T_pred, H, W]
        velocity_target = self.fm.velocity_target(z_pred, eps_v)  # [B, C, T_pred, H, W]

        loss = self.fm.compute_loss(velocity_pred.float(), velocity_target.float())

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        t0 = time.time()
        loss.backward()
        timings["backward"] = time.time() - t0

        return {"loss": loss.item() * self.gradient_accumulation_steps, "timings": timings}

    def train(self, start_step: int = 0):
        """Run the full training loop."""
        self.backbone.transformer.train()
        self.backbone.vae.eval()

        # Move VAE to GPU once — keep it there for all micro-batches
        self.backbone.move_vae_to(self.device)

        data_iter = iter(self.train_dataloader)
        running_loss = 0.0
        global_step = start_step
        epoch = start_step // len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0

        # Timing accumulators for logging
        timing_accum = {}

        pbar = tqdm(total=self.total_steps, initial=start_step, desc="Stage 1 Training", disable=not self.is_main)

        while global_step < self.total_steps:
            self.optimizer.zero_grad()
            step_start = time.time()

            # Reset timing accumulators for this step
            timing_accum = {}

            # Gradient accumulation
            for micro_step in range(self.gradient_accumulation_steps):
                t_data = time.time()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if self.is_distributed and hasattr(self.train_dataloader, 'sampler'):
                        self.train_dataloader.sampler.set_epoch(epoch)
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                data_load_time = time.time() - t_data
                timing_accum.setdefault("data_load", []).append(data_load_time)

                metrics = self.train_step(batch)
                running_loss += metrics["loss"]

                # Accumulate micro-step timings
                for k, v in metrics.get("timings", {}).items():
                    timing_accum.setdefault(k, []).append(v)

                # Log every micro-batch so we can see progress in real time (rank 0 only)
                if self.is_main:
                    micro_timings = metrics.get("timings", {})
                    elapsed = time.time() - step_start
                    print(f"  micro {micro_step+1}/{self.gradient_accumulation_steps} | "
                          f"data={data_load_time:.2f}s "
                          f"vae={micro_timings.get('vae_encode', 0):.2f}s "
                          f"fwd={micro_timings.get('transformer_fwd', 0):.2f}s "
                          f"bwd={micro_timings.get('backward', 0):.2f}s "
                          f"gpu_xfer={micro_timings.get('data_to_gpu', 0):.2f}s "
                          f"| elapsed={elapsed:.1f}s",
                          flush=True)

            step_time = time.time() - step_start

            # Clip gradients and record norm
            # Synchronize loss across GPUs for accurate logging
            if self.is_distributed:
                loss_tensor = torch.tensor([running_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                running_loss = loss_tensor.item()

            trainable_params = [p for p in self.backbone.transformer.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            global_step += 1
            avg_loss = running_loss / self.gradient_accumulation_steps
            running_loss = 0.0

            # Logging (rank 0 only)
            if global_step % self.log_every == 0 and self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]

                # Compute average timings across micro-batches
                avg_timings = {k: sum(v) / len(v) for k, v in timing_accum.items()}
                total_timings = {k: sum(v) for k, v in timing_accum.items()}

                timing_str = " | ".join(f"{k}={total_timings[k]:.1f}s" for k in sorted(total_timings))
                print(f"\n  [step {global_step}] step_time={step_time:.1f}s | {timing_str}")
                print(f"  [step {global_step}] per micro-batch avg: " +
                      " | ".join(f"{k}={avg_timings[k]:.2f}s" for k in sorted(avg_timings)))

                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", gnorm=f"{grad_norm:.2f}",
                                 step_s=f"{step_time:.1f}")

                if self.use_wandb:
                    import wandb
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/step": global_step,
                        "perf/step_time_s": step_time,
                    }
                    for k, v in avg_timings.items():
                        log_dict[f"perf/avg_{k}_s"] = v
                    for k, v in total_timings.items():
                        log_dict[f"perf/total_{k}_s"] = v
                    wandb.log(log_dict, step=global_step)

            # Save checkpoint + visual validation (rank 0 only for save)
            if global_step % self.save_every == 0:
                # Offload VAE before validation (validation manages its own VAE placement)
                self.backbone.offload_vae_and_text_encoder("cpu")
                if self.is_main:
                    self._save_checkpoint(global_step)
                    if self.use_wandb:
                        self.validate_visual(global_step)
                # Sync all ranks before continuing
                if self.is_distributed:
                    dist.barrier()
                # Move VAE back to GPU for training
                self.backbone.move_vae_to(self.device)

            pbar.update(1)

        pbar.close()

        # Offload VAE before final save
        self.backbone.offload_vae_and_text_encoder("cpu")

        # Save final checkpoint (rank 0 only)
        if self.is_main:
            self._save_checkpoint(global_step, is_final=True)
            print(f"Stage 1 training complete. Final checkpoint saved to {self.output_dir}")
        if self.is_distributed:
            dist.barrier()

    @torch.no_grad()
    def validate_visual(self, step: int):
        """Generate predicted video frames and log side-by-side comparison to wandb."""
        import wandb
        self.backbone.transformer.eval()

        # Grab one sample from dataloader
        try:
            batch = next(iter(self.train_dataloader))
        except StopIteration:
            return

        video = batch["video"][:1]  # [1, T, C, H, W]
        video = video.permute(0, 2, 1, 3, 4).to(self.device)  # [1, C, T, H, W]

        # Encode full video to latents
        self.backbone.move_vae_to(self.device)
        z_0 = self.backbone.encode_video(video)  # [1, C_lat, T_lat, H_lat, W_lat]

        z_cond = z_0[:, :, :self.num_cond_latent_frames]
        z_pred_gt = z_0[:, :, self.num_cond_latent_frames:]

        # Get T5 embedding
        if self.precomputed_t5_embedding is not None:
            t5_emb = self.precomputed_t5_embedding.to(self.device, dtype=self.compute_dtype)
        else:
            t5_emb = batch["t5_embedding"][:1].to(self.device, dtype=self.compute_dtype)

        # Start from pure noise and denoise via Euler ODE
        z_noise = torch.randn_like(z_pred_gt)

        def model_fn(z_t, tau):
            tau_tensor = torch.tensor([tau], device=z_t.device, dtype=z_t.dtype)
            with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                raw_out, _ = self.backbone.forward_transformer(
                    z_noisy=z_t,
                    z_cond=z_cond,
                    tau_v=tau_tensor,
                    encoder_hidden_states=t5_emb,
                )
            T_cond = self.num_cond_latent_frames
            return raw_out[:, :, T_cond:]  # velocity for prediction frames only

        z_pred_denoised = self.fm.ode_solve_euler(
            model_fn, z_noise, num_steps=self.ode_steps, tau_start=1.0, tau_end=0.0
        )

        # Decode ground truth and predicted latents to pixels
        gt_full = self.backbone.decode_video(z_0)  # [1, C, T, H, W]
        pred_latents = torch.cat([z_cond, z_pred_denoised], dim=2)
        pred_full = self.backbone.decode_video(pred_latents)  # [1, C, T, H, W]

        self.backbone.offload_vae_and_text_encoder("cpu")

        # Convert to uint8 numpy: [T, H, W, C]
        def to_video_np(x):
            x = (x.squeeze(0).permute(1, 2, 3, 0).clamp(-1, 1) * 0.5 + 0.5) * 255
            return x.cpu().to(torch.uint8).numpy()

        gt_np = to_video_np(gt_full)
        pred_np = to_video_np(pred_full)

        # Side-by-side: stack horizontally
        side_by_side = np.concatenate([gt_np, pred_np], axis=2)  # [T, H, 2*W, C]

        wandb.log({
            "val/video_comparison": wandb.Video(
                side_by_side.transpose(0, 3, 1, 2),  # [T, C, H, 2*W] for wandb
                fps=4, format="mp4", caption=f"Left: GT | Right: Predicted (step {step})"
            ),
        }, step=step)

        # Also log individual frame grid at the midpoint
        T = gt_np.shape[0]
        mid = T // 2
        frames_to_log = {
            "val/cond_frame_0": wandb.Image(gt_np[0], caption="Conditioning frame 0"),
            "val/gt_mid": wandb.Image(gt_np[mid], caption=f"GT frame {mid}"),
            "val/pred_mid": wandb.Image(pred_np[mid], caption=f"Pred frame {mid}"),
            "val/gt_last": wandb.Image(gt_np[-1], caption=f"GT frame {T-1}"),
            "val/pred_last": wandb.Image(pred_np[-1], caption=f"Pred frame {T-1}"),
        }
        wandb.log(frames_to_log, step=step)

        self.backbone.transformer.train()
        print(f"  [val] Visual validation logged to wandb at step {step}")

    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Save LoRA checkpoint."""
        suffix = "final" if is_final else f"step_{step}"
        save_path = os.path.join(self.output_dir, suffix)
        os.makedirs(save_path, exist_ok=True)

        # Unwrap DDP for saving
        transformer_unwrapped = self._unwrap_transformer()
        # Temporarily swap to unwrapped for saving
        original_transformer = self.backbone.transformer
        self.backbone.transformer = transformer_unwrapped
        self.backbone.save_lora(save_path)
        self.backbone.transformer = original_transformer

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "step": step,
        }, os.path.join(save_path, "training_state.pt"))

        print(f"Checkpoint saved to {save_path}")

    def _load_checkpoint(self, path: str):
        """Load a checkpoint."""
        # Load LoRA weights (ensure they are trainable for stage 1 resumption)
        self.backbone.load_lora(path, is_trainable=True)

        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            return state["step"]
        return 0
