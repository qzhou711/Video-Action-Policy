"""Stage 2 Trainer: Action decoder training with frozen video backbone.

Trains the lightweight DiT action decoder using flow matching,
conditioned on hidden states from the frozen LoRA-finetuned backbone.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.action_decoder import ActionDecoderDiT
from mimic_video.models.flow_matching import FlowMatchingScheduler


class Stage2Trainer:
    """Stage 2: Action decoder training with frozen video backbone."""

    def __init__(
        self,
        backbone: CosmosVideoBackbone,
        action_decoder: ActionDecoderDiT,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        grad_clip: float = 10.0,
        total_steps: int = 26000,
        gradient_accumulation_steps: int = 32,
        lr_schedule: str = "linear_decay",
        tau_power: float = 0.999,
        dtype: str = "bf16",
        output_dir: str = "checkpoints/stage2",
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
        self.action_decoder = action_decoder
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.total_steps = total_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_schedule = lr_schedule
        self.tau_power = tau_power
        self.output_dir = output_dir
        self.log_every = log_every
        self.save_every = save_every
        self.num_cond_latent_frames = num_cond_latent_frames
        self.device = device
        self.precomputed_t5_embedding = precomputed_t5_embedding
        self.rank = rank
        self.world_size = world_size

        self.compute_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.fm = FlowMatchingScheduler()

        # Freeze backbone completely
        backbone.freeze_for_stage2()
        backbone.eval()

        # Enable gradient checkpointing on transformer for memory efficiency
        if hasattr(backbone.transformer, 'base_model'):
            backbone.transformer.base_model.model.gradient_checkpointing = True
        else:
            backbone.transformer.gradient_checkpointing = True

        # Move action decoder to device
        self.action_decoder = action_decoder.to(device)

        # Setup optimizer (only action decoder params)
        self.optimizer = torch.optim.AdamW(
            action_decoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Setup LR scheduler
        self.lr_scheduler = self._build_lr_scheduler()

        # Wandb logging
        self.use_wandb = wandb_project is not None
        if self.use_wandb and self.rank == 0:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)

        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

    def _build_lr_scheduler(self):
        """Build learning rate scheduler with warmup and linear decay."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            if self.lr_schedule == "linear_decay":
                # Linear decay from warmup_steps to total_steps
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                return max(0.0, 1.0 - progress)
            return 1.0  # constant

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dict with "video", "proprio", "actions".

        Returns:
            Dict with loss metrics.
        """
        video = batch["video"]        # [B, T, C, H, W]
        proprio = batch["proprio"]    # [B, proprio_dim]
        actions = batch["actions"]    # [B, T_action, action_dim]
        B = video.shape[0]

        # Rearrange video to [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4).to(self.device)
        proprio = proprio.to(self.device).float()
        actions = actions.to(self.device).float()

        # 1. Encode video with frozen VAE
        with torch.no_grad():
            self.backbone.move_vae_to(self.device)
            z_0 = self.backbone.encode_video(video)
            self.backbone.offload_vae_and_text_encoder("cpu")

        # Split into conditioning and prediction
        z_cond = z_0[:, :, :self.num_cond_latent_frames]
        z_pred = z_0[:, :, self.num_cond_latent_frames:]

        # 2. Sample video noise and timestep
        tau_v = self.fm.sample_tau_video(B, device=z_pred.device)
        eps_v = torch.randn_like(z_pred)
        z_noisy = self.fm.interpolate(z_pred, eps_v, tau_v)

        # 3. Forward through frozen backbone, extract hidden states
        T_total = z_0.shape[2]

        if self.precomputed_t5_embedding is not None:
            t5_emb = self.precomputed_t5_embedding.to(self.device, dtype=self.compute_dtype)
            if t5_emb.shape[0] == 1:
                t5_emb = t5_emb.expand(B, -1, -1)
        elif "t5_embedding" in batch:
            t5_emb = batch["t5_embedding"].to(self.device, dtype=self.compute_dtype)
        else:
            raise ValueError("No T5 embedding available.")

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                self.backbone.forward_transformer(
                    z_noisy=z_noisy,
                    z_cond=z_cond,
                    tau_v=tau_v,
                    encoder_hidden_states=t5_emb,
                )

        # Extract and pool hidden states
        h_raw = self.backbone.get_captured_hidden_states()  # [B, T*H'*W', hidden_dim]
        if h_raw is None:
            raise RuntimeError("No hidden states captured. Check hook registration.")

        h_pooled = self.backbone.pool_hidden_states(
            h_raw.float(), num_latent_frames=T_total
        )  # [B, T_lat, hidden_dim]

        # Detach hidden states (backbone is frozen, but be explicit)
        h_pooled = h_pooled.detach()

        # 4. Sample action noise and timestep
        tau_a = self.fm.sample_tau_action(B, device=actions.device, power=self.tau_power)
        eps_a = torch.randn_like(actions)

        # Create noisy actions
        a_noisy = self.fm.interpolate(actions, eps_a, tau_a)

        # 5. Forward through action decoder
        with torch.amp.autocast("cuda", dtype=self.compute_dtype):
            velocity_pred = self.action_decoder(
                noisy_actions=a_noisy,
                proprio=proprio,
                h_video=h_pooled,
                tau_a=tau_a,
                tau_v=tau_v,
                training=True,
            )  # [B, T_action, action_dim]

        # 6. Compute loss
        velocity_target = self.fm.velocity_target(actions, eps_a)
        loss = self.fm.compute_loss(velocity_pred.float(), velocity_target.float())

        # Scale for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        return {"loss": loss.item() * self.gradient_accumulation_steps}

    def train(self):
        """Run the full training loop."""
        self.backbone.eval()
        self.action_decoder.train()

        data_iter = iter(self.train_dataloader)
        running_loss = 0.0
        global_step = 0

        if self.rank == 0:
            pbar = tqdm(total=self.total_steps, desc="Stage 2 Training")
        else:
            pbar = None

        while global_step < self.total_steps:
            self.optimizer.zero_grad()

            # Gradient accumulation
            for micro_step in range(self.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                metrics = self.train_step(batch)
                running_loss += metrics["loss"]

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.action_decoder.parameters(), self.grad_clip
            )

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            global_step += 1
            avg_loss = running_loss / self.gradient_accumulation_steps
            running_loss = 0.0

            # Logging
            if global_step % self.log_every == 0 and self.rank == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                if pbar:
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step": global_step,
                    }, step=global_step)

            # Save checkpoint
            if self.rank == 0 and global_step % self.save_every == 0:
                self._save_checkpoint(global_step)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # Save final checkpoint
        if self.rank == 0:
            self._save_checkpoint(global_step, is_final=True)
            print(f"Stage 2 training complete. Final checkpoint saved to {self.output_dir}")

    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Save action decoder checkpoint."""
        suffix = "final" if is_final else f"step_{step}"
        save_path = os.path.join(self.output_dir, suffix)
        os.makedirs(save_path, exist_ok=True)

        # Save action decoder weights
        torch.save(
            self.action_decoder.state_dict(),
            os.path.join(save_path, "action_decoder.pt"),
        )

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "step": step,
        }, os.path.join(save_path, "training_state.pt"))

        print(f"Checkpoint saved to {save_path}")

    def _load_checkpoint(self, path: str) -> int:
        """Load a checkpoint. Returns the step number."""
        # Load action decoder
        decoder_path = os.path.join(path, "action_decoder.pt")
        if os.path.exists(decoder_path):
            self.action_decoder.load_state_dict(
                torch.load(decoder_path, map_location=self.device, weights_only=True)
            )

        # Load training state
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            return state["step"]
        return 0
