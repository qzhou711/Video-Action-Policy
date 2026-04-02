"""Inference policy for mimic-video (Algorithm 1 from the paper).

Implements the full inference pipeline:
1. Encode past frames with VAE
2. Optionally partially denoise future video frames
3. Extract hidden states from the backbone at a specified noise level
4. Fully denoise actions using the action decoder via Euler ODE integration
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, List

from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.action_decoder import ActionDecoderDiT
from mimic_video.models.flow_matching import FlowMatchingScheduler
from mimic_video.data.transforms import concat_cameras_2x2, normalize_to_neg1_pos1


class MimicVideoPolicy(nn.Module):
    """Inference policy implementing Algorithm 1 from the mimic-video paper.

    At inference time:
    1. Encode the last 5 condition frames (17 pixel -> 5 latent, 2 cond)
       Actually: we use the 2 conditioning latent frames from encoded video
    2. Sample noise for future frames (or partially denoise)
    3. Forward through backbone at noise level tau_v to get hidden states
    4. Fully denoise actions from noise via Euler integration (10 steps)
    5. Return denormalized action chunk

    Key insight: at tau_v=1 (default), only ONE backbone forward pass is needed.
    """

    def __init__(
        self,
        backbone: CosmosVideoBackbone,
        action_decoder: ActionDecoderDiT,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
        t5_embedding: Optional[torch.Tensor] = None,
        t5_embeddings_dict: Optional[Dict[int, torch.Tensor]] = None,
        task_descriptions: Optional[Dict[int, str]] = None,
        tau_v: float = 1.0,
        num_video_denoise_steps: int = 0,
        num_action_denoise_steps: int = 10,
        num_cond_latent_frames: int = 2,
        num_pred_latent_frames: int = 3,
        num_pixel_frames: int = 17,
        num_infer_real_frames: int = 5,
        action_stats_path: str = None,
        camera_names: list = None,
        target_height: int = 480,
        target_width: int = 640,
        action_norm_type: str = "min-max",
        hidden_state_pool: str = "mean",
        device: str = "cuda",
    ):
        super().__init__()

        self.backbone = backbone
        self.action_decoder = action_decoder
        self.fm = FlowMatchingScheduler()
        self.tau_v = tau_v
        self.num_video_denoise_steps = num_video_denoise_steps
        self.num_action_denoise_steps = num_action_denoise_steps
        self.num_cond_latent_frames = num_cond_latent_frames
        self.num_pred_latent_frames = num_pred_latent_frames
        self.num_pixel_frames = num_pixel_frames
        self.num_infer_real_frames = num_infer_real_frames
        self.camera_names = camera_names or []
        self.target_height = target_height
        self.target_width = target_width
        self.action_norm_type = action_norm_type
        self.hidden_state_pool = hidden_state_pool
        self.device = device
        if self.num_infer_real_frames <= 0:
            raise ValueError("num_infer_real_frames must be > 0")
        if self.num_infer_real_frames > self.num_pixel_frames:
            raise ValueError(
                f"num_infer_real_frames ({self.num_infer_real_frames}) must be <= "
                f"num_pixel_frames ({self.num_pixel_frames})"
            )

        # Action normalization stats
        self.action_mean = None
        self.action_std = None
        self.action_min = None
        self.action_max = None

        if action_stats is not None:
            # Direct dict passed in (e.g. from eval_server)
            self.action_mean = action_stats.get("mean", None)
            self.action_std = action_stats.get("std", None)
            self.action_min = action_stats.get("min", None)
            self.action_max = action_stats.get("max", None)
        elif action_stats_path and os.path.exists(action_stats_path):
            print(f"Loading action stats from {action_stats_path}")
            stats = torch.load(action_stats_path, map_location="cpu", weights_only=True)
            self.action_mean = stats.get("mean", None)
            self.action_std = stats.get("std", None)
            self.action_min = stats.get("min", None)
            self.action_max = stats.get("max", None)
        else:
            print(f"Warning: No action stats found. Output actions won't be denormalized.")

        # T5 embeddings: single-task or multi-task
        self.t5_embedding = t5_embedding
        self.t5_embeddings_dict = t5_embeddings_dict or {}
        self.task_descriptions = task_descriptions or {}

    def get_t5_embedding_for_prompt(self, prompt: str) -> Optional[torch.Tensor]:
        """Find the precomputed T5 embedding matching the given prompt.

        Matches by checking if the prompt is a substring of any task description
        (or vice versa), with case-insensitive comparison.

        Args:
            prompt: Task description string from the LIBERO client.

        Returns:
            Matching T5 embedding [1, seq_len, dim], or None if no match found.
        """
        if not self.t5_embeddings_dict:
            return self.t5_embedding

        prompt_lower = prompt.strip().lower()

        # Exact match first
        for task_idx, desc in self.task_descriptions.items():
            if desc.strip().lower() == prompt_lower:
                return self.t5_embeddings_dict[task_idx]

        # Substring match
        for task_idx, desc in self.task_descriptions.items():
            desc_lower = desc.strip().lower()
            if prompt_lower in desc_lower or desc_lower in prompt_lower:
                return self.t5_embeddings_dict[task_idx]

        # No match — fallback to first embedding
        first_key = sorted(self.t5_embeddings_dict.keys())[0]
        return self.t5_embeddings_dict[first_key]

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions back to original scale."""
        if self.action_norm_type == "min-max":
            if self.action_min is None or self.action_max is None:
                return actions
            # Denormalize from [-1, 1] to original range
            actions = (actions + 1) / 2  # Scale to [0, 1]
            actions = actions * (self.action_max.to(actions.device) - self.action_min.to(actions.device) + 1e-4) + self.action_min.to(actions.device)
        elif self.action_norm_type == "mean-std":
            if self.action_mean is None or self.action_std is None:
                return actions
            actions = actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
        else:
            raise ValueError(f"Unknown action normalization type: {self.action_norm_type}")
        return actions

    @torch.no_grad()
    def predict_action(
        self,
        video_frames: torch.Tensor,
        proprio: torch.Tensor,
        t5_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict action chunk from observation.

        This implements Algorithm 1 from the mimic-video paper.

        Args:
            video_frames: Concatenated 2x2 camera frames [1, T, C, H, W] in [-1, 1].
                T can be any positive length. Only the most recent
                num_infer_real_frames are encoded with VAE.
            proprio: Current proprioception [1, proprio_dim].
            t5_embedding: Optional T5 text embedding [1, seq_len, text_dim].

        Returns:
            Predicted action chunk [1, action_chunk_size, action_dim] (denormalized).
        """
        B = 1  # Single sample inference
        device = self.device

        video_frames = video_frames.to(device)
        proprio = proprio.to(device).float()
        if video_frames.ndim != 5 or video_frames.shape[1] < 1:
            raise ValueError(f"Expected video_frames shape [B, T, C, H, W], got {tuple(video_frames.shape)}")
        if video_frames.shape[1] > self.num_infer_real_frames:
            video_frames = video_frames[:, -self.num_infer_real_frames:]

        # Get T5 embedding
        if t5_embedding is not None:
            t5_emb = t5_embedding.to(device)
        elif self.t5_embedding is not None:
            t5_emb = self.t5_embedding.to(device)
        else:
            raise ValueError("No T5 embedding available for inference.")

        # 1. Encode video frames with VAE
        # video_frames: [1, T, C, H, W] -> [1, C, T, H, W]
        video_bcthw = video_frames.permute(0, 2, 1, 3, 4)

        self.backbone.move_vae_to(device)
        z_all = self.backbone.encode_video(video_bcthw)  # [1, C_lat, T_lat, H_lat, W_lat]
        self.backbone.move_vae_to("cpu")

        # Split: z_cond (first 2 latent frames), z_pred_clean (last 3, for reference)
        z_cond = z_all[:, :, :self.num_cond_latent_frames]
        T_lat_total = z_all.shape[2]

        # 2. Handle future video frames
        C_lat, H_lat, W_lat = z_all.shape[1], z_all.shape[3], z_all.shape[4]

        if self.tau_v >= 1.0:
            # tau_v = 1: pure noise, no video denoising needed (fastest)
            z_future = torch.randn(
                B, C_lat, self.num_pred_latent_frames, H_lat, W_lat, device=device
            )
            current_tau_v = torch.ones(B, device=device)
        elif self.num_video_denoise_steps > 0:
            # Partially denoise future frames from tau=1 to tau=tau_v
            z_future = torch.randn(
                B, C_lat, self.num_pred_latent_frames, H_lat, W_lat, device=device
            )

            # Euler integration from tau=1 to tau=tau_v
            num_steps = self.num_video_denoise_steps
            dt = (self.tau_v - 1.0) / num_steps
            tau = 1.0

            for _ in range(num_steps):
                tau_tensor = torch.full((B,), tau, device=device)
                _, full_output = self.backbone.forward_transformer(
                    z_noisy=z_future,
                    z_cond=z_cond,
                    tau_v=tau_tensor,
                    encoder_hidden_states=t5_emb,
                )
                # Derive velocity from x_0 prediction: v = (x_noisy - x_0) / tau
                T_cond = self.num_cond_latent_frames
                x0_pred = full_output[:, :, T_cond:]
                v_pred = (z_future - x0_pred) / max(tau, 1e-6)
                z_future = z_future + v_pred * dt
                tau = tau + dt

            current_tau_v = torch.full((B,), self.tau_v, device=device)
        else:
            # Use the encoded future frames at tau_v=0 (clean)
            z_future = z_all[:, :, self.num_cond_latent_frames:]
            current_tau_v = torch.zeros(B, device=device)

        # 3. Forward through backbone to extract hidden states at tau_v
        self.backbone.forward_transformer(
            z_noisy=z_future,
            z_cond=z_cond,
            tau_v=current_tau_v,
            encoder_hidden_states=t5_emb.to(self.backbone.dtype),
        )

        # Get and pool hidden states
        h_raw = self.backbone.get_captured_hidden_states()
        h_pooled = self.backbone.pool_hidden_states(
            h_raw.float(), num_latent_frames=T_lat_total, mode=self.hidden_state_pool
        )  # shape depends on pool mode

        # 4. Fully denoise actions via Euler integration
        action_chunk_size = self.action_decoder.action_chunk_size
        action_dim = self.action_decoder.action_dim

        # Start from pure noise
        a_noise = torch.randn(B, action_chunk_size, action_dim, device=device)

        def action_model_fn(a_t, tau):
            tau_a_tensor = torch.full((B,), tau, device=device)
            velocity = self.action_decoder(
                noisy_actions=a_t,
                proprio=proprio,
                h_video=h_pooled,
                t5_embedding=t5_emb,
                tau_a=tau_a_tensor,
                tau_v=current_tau_v,
                training=False,
            )
            return velocity

        # Euler ODE solve from tau=1 (noise) to tau=0 (clean)
        a_clean = self.fm.ode_solve_euler(
            model_fn=action_model_fn,
            x_init=a_noise,
            num_steps=self.num_action_denoise_steps,
            tau_start=1.0,
            tau_end=0.0,
        )

        # 5. Denormalize actions
        a_clean = self.denormalize_actions(a_clean)

        return a_clean  # [1, action_chunk_size, action_dim]

    @torch.no_grad()
    def predict_action_from_obs(
        self,
        camera_images: dict,
        proprio: torch.Tensor,
        t5_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict actions directly from raw camera observations.

        This is a convenience method that handles camera concatenation.

        Args:
            camera_images: Dict mapping camera names to image tensors [T, C, H, W].
            proprio: Current proprioception [proprio_dim].
            t5_embedding: Optional T5 text embedding.

        Returns:
            Action chunk [1, action_chunk_size, action_dim] (denormalized).
        """
        # Concatenate cameras
        frames_list = [camera_images[name] for name in self.camera_names]
        video = concat_cameras_2x2(frames_list, self.target_height, self.target_width)
        video = normalize_to_neg1_pos1(video)

        # Add batch dimension
        video = video.unsqueeze(0)  # [1, T, C, H, W]
        proprio = proprio.unsqueeze(0)  # [1, proprio_dim]

        return self.predict_action(video, proprio, t5_embedding)
