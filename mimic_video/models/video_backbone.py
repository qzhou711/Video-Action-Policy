"""Video backbone wrapper around Cosmos Predict2 Video2World pipeline.

Handles:
- Loading and configuring the Cosmos pipeline components
- VAE encoding/decoding with proper normalization
- LoRA application to the transformer
- Hidden state extraction via forward hooks
- Spatial pooling of hidden states for the action decoder
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from peft import LoraConfig, get_peft_model


class CosmosVideoBackbone(nn.Module):
    """Wrapper around Cosmos Predict2 2B Video2World for mimic-video.

    Extracts the transformer, VAE, and text encoder from the pipeline,
    applies LoRA to the transformer, and provides hooks for hidden state
    extraction at a specified layer.
    """

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos-Predict2-2B-Video2World",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: list = None,
        hidden_state_layer: int = 19,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()

        self.model_id = model_id
        self.hidden_state_layer = hidden_state_layer
        self.dtype = dtype
        self._device = device

        # Storage for captured hidden states
        self._hidden_states_cache = {}
        self._hook_handles = []

        # Load pipeline components
        self._load_pipeline(model_id, dtype)

        # Get actual hidden dim from the loaded transformer
        self.hidden_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.patch_size = self.transformer.config.patch_size

        # Apply LoRA to transformer
        if lora_target_modules is None:
            lora_target_modules = [
                "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                "ff.net.0.proj", "ff.net.2",
            ]

        self._apply_lora(lora_rank, lora_alpha, lora_target_modules)

        # Register hidden state extraction hook
        self._register_hooks()

        # Get VAE latent normalization constants
        self._setup_vae_normalization()

    def _load_pipeline(self, model_id: str, dtype: torch.dtype):
        """Load the Cosmos pipeline and extract components."""
        import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _cosmos_mod

        # Stub out CosmosSafetyChecker to avoid needing the gated Guardrail repo
        _cosmos_mod.CosmosSafetyChecker = lambda *a, **kw: None

        from diffusers import Cosmos2VideoToWorldPipeline

        pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )

        # cosmos_guardrail import disables grad globally - re-enable it
        torch.set_grad_enabled(True)

        # Extract components
        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler

        # Store VAE scale factors
        self.vae_scale_factor_temporal = pipeline.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = pipeline.vae_scale_factor_spatial

        # Delete the pipeline shell (we keep the components)
        del pipeline

    def _setup_vae_normalization(self):
        """Set up VAE latent normalization constants."""
        z_dim = self.vae.config.z_dim
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1),
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1),
        )
        self.sigma_data = self.scheduler.config.sigma_data

    def _apply_lora(self, rank: int, alpha: int, target_modules: list):
        """Apply LoRA to the transformer."""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        self.transformer = get_peft_model(self.transformer, lora_config)

    def _register_hooks(self):
        """Register forward hooks on the specified transformer block for hidden state extraction."""
        # Clear existing hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

        transformer = self.transformer.module if hasattr(self.transformer, "module") else self.transformer
        block = transformer.base_model.model.transformer_blocks[self.hidden_state_layer]

        def hook_fn(module, input, output):
            # output is the hidden_states tensor after the block: [B, T*H'*W', hidden_dim]
            self._hidden_states_cache["layer_output"] = output

        handle = block.register_forward_hook(hook_fn)
        self._hook_handles.append(handle)

    def get_captured_hidden_states(self) -> Optional[torch.Tensor]:
        """Retrieve the hidden states captured by the hook.

        Returns:
            Hidden states from layer k, shape [B, T*H'*W', hidden_dim], or None.
        """
        return self._hidden_states_cache.get("layer_output", None)

    def clear_hidden_states_cache(self):
        """Clear the captured hidden states."""
        self._hidden_states_cache.clear()

    @torch.no_grad()
    def encode_video(self, pixel_frames: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to VAE latents with normalization.

        Args:
            pixel_frames: Video tensor [B, C, T, H, W] in [-1, 1].

        Returns:
            Normalized latents [B, z_dim, T_lat, H_lat, W_lat].
        """
        # VAE expects [B, C, T, H, W]
        vae_device = next(self.vae.parameters()).device
        vae_dtype = self.vae.dtype

        pixel_frames = pixel_frames.to(device=vae_device, dtype=vae_dtype)
        posterior = self.vae.encode(pixel_frames).latent_dist
        latents = posterior.mode()  # Use mode (deterministic) for training

        # Normalize: (latents - mean) / std * sigma_data
        latents = latents.float()
        latents = (latents - self.latents_mean.to(latents.device)) / self.latents_std.to(latents.device) * self.sigma_data

        return latents

    @torch.no_grad()
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents back to pixel frames.

        Args:
            latents: Normalized latents [B, z_dim, T_lat, H_lat, W_lat].

        Returns:
            Pixel frames [B, C, T, H, W] in [-1, 1].
        """
        vae_device = next(self.vae.parameters()).device
        vae_dtype = self.vae.dtype

        # Denormalize: latents * std / sigma_data + mean
        latents = latents * self.latents_std.to(latents.device) / self.sigma_data + self.latents_mean.to(latents.device)
        latents = latents.to(device=vae_device, dtype=vae_dtype)

        decoded = self.vae.decode(latents, return_dict=False)[0]
        return decoded

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt to T5 embeddings.

        Args:
            prompt: Text string to encode.

        Returns:
            T5 text embeddings [1, seq_len, text_embed_dim].
        """
        text_encoder_device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(text_encoder_device)
        attention_mask = text_inputs.attention_mask.to(text_encoder_device)

        text_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

        return text_embeds

    def forward_transformer(
        self,
        z_noisy: torch.Tensor,
        z_cond: torch.Tensor,
        tau_v: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Cosmos transformer.

        This constructs the proper input format for the Cosmos transformer,
        including conditioning frames, per-frame timesteps, and condition masks.

        Args:
            z_noisy: Noisy latents for prediction frames [B, C, T_pred, H_lat, W_lat].
            z_cond: Clean conditioning latents [B, C, T_cond, H_lat, W_lat].
            tau_v: Video denoising timestep(s) [B] in [0, 1].
            encoder_hidden_states: T5 text embeddings [B, seq_len, text_dim].
            condition_mask: Optional [B, 1, T_total, H_lat, W_lat] mask.

        Returns:
            Transformer output (velocity prediction) [B, C, T_total, H_lat, W_lat].
        """
        B, C, T_cond, H, W = z_cond.shape
        T_pred = z_noisy.shape[2]
        T_total = T_cond + T_pred
        device = z_noisy.device

        # Cosmos parameterization: t = sigma / (sigma + 1)
        # For flow matching: c_in = 1-t scales the noisy input
        # The input to transformer is: c_in * z_noisy = (1-t) * z_noisy
        # But we've already constructed z_noisy = (1-tau)*z_0 + tau*eps
        # So we need to scale by c_in = (1-t) where t maps from our tau

        # In Cosmos, the raw network output = velocity v = eps - x_0
        # So we directly use tau_v as the timestep t

        # Construct per-frame timesteps [B, 1, T_total, 1, 1]
        # Conditioning frames get near-zero timestep (sigma_conditioning ~ 0.0001)
        sigma_cond = 0.0001
        t_cond = sigma_cond / (sigma_cond + 1)  # near zero

        # Build per-frame timestep tensor
        timestep = torch.zeros(B, 1, T_total, 1, 1, device=device, dtype=z_noisy.dtype)
        timestep[:, :, :T_cond] = t_cond
        for b in range(B):
            t_val = tau_v[b].item() if tau_v.ndim > 0 else tau_v.item()
            timestep[b, :, T_cond:] = t_val

        # Scale inputs using Cosmos parameterization
        # c_in = 1 - t for each frame
        c_in_cond = 1.0 - t_cond
        c_in_pred = 1.0 - tau_v  # [B]
        while c_in_pred.ndim < z_noisy.ndim:
            c_in_pred = c_in_pred.unsqueeze(-1)

        z_cond_scaled = z_cond * c_in_cond
        z_noisy_scaled = z_noisy * c_in_pred

        # Concatenate conditioning and prediction frames along temporal dim
        hidden_states = torch.cat([z_cond_scaled, z_noisy_scaled], dim=2)  # [B, C, T_total, H, W]

        # Build condition mask: 1 for conditioning frames, 0 for prediction frames
        if condition_mask is None:
            condition_mask = torch.zeros(B, 1, T_total, H, W, device=device, dtype=z_noisy.dtype)
            condition_mask[:, :, :T_cond] = 1.0

        # Build padding mask (zeros)
        padding_mask = torch.zeros(1, 1, H, W, device=device, dtype=z_noisy.dtype)

        # Clear hidden states cache before forward
        self.clear_hidden_states_cache()

        # Forward through transformer
        transformer_dtype = self.dtype
        output = self.transformer(
            hidden_states=hidden_states.to(transformer_dtype),
            timestep=timestep.to(transformer_dtype),
            encoder_hidden_states=encoder_hidden_states.to(transformer_dtype),
            condition_mask=condition_mask.to(transformer_dtype),
            padding_mask=padding_mask.to(transformer_dtype),
            return_dict=False,
        )[0]

        # Apply Cosmos output parameterization
        # final_output = c_skip * input + c_out * raw_output
        # c_skip = 1 - t, c_out = -t
        # For conditioning frames: nearly identity (c_skip≈1, c_out≈0)
        # For prediction frames: c_skip = 1-tau_v, c_out = -tau_v

        # Build per-frame c_skip and c_out
        full_output = torch.zeros_like(hidden_states, dtype=torch.float32)

        # Conditioning frames
        c_skip_cond = 1.0 - t_cond
        c_out_cond = -t_cond
        full_output[:, :, :T_cond] = c_skip_cond * hidden_states[:, :, :T_cond].float() + c_out_cond * output[:, :, :T_cond].float()

        # Prediction frames
        c_skip_pred = 1.0 - tau_v  # [B]
        c_out_pred = -tau_v  # [B]
        while c_skip_pred.ndim < z_noisy.ndim:
            c_skip_pred = c_skip_pred.unsqueeze(-1)
            c_out_pred = c_out_pred.unsqueeze(-1)

        # The unscaled latents for pred frames are needed for c_skip
        # hidden_states already contains scaled versions, but we need original z_noisy for c_skip
        # Actually, looking at the pipeline code: c_skip * latents (unscaled) + c_out * output
        # The pipeline uses the original unscaled latents for c_skip, not the scaled input
        full_output[:, :, T_cond:] = c_skip_pred * z_noisy.float() + c_out_pred * output[:, :, T_cond:].float()

        # For conditioning frames, replace with original conditioning latents
        full_output[:, :, :T_cond] = z_cond.float()

        # The output for prediction frames is the denoised prediction
        # In flow matching terms, the velocity = eps - x_0
        # Since raw output of the network IS the velocity (verified from source),
        # we return the raw output for prediction frames for loss computation
        return output.float(), full_output

    def pool_hidden_states(
        self, hidden_states: torch.Tensor, num_latent_frames: int, mode: str = "mean"
    ) -> torch.Tensor:
        """Reduce hidden states for the action decoder's cross-attention.

        The hidden states from the transformer block have shape [B, T*H'*W', hidden_dim]
        where H'=H_lat/patch_h, W'=W_lat/patch_w.

        Args:
            hidden_states: [B, T*H'*W', hidden_dim] from the hook.
            num_latent_frames: Number of latent time frames T.
            mode: "mean" pools spatially to [B, T, D] (5 tokens).
                  "none" passes all tokens [B, T*H'*W', D] (~6000 tokens).

        Returns:
            Hidden states for cross-attention. Shape depends on mode.
        """
        if mode == "none":
            return hidden_states  # [B, T*H'*W', D]

        B, THW, D = hidden_states.shape
        HW = THW // num_latent_frames
        hidden_states = hidden_states.view(B, num_latent_frames, HW, D)
        pooled = hidden_states.mean(dim=2)  # [B, T, D]
        return pooled

    def freeze_for_stage2(self):
        """Freeze all backbone parameters for stage 2 training.

        After stage 1, the LoRA weights are merged or kept frozen,
        and only the action decoder is trained.
        """
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def save_lora(self, path: str):
        """Save LoRA weights."""
        self.transformer.save_pretrained(path)

    def load_lora(self, path: str, is_trainable: bool = False):
        """Load LoRA weights."""
        from peft import PeftModel
        
        # Unwrap DDP if needed
        transformer = self.transformer.module if hasattr(self.transformer, 'module') else self.transformer
        
        # If transformer is already a PeftModel, load adapter
        if hasattr(transformer, 'load_adapter'):
            transformer.load_adapter(path, adapter_name="default", is_trainable=is_trainable)
        else:
            self.transformer = PeftModel.from_pretrained(
                self.transformer, path, is_trainable=is_trainable
            )
            self._register_hooks()  # Re-register hooks after wrapping

    def offload_vae_and_text_encoder(self, target_device: str = "cpu"):
        """Move VAE and text encoder to CPU to save GPU memory during training."""
        self.vae.to(target_device)
        if self.text_encoder is not None:
            self.text_encoder.to(target_device)

    def move_vae_to(self, device: str):
        """Move VAE to specified device."""
        self.vae.to(device)

    def move_text_encoder_to(self, device: str):
        """Move text encoder to specified device."""
        if self.text_encoder is not None:
            self.text_encoder.to(device)
