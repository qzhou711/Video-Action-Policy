"""Action Decoder DiT for mimic-video.

Lightweight DiT that takes video backbone hidden states and noisy actions,
and predicts the velocity field for action denoising via flow matching.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal embedding for scalar timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scalar timesteps.

        Args:
            x: Tensor of shape [B], values in [0, 1].

        Returns:
            Tensor of shape [B, dim].
        """
        device = x.device
        input_dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        # Cast back to input dtype (computation stays in fp32 for precision,
        # but output must match the dtype of downstream Linear weights).
        return emb.to(input_dtype)


class BilinearAffineTimestepEmbedding(nn.Module):
    """Bilinear-affine timestep embedding combining tau_v and tau_a.

    Projects both timesteps independently via sinusoidal + MLP,
    then combines via element-wise product (bilinear) and affine transform.
    """

    def __init__(self, hidden_dim: int = 512, sinusoidal_dim: int = 256):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim

        self.sin_embed_v = SinusoidalPositionalEmbedding(sinusoidal_dim)
        self.sin_embed_a = SinusoidalPositionalEmbedding(sinusoidal_dim)

        self.mlp_v = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_a = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Affine after bilinear product
        self.affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tau_v: torch.Tensor, tau_a: torch.Tensor) -> torch.Tensor:
        """Compute bilinear-affine timestep embedding.

        Args:
            tau_v: Video denoising timestep, shape [B].
            tau_a: Action denoising timestep, shape [B].

        Returns:
            Combined embedding, shape [B, hidden_dim].
        """
        h_v = self.mlp_v(self.sin_embed_v(tau_v))  # [B, hidden_dim]
        h_a = self.mlp_a(self.sin_embed_a(tau_a))  # [B, hidden_dim]

        # Bilinear: element-wise product
        h = h_v * h_a  # [B, hidden_dim]

        # Affine
        h = self.affine(h)  # [B, hidden_dim]
        return h


class AdaLNZeroModulation(nn.Module):
    """Adaptive Layer Normalization with Zero initialization (AdaLN-Zero).

    Computes scale, shift, and gate parameters from a conditioning vector.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Produce: shift, scale, gate (3 * hidden_dim)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        # Zero-initialize the projection so gates start at zero
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple:
        """Apply adaptive layer norm.

        Args:
            x: Input tensor, shape [B, L, D].
            cond: Conditioning vector, shape [B, D].

        Returns:
            Tuple of (normalized_x, gate) where gate is shape [B, 1, D].
        """
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        # shift, scale, gate: [B, D]
        x_norm = self.norm(x)
        x_mod = x_norm * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x_mod, gate.unsqueeze(1)


class ActionDecoderBlock(nn.Module):
    """Single transformer block for the action decoder.

    Each block consists of:
    1. AdaLN-Zero + Cross-attention to video hidden states
    2. AdaLN-Zero + Self-attention over action sequence
    3. AdaLN-Zero + 2-layer MLP (GELU)

    All with residual connections and gate modulation.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        backbone_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        # 1. Cross-attention to video hidden states
        self.adaln_cross = AdaLNZeroModulation(hidden_dim)
        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_k = nn.Linear(backbone_hidden_dim, hidden_dim)
        self.cross_attn_v = nn.Linear(backbone_hidden_dim, hidden_dim)
        self.cross_attn_q_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.cross_attn_k_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.cross_attn_out = nn.Linear(hidden_dim, hidden_dim)

        # 2. Self-attention over action sequence
        self.adaln_self = AdaLNZeroModulation(hidden_dim)
        self.self_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn_q_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.self_attn_k_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.self_attn_out = nn.Linear(hidden_dim, hidden_dim)

        # 3. MLP
        self.adaln_mlp = AdaLNZeroModulation(hidden_dim)
        mlp_hidden = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        use_rope: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention.

        Args:
            q: [B, Lq, D], k: [B, Lk, D], v: [B, Lk, D]

        Returns:
            [B, Lq, D]
        """
        B, Lq, D = q.shape
        Lk = k.shape[1]
        head_dim = D // self.num_heads

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        if use_rope:
            q = self._apply_rope_1d(q)
            k = self._apply_rope_1d(k)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        return out

    @staticmethod
    def _apply_rope_1d(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
        """Apply 1D RoPE to tensor shaped [B, H, L, D]."""
        _, _, seq_len, head_dim = x.shape
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

        device = x.device
        dtype = x.dtype

        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        freqs = torch.outer(pos, inv_freq)  # [L, D/2]
        cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
        sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        h_video: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of action decoder block.

        Args:
            x: Action sequence, shape [B, L, hidden_dim].
            h_video: Video hidden states, shape [B, T_lat, backbone_hidden_dim].
            cond: Timestep conditioning, shape [B, hidden_dim].

        Returns:
            Updated action sequence, shape [B, L, hidden_dim].
        """
        # 1. Self-attention over action sequence
        x_mod, gate = self.adaln_self(x, cond)
        q = self.self_attn_q_norm(self.self_attn_q(x_mod))
        k = self.self_attn_k_norm(self.self_attn_k(x_mod))
        v = self.self_attn_v(x_mod)
        attn_out = self._attention(q, k, v, use_rope=True)
        attn_out = self.self_attn_out(attn_out)
        x = x + gate * attn_out

        # 2. Cross-attention to video hidden states
        x_mod, gate = self.adaln_cross(x, cond)
        q = self.cross_attn_q_norm(self.cross_attn_q(x_mod))
        k = self.cross_attn_k_norm(self.cross_attn_k(h_video))
        v = self.cross_attn_v(h_video)
        attn_out = self._attention(q, k, v)
        attn_out = self.cross_attn_out(attn_out)
        x = x + gate * attn_out

        # 3. MLP
        x_mod, gate = self.adaln_mlp(x, cond)
        mlp_out = self.mlp(x_mod)
        x = x + gate * mlp_out

        return x


class ActionDecoderDiT(nn.Module):
    """Action Decoder DiT (Diffusion Transformer) for mimic-video.

    A lightweight transformer that takes:
    - Noisy actions (flow matching)
    - Proprioception
    - Video backbone hidden states (from layer k)
    - Both video and action timesteps

    And predicts the velocity field for action denoising.

    Architecture (~60M params):
    - Input MLPs for proprio and actions
    - Learned mask token for proprioception masking
    - Sequence: [proprio_token, action_1, ..., action_T]
    - 8 transformer blocks with cross-attention to video hidden states
    - Zero-initialized output head
    """

    def __init__(
        self,
        action_dim: int = 16,
        proprio_dim: int = 16,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        backbone_hidden_dim: int = 2048,
        action_chunk_size: int = 16,
        proprio_mask_prob: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.action_chunk_size = action_chunk_size
        self.proprio_mask_prob = proprio_mask_prob
        self.backbone_hidden_dim = backbone_hidden_dim

        # Input projections
        self.action_input_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proprio_input_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Optional direct text-to-video-context path:
        # compress T5 tokens to one token in backbone_hidden_dim and prepend to h_video.
        self.text_context_proj = nn.Sequential(
            nn.Linear(text_dim, backbone_hidden_dim),
            nn.GELU(),
            nn.Linear(backbone_hidden_dim, backbone_hidden_dim),
        )

        # Mask token for proprioception masking (buffer, not a trainable parameter).
        # Registered as a buffer so DDP does not track it for gradient sync —
        # avoids "Expected to have finished reduction" errors when mask_prob=0.
        self.register_buffer('proprio_mask_token', torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional embeddings: 1 (proprio) + action_chunk_size (actions)
        seq_len = 1 + action_chunk_size
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Bilinear-affine timestep embedding
        self.timestep_embed = BilinearAffineTimestepEmbedding(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ActionDecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                backbone_hidden_dim=backbone_hidden_dim,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection (zero-initialized for flow matching)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        proprio: torch.Tensor,
        h_video: torch.Tensor,
        tau_a: torch.Tensor,
        tau_v: torch.Tensor,
        t5_embedding: torch.Tensor | None = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the action decoder.

        Args:
            noisy_actions: Noisy action chunk, shape [B, T_action, action_dim].
            proprio: Proprioception, shape [B, proprio_dim].
            h_video: Pooled video hidden states, shape [B, T_lat, backbone_hidden_dim].
            t5_embedding: Optional T5 tokens [B, seq_len, text_dim].
            tau_a: Action denoising timestep, shape [B].
            tau_v: Video denoising timestep, shape [B].
            training: Whether in training mode (enables proprio masking).

        Returns:
            Predicted velocity field, shape [B, T_action, action_dim].
        """
        B = noisy_actions.shape[0]

        # Project actions: [B, T_action, hidden_dim]
        action_tokens = self.action_input_mlp(noisy_actions)

        # Project proprioception: [B, 1, hidden_dim]
        proprio_tokens = self.proprio_input_mlp(proprio.unsqueeze(1))

        # Random proprioception masking during training
        if training and self.proprio_mask_prob > 0:
            mask = torch.rand(B, 1, 1, device=proprio.device) < self.proprio_mask_prob
            proprio_tokens = torch.where(
                mask, self.proprio_mask_token.expand(B, -1, -1), proprio_tokens
            )

        # Build sequence: [proprio, action_1, ..., action_T]
        # Shape: [B, 1 + T_action, hidden_dim]
        x = torch.cat([proprio_tokens, action_tokens], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed[:, : x.shape[1]]

        # Compute timestep conditioning
        cond = self.timestep_embed(tau_v, tau_a)  # [B, hidden_dim]

        # Optional direct text conditioning for action decoder:
        # one compressed token prepended to video context.
        if t5_embedding is not None:
            text_tokens = t5_embedding.to(h_video.dtype)
            if text_tokens.shape[-1] != self.text_context_proj[0].in_features:
                raise ValueError(
                    f"t5_embedding dim mismatch: got {text_tokens.shape[-1]}, "
                    f"expected {self.text_context_proj[0].in_features}"
                )
            text_token = text_tokens.mean(dim=1, keepdim=True)  # [B, 1, text_dim]
            text_token = self.text_context_proj(text_token)  # [B, 1, backbone_hidden_dim]
            h_video = torch.cat([text_token, h_video], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, h_video, cond)

        # Final norm
        x = self.final_norm(x)

        # Output projection - only for action tokens (skip proprio token)
        action_out = x[:, 1:]  # [B, T_action, hidden_dim]
        velocity = self.output_proj(action_out)  # [B, T_action, action_dim]

        return velocity
