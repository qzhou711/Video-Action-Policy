"""Flow Matching scheduler for both video and action denoising."""

import torch
import torch.nn.functional as F
from typing import Callable, Optional


class FlowMatchingScheduler:
    """Flow Matching scheduler with support for video and action noise distributions.

    Implements:
    - Linear interpolation between data and noise
    - Velocity field targets (eps - x_0)
    - Logit-normal tau sampling for video
    - Pi0-style power-law tau sampling for actions
    - Euler ODE integration for sampling
    """

    def __init__(self):
        pass

    @staticmethod
    def interpolate(x_0: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Compute flow matching interpolation: x_tau = (1 - tau) * x_0 + tau * eps.

        Args:
            x_0: Clean data, shape [B, ...].
            eps: Noise, shape [B, ...].
            tau: Timestep in [0, 1], shape [B] or broadcastable.

        Returns:
            Interpolated sample x_tau, same shape as x_0.
        """
        # Reshape tau for broadcasting
        while tau.ndim < x_0.ndim:
            tau = tau.unsqueeze(-1)
        return (1.0 - tau) * x_0 + tau * eps

    @staticmethod
    def velocity_target(x_0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Compute the velocity field target: v = eps - x_0.

        In flow matching, the ODE is dx/dt = v(x_t, t), and the optimal
        velocity field for the linear interpolation path is v = eps - x_0.

        Args:
            x_0: Clean data, shape [B, ...].
            eps: Noise, shape [B, ...].

        Returns:
            Velocity target, same shape as x_0.
        """
        return eps - x_0

    @staticmethod
    def sample_tau_video(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from logit-normal distribution for video denoising.

        tau = sigmoid(N(0, 1)), which concentrates samples around 0.5
        and provides good coverage of the full [0, 1] interval.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.

        Returns:
            Tensor of shape [B] with values in (0, 1).
        """
        z = torch.randn(batch_size, device=device)
        return torch.sigmoid(z)

    @staticmethod
    def sample_tau_action(
        batch_size: int,
        device: torch.device,
        power: float = 0.999,
    ) -> torch.Tensor:
        """Sample timesteps with pi0-style power-law distribution for actions.

        tau = U^(1/power) where U ~ Uniform(0, 1).
        With power close to 1 (e.g., 0.999), this biases toward tau=1 (noisy),
        making the model focus on the difficult early denoising steps.

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.
            power: Power parameter (default 0.999).

        Returns:
            Tensor of shape [B] with values in (0, 1).
        """
        u = torch.rand(batch_size, device=device).clamp(min=1e-6)
        return u.pow(1.0 / power)

    @staticmethod
    def ode_solve_euler(
        model_fn: Callable,
        x_init: torch.Tensor,
        num_steps: int,
        tau_start: float = 1.0,
        tau_end: float = 0.0,
    ) -> torch.Tensor:
        """Euler integration of the ODE dx/dt = v(x_t, t) from tau_start to tau_end.

        Args:
            model_fn: Callable that takes (x_t, tau) and returns velocity v(x_t, tau).
                tau is a scalar float.
            x_init: Initial state at tau_start, shape [B, ...].
            num_steps: Number of Euler steps.
            tau_start: Starting timestep (typically 1.0 = pure noise).
            tau_end: Ending timestep (typically 0.0 = clean data).

        Returns:
            Final state at tau_end, same shape as x_init.
        """
        dt = (tau_end - tau_start) / num_steps
        x_t = x_init
        tau = tau_start

        for _ in range(num_steps):
            v = model_fn(x_t, tau)
            x_t = x_t + v * dt
            tau = tau + dt

        return x_t

    @staticmethod
    def compute_loss(
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and target velocity fields.

        Args:
            pred_velocity: Predicted velocity, shape [B, ...].
            target_velocity: Target velocity, shape [B, ...].
            mask: Optional mask, shape [B, ...]. Loss is computed only where mask > 0.

        Returns:
            Scalar MSE loss.
        """
        diff = pred_velocity - target_velocity
        if mask is not None:
            diff = diff * mask
            return (diff ** 2).sum() / mask.sum().clamp(min=1.0)
        return F.mse_loss(pred_velocity, target_velocity)
