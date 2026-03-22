"""Image transforms for mimic-video: camera concat and normalization."""

import torch
import torch.nn.functional as F
from typing import List


def concat_cameras(
    images: List[torch.Tensor],
    target_height: int = 480,
    target_width: int = 640,
) -> torch.Tensor:
    """Stack camera images into a grid and resize back to target resolution.

    Supports 2 cameras (side-by-side), 3 cameras (top-left, top-right, bottom-center),
    or 4 cameras (2x2 grid).

    Args:
        images: List of 3 or 4 tensors, each [C, H, W] or [T, C, H, W].
        target_height: Output height after resize.
        target_width: Output width after resize.

    Returns:
        Tensor of shape [C, target_height, target_width] or [T, C, target_height, target_width].
    """
    n = len(images)
    assert n in (2, 3, 4), f"Expected 2, 3 or 4 cameras, got {n}"

    has_time = images[0].ndim == 4

    if n == 4:
        if has_time:
            top = torch.cat([images[0], images[1]], dim=-1)
            bottom = torch.cat([images[2], images[3]], dim=-1)
            grid = torch.cat([top, bottom], dim=-2)
        else:
            top = torch.cat([images[0], images[1]], dim=-1)
            bottom = torch.cat([images[2], images[3]], dim=-1)
            grid = torch.cat([top, bottom], dim=-2)
    elif n == 3:
        # 3 cameras: top row has 2, bottom row has 1 centered with black padding
        if has_time:
            T, C, H, W = images[0].shape
            top = torch.cat([images[0], images[1]], dim=-1)  # [T, C, H, 2W]
            pad_left = torch.zeros(T, C, H, W // 2, device=images[2].device, dtype=images[2].dtype)
            pad_right = torch.zeros(T, C, H, W - W // 2, device=images[2].device, dtype=images[2].dtype)
            bottom = torch.cat([pad_left, images[2], pad_right], dim=-1)  # [T, C, H, 2W]
            grid = torch.cat([top, bottom], dim=-2)  # [T, C, 2H, 2W]
        else:
            C, H, W = images[0].shape
            top = torch.cat([images[0], images[1]], dim=-1)
            pad_left = torch.zeros(C, H, W // 2, device=images[2].device, dtype=images[2].dtype)
            pad_right = torch.zeros(C, H, W - W // 2, device=images[2].device, dtype=images[2].dtype)
            bottom = torch.cat([pad_left, images[2], pad_right], dim=-1)
            grid = torch.cat([top, bottom], dim=-2)
    else:
        # 2 cameras: side-by-side horizontally
        grid = torch.cat([images[0], images[1]], dim=-1)  # [..., H, 2W]

    # Resize to target
    if has_time:
        T, C = grid.shape[:2]
        grid_flat = grid.reshape(T * C, grid.shape[2], grid.shape[3]).unsqueeze(1)
        grid_resized = F.interpolate(
            grid_flat, size=(target_height, target_width), mode="bilinear", align_corners=False
        )
        return grid_resized.reshape(T, C, target_height, target_width)
    else:
        grid = grid.unsqueeze(0)
        grid = F.interpolate(
            grid, size=(target_height, target_width), mode="bilinear", align_corners=False
        )
        return grid.squeeze(0)


# Keep old name as alias for backward compat
concat_cameras_2x2 = concat_cameras


def normalize_to_neg1_pos1(images: torch.Tensor) -> torch.Tensor:
    """Normalize images from [0, 1] or [0, 255] to [-1, 1].

    Args:
        images: Tensor of pixel values.

    Returns:
        Normalized tensor in [-1, 1].
    """
    if images.max() > 1.0:
        images = images.float() / 255.0
    return images * 2.0 - 1.0


def denormalize_from_neg1_pos1(images: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1].

    Args:
        images: Tensor in [-1, 1].

    Returns:
        Tensor in [0, 1].
    """
    return (images + 1.0) / 2.0
