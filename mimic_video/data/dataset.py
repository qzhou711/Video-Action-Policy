"""MimicVideoDataset wrapping LeRobot for the mimic-video pipeline."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from mimic_video.data.transforms import concat_cameras, normalize_to_neg1_pos1


class MimicVideoDataset(Dataset):
    """Dataset for mimic-video training.

    Uses delta_timestamps only for camera video frames (which need multi-frame decode).
    State and action data are loaded by manually indexing consecutive frames,
    avoiding the slow delta_timestamps lookup for tabular data.
    """

    def __init__(
        self,
        repo_id: str,
        camera_names: list,
        state_keys: List[str],
        action_keys: List[str],
        num_pixel_frames: int = 17,
        action_chunk_size: int = 16,
        action_dim: int = 16,
        proprio_dim: int = 16,
        target_height: int = 480,
        target_width: int = 640,
        episode_indices: Optional[list] = None,
        precomputed_dir: Optional[str] = None,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
        fps: int = 10,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.camera_names = camera_names
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.num_pixel_frames = num_pixel_frames
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.target_height = target_height
        self.target_width = target_width
        self.precomputed_dir = precomputed_dir
        self.fps = fps

        # Only use delta_timestamps for cameras (video decode needs it)
        delta_timestamps = {}
        frame_deltas = [i / fps for i in range(num_pixel_frames)]
        for cam_name in camera_names:
            delta_timestamps[cam_name] = frame_deltas

        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
        )

        self._build_valid_indices(episode_indices)

        if action_stats is not None:
            self.action_mean = action_stats["mean"]
            self.action_std = action_stats["std"]
        else:
            self.action_mean = None
            self.action_std = None

        self.t5_embedding = None
        if precomputed_dir and os.path.exists(os.path.join(precomputed_dir, "t5_embedding.pt")):
            self.t5_embedding = torch.load(
                os.path.join(precomputed_dir, "t5_embedding.pt"), map_location="cpu", weights_only=True
            )

    def _build_valid_indices(self, episode_indices: Optional[list] = None):
        self.valid_indices = []
        episodes = self.lerobot_dataset.meta.episodes

        for i in range(len(episodes)):
            ep = episodes[i]
            ep_idx = ep["episode_index"]
            if episode_indices is not None and ep_idx not in episode_indices:
                continue

            ep_start = ep["dataset_from_index"]
            ep_end = ep["dataset_to_index"]
            ep_len = ep_end - ep_start

            min_frames_needed = self.num_pixel_frames + self.action_chunk_size
            if ep_len < min_frames_needed:
                continue

            for frame_offset in range(ep_len - min_frames_needed + 1):
                global_idx = ep_start + frame_offset
                self.valid_indices.append(global_idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None:
            return actions
        device = actions.device
        return (actions - self.action_mean.to(device)) / self.action_std.to(device)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None:
            return actions
        device = actions.device
        return actions * self.action_std.to(device) + self.action_mean.to(device)

    def _get_state_action(self, global_idx: int):
        """Load state and action chunk by directly indexing consecutive frames."""
        # State at current frame
        row = self.lerobot_dataset.hf_dataset[global_idx]
        proprio_parts = []
        for sk in self.state_keys:
            val = row[sk]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            proprio_parts.append(val.float().flatten())
        proprio = torch.cat(proprio_parts)[:self.proprio_dim]

        # Action chunk: gather from consecutive frames
        action_rows = []
        for offset in range(1, self.action_chunk_size + 1):
            a_row = self.lerobot_dataset.hf_dataset[global_idx + offset]
            parts = []
            for ak in self.action_keys:
                val = a_row[ak]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                parts.append(val.float().flatten())
            action_rows.append(torch.cat(parts))
        actions = torch.stack(action_rows)[:, :self.action_dim]

        return proprio, actions

    def compute_action_stats(self, max_samples: int = 10000) -> Dict[str, torch.Tensor]:
        """Compute mean and standard deviation of actions from the dataset."""
        num_samples = min(len(self.valid_indices), max_samples)
        indices = np.random.choice(len(self.valid_indices), num_samples, replace=False)
        
        all_actions = []
        for idx in indices:
            global_idx = self.valid_indices[idx]
            _, actions = self._get_state_action(global_idx)
            all_actions.append(actions)
            
        all_actions = torch.cat(all_actions, dim=0)
        
        return {
            "mean": all_actions.mean(dim=0),
            "std": all_actions.std(dim=0).clamp(min=1e-4)
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = self.valid_indices[idx]

        # Get video frames via lerobot (uses torchcodec for fast video decode)
        sample = self.lerobot_dataset[global_idx]

        camera_frames = []
        for cam_name in self.camera_names:
            frames = sample[cam_name]
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            camera_frames.append(frames)

        video = concat_cameras(camera_frames, self.target_height, self.target_width)
        video = normalize_to_neg1_pos1(video)

        # Get state/action by direct parquet indexing (fast)
        proprio, actions = self._get_state_action(global_idx)
        actions = self.normalize_actions(actions)

        result = {
            "video": video,
            "proprio": proprio,
            "actions": actions,
        }

        if self.t5_embedding is not None:
            result["t5_embedding"] = self.t5_embedding

        return result
