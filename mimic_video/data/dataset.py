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
        action_norm_type: str = "min-max",
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
        self.action_norm_type = action_norm_type
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
            self.action_mean = action_stats.get("mean", None)
            self.action_std = action_stats.get("std", None)
            self.action_min = action_stats.get("min", None)
            self.action_max = action_stats.get("max", None)
        else:
            self.action_mean = None
            self.action_std = None
            self.action_min = None
            self.action_max = None

        self.t5_embedding = None       # single-task fallback [1, seq_len, dim]
        self.t5_embeddings = None       # multi-task dict {task_index: [1, seq_len, dim]}

        # Try multi-task first, then single-task
        if precomputed_dir:
            multi_path = os.path.join(precomputed_dir, "t5_embeddings.pt")
            single_path = os.path.join(precomputed_dir, "t5_embedding.pt")
            if os.path.exists(multi_path):
                self.t5_embeddings = torch.load(multi_path, map_location="cpu", weights_only=True)
            elif os.path.exists(single_path):
                self.t5_embedding = torch.load(single_path, map_location="cpu", weights_only=True)

        # Build episode_index → task_index mapping for multi-task datasets
        self.episode_to_task = {}
        self._build_episode_task_map()

    def _build_episode_task_map(self):
        """Build a mapping from episode_index to task_index using dataset metadata."""
        try:
            hf = self.lerobot_dataset.hf_dataset
            if "task_index" in hf.column_names and "episode_index" in hf.column_names:
                # Sample the first row of each episode to get its task_index
                episodes = self.lerobot_dataset.meta.episodes
                for ep in episodes:
                    ep_idx = ep["episode_index"]
                    first_frame = ep["dataset_from_index"]
                    row = hf[first_frame]
                    t_idx = row["task_index"]
                    if isinstance(t_idx, torch.Tensor):
                        t_idx = t_idx.item()
                    self.episode_to_task[ep_idx] = int(t_idx)
        except Exception:
            pass  # Not a multi-task dataset, no mapping needed

    def _get_task_index_for_global_idx(self, global_idx: int) -> int:
        """Get the task_index for a given global frame index."""
        if not self.episode_to_task:
            return 0
        row = self.lerobot_dataset.hf_dataset[global_idx]
        t_idx = row.get("task_index", 0)
        if isinstance(t_idx, torch.Tensor):
            t_idx = t_idx.item()
        return int(t_idx)

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
        device = actions.device
        if self.action_norm_type == "min-max":
            if self.action_min is None or self.action_max is None:
                return actions
            # Normalize to [-1, 1]
            # First map to [0, 1]
            a_min = self.action_min.to(device)
            a_max = self.action_max.to(device)
            scaled = (actions - a_min) / (a_max - a_min + 1e-4) # 1e-4 prevents div by 0
            # Map to [-1, 1]
            return scaled * 2.0 - 1.0
        elif self.action_norm_type == "mean-std":
            if self.action_mean is None or self.action_std is None:
                return actions
            return (actions - self.action_mean.to(device)) / self.action_std.to(device)
        else:
            return actions

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        device = actions.device
        if self.action_norm_type == "min-max":
            if self.action_min is None or self.action_max is None:
                return actions
            a_min = self.action_min.to(device)
            a_max = self.action_max.to(device)
            # Reverse map from [-1, 1] to [0, 1]
            unscaled = (actions + 1.0) / 2.0
            # Map to original range
            return unscaled * (a_max - a_min + 1e-4) + a_min
        elif self.action_norm_type == "mean-std":
            if self.action_mean is None or self.action_std is None:
                return actions
            return actions * self.action_std.to(device) + self.action_mean.to(device)
        else:
            return actions

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
            "std": all_actions.std(dim=0).clamp(min=1e-4),
            "min": all_actions.min(dim=0)[0],
            "max": all_actions.max(dim=0)[0],
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

        # Multi-task: return per-sample T5 embedding based on task_index
        if self.t5_embeddings is not None:
            task_idx = self._get_task_index_for_global_idx(global_idx)
            if task_idx in self.t5_embeddings:
                result["t5_embedding"] = self.t5_embeddings[task_idx]
            else:
                # Fallback to first available embedding
                first_key = sorted(self.t5_embeddings.keys())[0]
                result["t5_embedding"] = self.t5_embeddings[first_key]
        elif self.t5_embedding is not None:
            result["t5_embedding"] = self.t5_embedding

        return result
