"""WebSocket server for mimic-video model inference.

Loads the trained model (Cosmos backbone + LoRA + Action Decoder) and serves
action predictions via WebSocket. Designed to work with a separate LIBERO
simulation client running in a different conda environment.

Usage:
    conda activate mimic
    python scripts/eval_server.py \
        --stage1_checkpoint checkpoints/stage1/final \
        --stage2_checkpoint checkpoints/stage2/final

The server expects JSON messages with:
    {
        "image": [[H, W, 3], [H, W, 3]],    # agentview + wrist images (uint8)
        "state": [8 floats],                   # proprio state
        "prompt": "task description string",   # (used on first call only)
        "reset": true/false                    # whether to reset frame buffer
    }

And returns JSON:
    [[7 floats], [7 floats], ...]  # action chunk (action_chunk_size × action_dim)
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import json
import asyncio
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig
from mimic_video.models.video_backbone import CosmosVideoBackbone
from mimic_video.models.action_decoder import ActionDecoderDiT
from mimic_video.inference.policy import MimicVideoPolicy
from mimic_video.data.transforms import normalize_to_neg1_pos1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class FrameBuffer:
    """Maintains a sliding window of video frames for the model.

    The model expects 17 frames as input. This buffer accumulates frames
    from LIBERO step-by-step and provides the full window when needed.
    """

    def __init__(self, num_frames: int = 17, height: int = 256, width: int = 256):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frames = []  # List of [C, H, W] tensors in [-1, 1]

    def reset(self):
        """Clear frame buffer."""
        self.frames = []

    def add_frame(self, agentview: np.ndarray, wrist: np.ndarray):
        """Add a pair of camera images to the buffer.

        Args:
            agentview: [H, W, 3] uint8 numpy array (main camera).
            wrist: [H, W, 3] uint8 numpy array (wrist camera).
        """
        # Convert to tensor [C, H, W] float32 [0, 1]
        agent_t = torch.from_numpy(agentview).permute(2, 0, 1).float() / 255.0
        wrist_t = torch.from_numpy(wrist).permute(2, 0, 1).float() / 255.0

        # Concat cameras side by side → [C, H, 2W]
        concat = torch.cat([agent_t, wrist_t], dim=-1)

        # Resize to target → [C, H, W]
        concat = F.interpolate(
            concat.unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Normalize to [-1, 1]
        concat = concat * 2.0 - 1.0

        self.frames.append(concat)

        # Keep at most num_frames
        if len(self.frames) > self.num_frames:
            self.frames = self.frames[-self.num_frames:]

    def get_video(self) -> torch.Tensor:
        """Get the full video tensor for model input.

        Returns:
            [1, T, C, H, W] tensor in [-1, 1] with T = num_frames.
        """
        if len(self.frames) == 0:
            raise ValueError("Frame buffer is empty. Add frames first.")

        # Pad if not enough frames (repeat first frame)
        frames = list(self.frames)
        while len(frames) < self.num_frames:
            frames.insert(0, frames[0].clone())

        # Stack: [T, C, H, W] → [1, T, C, H, W]
        video = torch.stack(frames, dim=0).unsqueeze(0)
        return video


def load_model(args) -> MimicVideoPolicy:
    """Load the trained mimic-video model."""
    data_config = DataConfig()
    model_config = ModelConfig()

    device = args.device
    log.info("Loading Cosmos video backbone with Stage 1 LoRA...")
    backbone = CosmosVideoBackbone(
        model_id=model_config.cosmos_model_id,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_target_modules=model_config.lora_target_modules,
        hidden_state_layer=model_config.hidden_state_layer,
        dtype=torch.bfloat16,
        device=device,
    )
    backbone.load_lora(args.stage1_checkpoint)
    backbone.transformer.to(device)
    backbone.offload_vae_and_text_encoder("cpu")

    log.info("Loading action decoder...")
    action_decoder = ActionDecoderDiT(
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        hidden_dim=model_config.decoder_hidden_dim,
        num_layers=model_config.decoder_num_layers,
        num_heads=model_config.decoder_num_heads,
        mlp_ratio=model_config.decoder_mlp_ratio,
        backbone_hidden_dim=backbone.hidden_dim,
        action_chunk_size=data_config.action_chunk_size,
        proprio_mask_prob=0.0,  # No masking at inference
    )

    decoder_path = os.path.join(args.stage2_checkpoint, "action_decoder.pt")
    state_dict = torch.load(decoder_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    action_decoder.load_state_dict(state_dict)
    action_decoder.to(device)
    action_decoder.eval()

    # Load action stats
    precomputed_dir = args.precomputed_dir
    action_stats = None
    stats_path = os.path.join(precomputed_dir, "action_stats.pt")
    if os.path.exists(stats_path):
        action_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        log.info("Loaded action stats for denormalization.")
    else:
        log.warning("No action_stats.pt found — actions will NOT be denormalized!")

    # Load T5 embedding
    t5_embedding = None
    t5_path = os.path.join(precomputed_dir, "t5_embedding.pt")
    if os.path.exists(t5_path):
        t5_embedding = torch.load(t5_path, map_location="cpu", weights_only=True)
        log.info("Loaded precomputed T5 embedding.")

    log.info("Creating inference policy...")
    policy = MimicVideoPolicy(
        backbone=backbone,
        action_decoder=action_decoder,
        action_stats=action_stats,
        t5_embedding=t5_embedding,
        tau_v=args.tau_v,
        num_action_denoise_steps=args.num_action_steps,
        num_cond_latent_frames=data_config.num_cond_latent_frames,
        num_pred_latent_frames=data_config.num_pred_latent_frames,
        num_pixel_frames=data_config.num_pixel_frames,
        camera_names=data_config.camera_names,
        target_height=data_config.camera_height,
        target_width=data_config.camera_width,
        device=device,
    )
    policy.eval()

    return policy, data_config


async def handle_client(ws, policy, data_config, frame_buffer):
    """Handle WebSocket connection from LIBERO client.
    
    Message types:
        {"reset": true}                         → reset frame buffer
        {"add_frame": true, "image": [...]}     → add frame to buffer (no inference)
        {"query": true, "image": [...], "state": [...]} → add frame + run inference
    """
    log.info("Client connected.")

    async for message in ws:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            log.error("Invalid JSON received.")
            await ws.send(json.dumps({"error": "invalid JSON"}))
            continue

        # Reset signal
        if data.get("reset", False):
            frame_buffer.reset()
            log.info("Frame buffer reset.")
            await ws.send(json.dumps({"status": "reset"}))
            continue

        # Parse images (common to add_frame and query)
        images = data.get("image")
        if images:
            agentview = np.array(images[0], dtype=np.uint8)
            wrist = np.array(images[1], dtype=np.uint8)
            frame_buffer.add_frame(agentview, wrist)

        # Add frame only (during action execution, no inference needed)
        if data.get("add_frame", False) and not data.get("query", False):
            await ws.send(json.dumps({"status": "frame_added", "buffer_size": len(frame_buffer.frames)}))
            continue

        # Query: run inference
        if data.get("query", False):
            state = np.array(data["state"], dtype=np.float32)
            proprio = torch.from_numpy(state).unsqueeze(0)  # [1, proprio_dim]

            # Get full video window
            video = frame_buffer.get_video()  # [1, T, C, H, W]

            # Run inference
            with torch.no_grad():
                actions = policy.predict_action(video, proprio)  # [1, chunk_size, action_dim]

            action_list = actions[0].cpu().numpy().tolist()
            log.info(f"Inference done. Buffer: {len(frame_buffer.frames)} frames. "
                     f"Action[0]: [{action_list[0][0]:.3f}, {action_list[0][1]:.3f}, ..., {action_list[0][6]:.3f}]")
            await ws.send(json.dumps(action_list))
            continue

        # Unknown message
        await ws.send(json.dumps({"error": "unknown message type"}))

    log.info("Client disconnected.")


async def main_server(args):
    """Start the WebSocket server."""
    policy, data_config = load_model(args)
    frame_buffer = FrameBuffer(
        num_frames=data_config.num_pixel_frames,
        height=data_config.camera_height,
        width=data_config.camera_width,
    )

    log.info(f"Starting WebSocket server on ws://localhost:{args.port}")

    try:
        import websockets
    except ImportError:
        log.error("websockets not installed. Run: pip install websockets")
        return

    async def handler(ws):
        await handle_client(ws, policy, data_config, frame_buffer)

    async with websockets.serve(handler, "0.0.0.0", args.port, max_size=100 * 1024 * 1024):
        log.info("Server is ready. Waiting for LIBERO client...")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mimic-video model server for LIBERO evaluation")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--precomputed_dir", type=str, default="precomputed/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tau_v", type=float, default=1.0)
    parser.add_argument("--num_action_steps", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(main_server(args))
