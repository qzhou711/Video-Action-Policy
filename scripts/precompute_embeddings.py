"""Precompute T5 text embeddings and optionally VAE latents for faster training.

Usage:
    python scripts/precompute_embeddings.py [--latents] [--output_dir precomputed/]
"""

import os
import argparse
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig


def precompute_t5_embedding(
    model_id: str,
    prompt: str,
    output_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Precompute and cache the T5 text embedding for the task prompt.

    Args:
        model_id: HuggingFace model ID for the Cosmos pipeline.
        prompt: Text prompt to encode.
        output_dir: Directory to save the embedding.
        dtype: Compute dtype.
        device: Device to run on.
    """
    from transformers import T5EncoderModel, T5TokenizerFast

    print(f"Loading T5 text encoder from {model_id}...")

    # Load T5 components from the pipeline
    tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.eval()

    print(f"Encoding prompt: '{prompt}'")
    with torch.no_grad():
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        text_embeds = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]  # [1, seq_len, hidden_dim]

    print(f"T5 embedding shape: {text_embeds.shape}")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "t5_embedding.pt")
    torch.save(text_embeds.cpu(), save_path)
    print(f"Saved T5 embedding to {save_path}")

    # Clean up
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    return text_embeds


def precompute_vae_latents(
    model_id: str,
    data_config: DataConfig,
    output_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Precompute VAE latents for all episodes.

    This dramatically speeds up training by avoiding repeated VAE encoding.

    Args:
        model_id: HuggingFace model ID for the Cosmos pipeline.
        data_config: Data configuration.
        output_dir: Directory to save latents.
        dtype: Compute dtype.
        device: Device to run on.
    """
    from diffusers import AutoencoderKLWan
    from mimic_video.data.dataset import MimicVideoDataset
    from mimic_video.data.transforms import concat_cameras, normalize_to_neg1_pos1

    print(f"Loading VAE from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval()

    # Get normalization constants
    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device)
    latents_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device)

    print("Loading dataset...")
    dataset = MimicVideoDataset(
        repo_id=data_config.repo_id,
        camera_names=data_config.camera_names,
        state_keys=data_config.state_keys,
        action_keys=data_config.action_keys,
        num_pixel_frames=data_config.num_pixel_frames,
        action_chunk_size=data_config.action_chunk_size,
        action_dim=data_config.action_dim,
        proprio_dim=data_config.proprio_dim,
        target_height=data_config.camera_height,
        target_width=data_config.camera_width,
        fps=data_config.fps,
    )

    latents_dir = os.path.join(output_dir, "vae_latents")
    os.makedirs(latents_dir, exist_ok=True)

    print(f"Precomputing VAE latents for {len(dataset)} samples...")
    for idx in tqdm(range(len(dataset))):
        save_path = os.path.join(latents_dir, f"latent_{idx:06d}.pt")
        if os.path.exists(save_path):
            continue

        sample = dataset[idx]
        video = sample["video"]  # [T, C, H, W]

        # Rearrange to [1, C, T, H, W]
        video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype=dtype)

        with torch.no_grad():
            posterior = vae.encode(video).latent_dist
            latents = posterior.mode().float()
            # Normalize
            latents = (latents - latents_mean) / latents_std

        torch.save(latents.cpu(), save_path)

    print(f"Saved {len(dataset)} VAE latents to {latents_dir}")

    del vae
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings for mimic-video")
    parser.add_argument("--latents", action="store_true", help="Also precompute VAE latents")
    parser.add_argument("--output_dir", type=str, default="precomputed/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_config = DataConfig()
    model_config = ModelConfig()

    # Always precompute T5 embedding
    precompute_t5_embedding(
        model_id=model_config.cosmos_model_id,
        prompt=data_config.task_prompt,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Optionally precompute VAE latents
    if args.latents:
        precompute_vae_latents(
            model_id=model_config.cosmos_model_id,
            data_config=data_config,
            output_dir=args.output_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
