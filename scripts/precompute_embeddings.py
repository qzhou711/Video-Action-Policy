"""Precompute T5 text embeddings and optionally VAE latents for faster training.

Supports both single-task (one prompt) and multi-task (from dataset metadata).

Usage:
    # Multi-task: auto-detect tasks from dataset metadata
    python scripts/precompute_embeddings.py --output_dir precomputed/

    # Single-task: explicit prompt
    python scripts/precompute_embeddings.py --prompt "Pick up the bowl" --output_dir precomputed/

    # Also precompute VAE latents
    python scripts/precompute_embeddings.py --latents --output_dir precomputed/
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DataConfig, ModelConfig, get_suite_data_config, LIBERO_SUITES


def load_t5_encoder(model_id, dtype=torch.bfloat16, device="cuda"):
    """Load T5 tokenizer and encoder from a Cosmos pipeline."""
    from transformers import T5EncoderModel, T5TokenizerFast

    print(f"Loading T5 text encoder from {model_id}...")
    tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.eval()
    return tokenizer, text_encoder


def encode_prompt(tokenizer, text_encoder, prompt, device="cuda"):
    """Encode a single text prompt into a T5 embedding."""
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

    return text_embeds


def get_task_descriptions_from_dataset(repo_id):
    """Extract task descriptions from a LeRobot dataset's metadata.

    Returns:
        dict: {task_index (int): task_description (str)}
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset metadata from {repo_id}...")
    dataset = LeRobotDataset(repo_id=repo_id)

    # LeRobot v3.0 stores tasks in meta/tasks
    tasks = {}
    if hasattr(dataset.meta, 'tasks') and dataset.meta.tasks is not None:
        for task_entry in dataset.meta.tasks:
            task_idx = task_entry["task_index"]
            task_desc = task_entry["task"]
            tasks[task_idx] = task_desc
    else:
        # Fallback: try to read tasks.jsonl directly from cache
        cache_dir = dataset.root
        tasks_path = Path(cache_dir) / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    tasks[entry["task_index"]] = entry["task"]
        else:
            print("WARNING: Could not find task descriptions in dataset metadata.")

    del dataset
    return tasks


def precompute_multi_task_t5_embeddings(
    model_id: str,
    repo_id: str,
    output_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Precompute and cache T5 embeddings for all tasks in a multi-task dataset.

    Saves:
        t5_embeddings.pt: dict {task_index (int): embedding [1, seq_len, dim]}
        t5_task_descriptions.json: dict {task_index: task_description}
    """
    tasks = get_task_descriptions_from_dataset(repo_id)

    if not tasks:
        print("ERROR: No tasks found. Cannot precompute multi-task embeddings.")
        return None

    print(f"\nFound {len(tasks)} tasks:")
    for idx, desc in sorted(tasks.items()):
        print(f"  Task {idx}: {desc}")

    tokenizer, text_encoder = load_t5_encoder(model_id, dtype=dtype, device=device)

    embeddings = {}
    for task_idx, task_desc in sorted(tasks.items()):
        print(f"Encoding task {task_idx}: '{task_desc}'...")
        emb = encode_prompt(tokenizer, text_encoder, task_desc, device=device)
        embeddings[task_idx] = emb.cpu()
        print(f"  Shape: {emb.shape}")

    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings dict
    emb_path = os.path.join(output_dir, "t5_embeddings.pt")
    torch.save(embeddings, emb_path)
    print(f"\nSaved multi-task T5 embeddings to {emb_path}")

    # Save task descriptions for reference
    desc_path = os.path.join(output_dir, "t5_task_descriptions.json")
    # Convert int keys to str for JSON
    with open(desc_path, "w") as f:
        json.dump({str(k): v for k, v in tasks.items()}, f, indent=2, ensure_ascii=False)
    print(f"Saved task descriptions to {desc_path}")

    # Also save a single combined embedding (first task) as fallback
    first_key = sorted(embeddings.keys())[0]
    fallback_path = os.path.join(output_dir, "t5_embedding.pt")
    torch.save(embeddings[first_key], fallback_path)
    print(f"Saved fallback single T5 embedding (task {first_key}) to {fallback_path}")

    # Clean up
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    return embeddings


def precompute_single_t5_embedding(
    model_id: str,
    prompt: str,
    output_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Precompute and cache a single T5 text embedding (legacy single-task mode)."""
    tokenizer, text_encoder = load_t5_encoder(model_id, dtype=dtype, device=device)

    print(f"Encoding prompt: '{prompt}'")
    text_embeds = encode_prompt(tokenizer, text_encoder, prompt, device=device)
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
    parser.add_argument("--suite", type=str, default=None,
                        choices=list(LIBERO_SUITES.keys()),
                        help="LIBERO suite name (auto-sets repo_id and output_dir)")
    parser.add_argument("--latents", action="store_true", help="Also precompute VAE latents")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single task prompt (if not set, auto-detects tasks from dataset metadata)"
    )
    args = parser.parse_args()

    # Config: use suite if specified, otherwise defaults
    if args.suite:
        data_config = get_suite_data_config(args.suite)
    else:
        data_config = DataConfig()
    if args.output_dir:
        data_config.precomputed_dir = args.output_dir

    model_config = ModelConfig()

    if args.prompt:
        # Single-task mode: explicit prompt
        precompute_single_t5_embedding(
            model_id=model_config.cosmos_model_id,
            prompt=args.prompt,
            output_dir=data_config.precomputed_dir,
            device=args.device,
        )
    else:
        # Multi-task mode: auto-detect from dataset
        precompute_multi_task_t5_embeddings(
            model_id=model_config.cosmos_model_id,
            repo_id=data_config.repo_id,
            output_dir=data_config.precomputed_dir,
            device=args.device,
        )

    # Optionally precompute VAE latents
    if args.latents:
        precompute_vae_latents(
            model_id=model_config.cosmos_model_id,
            data_config=data_config,
            output_dir=data_config.precomputed_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
