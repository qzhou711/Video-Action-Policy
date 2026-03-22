# DIT4DIT

A **mimic-video**–style video–action model: LoRA-finetune a Cosmos video backbone, then train a lightweight DiT with flow matching for action prediction on **LIBERO** manipulation tasks (LeRobot-format data).

## Overview

- **Data**: default [`lerobot/libero_10`](https://huggingface.co/datasets/lerobot/libero_10) (379 episodes; train/val split in `configs/config.py`)
- **Cameras**: agentview + wrist (`observation.images.image` / `observation.images.wrist_image`), 256×256, concatenated side-by-side into a single stream
- **State / actions**: 8-D proprio, 7-D actions (x, y, z, roll, pitch, yaw, gripper), action chunk size 16
- **Video**: 17 pixel frames → 5 latent frames (2 conditional + 3 predicted), 10 FPS
- **Backbone**: `nvidia/Cosmos-Predict2-2B-Video2World` + LoRA
- **Action head**: ~67M-parameter DiT + flow matching
- **Training**: two stages — Stage 1 video prediction (LoRA) → Stage 2 action prediction

## Setup

```bash
pip install -r requirements.txt
# For LIBERO sim eval (WebSocket server + client):
pip install websockets
```

**Python 3.10+**, **CUDA**, and **bf16** are recommended (training defaults to bf16).

## Configuration

Hyperparameters and paths live in `configs/config.py` (`DataConfig`, `ModelConfig`, `Stage1Config`, `Stage2Config`). Edit that file to change the dataset, camera keys, checkpoint directories, W&B project names, etc.

## Training

```bash
# 1. Precompute T5 text embeddings (optional: also precompute VAE latents for speed)
python scripts/precompute_embeddings.py
# python scripts/precompute_embeddings.py --latents --output_dir precomputed/

# 2. Stage 1: video backbone LoRA
python scripts/train_stage1.py

# 3. Stage 2: action decoder (loads Stage 1 from checkpoints/stage1/final by default)
python scripts/train_stage2.py

# 4. Action MSE on the val set (optional video visualization)
python scripts/evaluate.py \
  --stage1_checkpoint checkpoints/stage1/final \
  --stage2_checkpoint checkpoints/stage2/final
```

Evaluation scripts set `HF_HUB_OFFLINE=1` so Hugging Face models are resolved from the local cache when possible; the first run still needs network access to download Cosmos / T5 weights.

## LIBERO simulation evaluation (optional)

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) in a **separate** environment, then run the WebSocket server and sim client (see also `LIBERO_evaluation/setup.sh`).

**Terminal 1 — model server:**

```bash
python scripts/eval_server.py \
  --stage1_checkpoint checkpoints/stage1/final \
  --stage2_checkpoint checkpoints/stage2/final \
  --port 8765
```

**Terminal 2 — LIBERO client:**

```bash
python LIBERO_evaluation/libero_client.py \
  --server_url ws://localhost:8765 \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --num_episodes 20
```

## Inference latency benchmark (optional)

```bash
python scripts/benchmark_latency.py --dry_run
python scripts/benchmark_latency.py --device cuda --warmup 3 --repeats 10
```

## Repository layout

```
DIT4DIT/
├── configs/
│   └── config.py                 # Data / model / two-stage training config
├── mimic_video/
│   ├── data/
│   │   ├── dataset.py            # MimicVideoDataset (LeRobot)
│   │   └── transforms.py         # Two-camera concat, normalization, etc.
│   ├── models/
│   │   ├── video_backbone.py     # Cosmos + LoRA + hidden-state hooks
│   │   ├── action_decoder.py     # ActionDecoderDiT (flow matching)
│   │   └── flow_matching.py      # Scheduler and ODE solver
│   ├── training/
│   │   ├── stage1_trainer.py
│   │   └── stage2_trainer.py
│   └── inference/
│       └── policy.py             # MimicVideoPolicy (inference wrapper)
├── LIBERO_evaluation/
│   ├── libero_client.py          # Sim client talking to eval_server
│   └── setup.sh                  # Two-env run notes (edit paths for your machine)
└── scripts/
    ├── precompute_embeddings.py
    ├── train_stage1.py
    ├── train_stage2.py
    ├── evaluate.py
    ├── eval_server.py            # WebSocket server for LIBERO
    └── benchmark_latency.py
```

## Reference

Paper: **mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs**. This repo is an independent reimplementation / extension; behavior and defaults are defined by the code and `configs/config.py`.
