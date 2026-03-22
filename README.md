# mimic-video Replication

Replication of **mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs**.

## Overview

This project implements the mimic-video approach from scratch, training on the `pierre818191/UnitreeBagClose` dataset (Unitree G1, 200 episodes, 4 cameras, 30fps, bag-closing task).

### Architecture
- **Video backbone**: Cosmos Predict2-2B with LoRA finetuning
- **Action decoder**: Lightweight DiT (~67M params) with flow matching
- **Two-stage training**: Stage 1 (video prediction) → Stage 2 (action prediction)

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# 1. Precompute T5 embeddings
python scripts/precompute_embeddings.py

# 2. Stage 1: LoRA finetuning of video backbone
python scripts/train_stage1.py

# 3. Stage 2: Action decoder training
python scripts/train_stage2.py

# 4. Evaluation
python scripts/evaluate.py --stage1_checkpoint checkpoints/stage1/final --stage2_checkpoint checkpoints/stage2/final
```

## Project Structure

```
mimic_approach/
├── configs/config.py              # Dataclass-based configs
├── mimic_video/
│   ├── data/
│   │   ├── dataset.py             # MimicVideoDataset (LeRobot wrapper)
│   │   └── transforms.py          # 2x2 camera concat, normalization
│   ├── models/
│   │   ├── video_backbone.py      # Cosmos backbone + LoRA + hidden state hooks
│   │   ├── action_decoder.py      # ActionDecoderDiT (flow matching)
│   │   └── flow_matching.py       # FM scheduler, ODE solver
│   ├── training/
│   │   ├── stage1_trainer.py      # Video backbone LoRA finetuning
│   │   └── stage2_trainer.py      # Action decoder training
│   └── inference/
│       └── policy.py              # MimicVideoPolicy (Algorithm 1)
└── scripts/
    ├── precompute_embeddings.py
    ├── train_stage1.py
    ├── train_stage2.py
    └── evaluate.py
```
