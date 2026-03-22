#!/bin/bash
# LIBERO Simulation Evaluation - Quick Reference
#
# LIBERO is already installed at: /home/hulab/NAS/qiang/RynnVLA-002/LIBERO
# LIBERO env: rynnvla (already has websockets + imageio)
# Model env: mimic (already has websockets)
#
# ========================================
# USAGE (two terminals):
# ========================================
#
# Terminal 1 (model server):
#   conda activate mimic
#   cd /home/hulab/projects/world_action_model/mimic_approach
#   python scripts/eval_server.py \
#       --stage1_checkpoint checkpoints/stage1/final \
#       --stage2_checkpoint checkpoints/stage2/final
#
# Terminal 2 (LIBERO simulation):
#   conda activate rynnvla
#   cd /home/hulab/projects/world_action_model/mimic_approach
#   python LIBERO_evaluation/libero_client.py \
#       --suites libero_spatial libero_object libero_goal libero_10 \
#       --num_episodes 20
