#!/bin/bash
#SBATCH --job-name="dit4dit_stage2_object"      # 任务名称
#SBATCH --partition=gpuA100x8
#SBATCH --account=bbju-delta-gpu         # 你的项目账号
#SBATCH --nodes=1                        # 申请 1 个节点
#SBATCH --ntasks-per-node=8              # 每个节点 8 个任务（对应 8 张卡）
#SBATCH --cpus-per-task=16               # 每个任务分配的 CPU 核心数
#SBATCH --gres=gpu:nvidia_a100:8         # 申请 8 张 A100 显卡
#SBATCH --mem=256G                       # 申请 256G 内存
#SBATCH --time=48:00:00                  # 运行限时 (天-小时:分:秒)
#SBATCH --output=logs/%j_train.log       # 标准输出日志 (需手动创建 logs 文件夹)
#SBATCH --error=logs/%j_train.err        # 错误日志

# --- 邮件通知设置 ---
#SBATCH --mail-type=BEGIN,END,FAIL       # 任务开始、结束、失败时发送邮件
#SBATCH --mail-user=qiz166@pitt.edu   # 接收邮件的地址

# --- 环境准备 ---
# 加载基础模块
module load pytorch-conda/2.8
# 初始化 Conda 并激活你在 projects 下的私人环境
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate /projects/bbju/qzhou6/conda_envs/mimic

# 设置环境变量，确保 WandB 和 HF 路径正确（避开 Home 目录）
export HF_HOME="/work/hdd/bbju/qzhou6/hf_cache"
export WANDB_PROJECT="dit4dit-stage2_object_a100_40g_8gpus"

# 进入代码目录
cd /work/hdd/bbju/qzhou6/DIT4DIT

# --- 执行训练 ---
# 注意：在 sbatch 中，torchrun 会自动识别 SLURM 的参数
torchrun --nproc_per_node=8 scripts/train_stage2.py \
    --suite libero_object \
    --preset a100_40g \
    --lr 1.4e-4 \
    --wandb_project "dit4dit-stage2_object_a100_40g_8gpus" \
    --stage1_checkpoint checkpoints/libero_object/stage1/step_6000


    #--resume checkpoints/libero_spatial/stage2/step_7000
