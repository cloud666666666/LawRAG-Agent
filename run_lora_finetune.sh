#!/bin/bash
#SBATCH --job-name=lora_finetune   # 作业名称
#SBATCH --time=7-00:00:00          # 最大运行时间（7天）
#SBATCH --partition=slurmpartition # 分区名称
#SBATCH --nodes=1                  # 需要的节点数
#SBATCH --ntasks=1                 # 任务数
#SBATCH --cpus-per-task=4          # 每个任务的CPU核心数
#SBATCH --mem=16G                  # 内存需求
#SBATCH --gres=gpu:A100:1               # 请求A100 GPU:1
#SBATCH --mail-type=BEGIN,END,FAIL # 邮件通知类型

# 打印作业信息
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "提交时间: $(date)"

# 激活虚拟环境
source /data/nway818/LawRAG-Agent/.venv/bin/activate

# 切换到项目目录
cd /data/nway818/LawRAG-Agent

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0

# 打印环境信息
echo "Python版本: $(python --version)"
echo "CUDA设备: $CUDA_VISIBLE_DEVICES"
echo "当前目录: $(pwd)"

# 步骤1: 下载数据集
echo "=== 步骤1: 下载数据集 ==="
python tools/dataset_download.py --dataset dzunggg/legal-qa-v1 --output_dir data/lora --format jsonl

# 步骤2: 准备微调数据
echo "=== 步骤2: 准备微调数据 ==="
python tools/prepare_finetune_data.py

# 步骤3: 执行LoRA微调
echo "=== 步骤3: 执行LoRA微调 ==="
python tools/lora_finetune.py \
  --model_path models/qwen2.5-7b-instruct \
  --output_dir models/qwen2.5-7b-law-lora \
  --train_file data/processed/train_data.json \
  --val_file data/processed/val_data.json \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --cutoff_len 1024 \
  --use_8bit  # 使用8bit量化以减少内存使用

# 步骤4: 测试微调后的模型
echo "=== 步骤4: 测试微调后的模型 ==="
python tools/load_lora_model.py

# 打印完成信息
echo "微调完成时间: $(date)"
