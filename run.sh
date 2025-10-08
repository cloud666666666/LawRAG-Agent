#!/bin/bash
#SBATCH --job-name=summary_gen        # 作业名称

#SBATCH --time=7-00:00:00             # 最大运行时间（7天）
#SBATCH --partition=slurmpartition    # 分区名称
#SBATCH --nodes=1                     # 需要的节点数
#SBATCH --ntasks=1                    # 任务数
#SBATCH --cpus-per-task=4             # 每个任务的CPU核心数
#SBATCH --mem=16G                     # 内存需求
#SBATCH --gres=gpu:1                  # 请求1个GPU（不限定型号）
#SBATCH --mail-type=BEGIN,END,FAIL    # 邮件通知类型（可选）
##SBATCH --mail-user=nway818@aucklanduni.ac.nz 接收邮件的地址（可选，取消注释并修改）

# 打印作业信息
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "提交时间: $(date)"

# 加载必要的环境模块（如果需要）
# module load python/3.11
# module load cuda/11.8

# 激活虚拟环境（如果使用）
source /data/nway818/LawRAG-Agent/.venv/bin/activate

# 切换到项目目录
cd /data/nway818/LawRAG-Agent

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=1

# 打印环境信息
echo "Python版本: $(python --version)"
echo "CUDA设备: $CUDA_VISIBLE_DEVICES"
echo "当前目录: $(pwd)"

# 运行摘要生成脚本
# 可以根据需要修改以下参数
/data/nway818/LawRAG-Agent/.venv/bin/python tools/lora_finetune.py 
# 打印完成信息
echo "作业完成时间: $(date)"
