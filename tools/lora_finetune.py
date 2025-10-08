#!/usr/bin/env python3
# lora_finetune.py - 使用LoRA技术微调大语言模型
import os
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("lora_finetune.log")
    ]
)
logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_MODEL_PATH = "models/qwen2.5-7b-instruct"
DEFAULT_OUTPUT_DIR = "models/qwen2.5-7b-law-lora"
# 直接使用已处理的 DISC-Law-SFT 数据集作为默认训练/验证集
# 如需改回使用 data/processed 下由 prepare_finetune_data.py 生成的数据，
# 可将以下两行改回原先的 data/processed/train_data.json 与 val_data.json
DEFAULT_TRAIN_FILE = "data/lora/DISC-Law-SFT-Pair_processed.json"
DEFAULT_VAL_FILE = "data/lora/DISC-Law-SFT-Pair-QA-released_processed.json"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用LoRA技术微调大语言模型")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="基础模型路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="输出模型路径")
    parser.add_argument("--train_file", type=str, default=DEFAULT_TRAIN_FILE,
                        help="训练数据文件")
    parser.add_argument("--val_file", type=str, default=DEFAULT_VAL_FILE,
                        help="验证数据文件")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout率")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="训练批次大小")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="微批次大小")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--cutoff_len", type=int, default=1024,
                        help="最大序列长度")
    parser.add_argument("--val_set_size", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存检查点的步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--use_4bit", action="store_true",
                        help="是否使用4bit量化")
    parser.add_argument("--use_8bit", action="store_true",
                        help="是否使用8bit量化")
    return parser.parse_args()

def prepare_data(train_file, val_file, tokenizer, cutoff_len):
    """准备训练和验证数据"""
    logger.info(f"加载训练数据: {train_file}")
    
    # 检查文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}")
    
    # 加载数据
    if train_file.endswith('.json'):
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    elif train_file.endswith('.jsonl'):
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    train_data.append(json.loads(line))
    else:
        raise ValueError("训练数据文件必须是.json或.jsonl格式")
    
    # 验证数据
    val_data = None
    if val_file and os.path.exists(val_file):
        logger.info(f"加载验证数据: {val_file}")
        if val_file.endswith('.json'):
            with open(val_file, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        elif val_file.endswith('.jsonl'):
            val_data = []
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        val_data.append(json.loads(line))
    
    # 转换为Hugging Face数据集格式
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None
    
    # 数据预处理函数
    def preprocess_function(examples):
        # 假设数据格式为: {"query": "...", "answer": "..."}
        # 构建提示模板
        prompts = []
        for i in range(len(examples["query"])):
            query = examples["query"][i]
            answer = examples["answer"][i]
            prompt = f"问题：{query}\n\n回答：{answer}"
            prompts.append(prompt)
        
        # 分词
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建标签
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 应用预处理
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if val_dataset:
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
    
    return train_dataset, val_dataset

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 若外部未指定可见GPU，则默认固定到第2块卡（cuda:1）
    if "CUDA_VISIBLE_DEVICES" not in os.environ or not os.environ["CUDA_VISIBLE_DEVICES"].strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载分词器
    logger.info(f"加载分词器: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 配置量化参数
    compute_dtype = torch.float16
    quant_config = None
    
    if args.use_4bit:
        logger.info("使用4bit量化")
        compute_dtype = torch.bfloat16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.use_8bit:
        logger.info("使用8bit量化")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 加载模型
    logger.info(f"加载基础模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行LoRA微调
    if args.use_4bit or args.use_8bit:
        logger.info("准备模型进行量化训练")
        model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    logger.info("配置LoRA参数")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 创建PEFT模型
    logger.info("创建PEFT模型")
    model = get_peft_model(model, peft_config)
    
    # 准备数据
    train_dataset, val_dataset = prepare_data(
        args.train_file, args.val_file, tokenizer, args.cutoff_len
    )
    
    # 配置训练参数
    logger.info("配置训练参数")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.save_steps if val_dataset else None,
        report_to="tensorboard",
        fp16=not (args.use_4bit or args.use_8bit),  # 如果使用量化，则不使用fp16
        bf16=args.use_4bit,  # 如果使用4bit量化，则使用bf16
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True if val_dataset else False,
    )
    
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建训练器
    logger.info("创建训练器")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 打印模型参数信息
    logger.info("模型参数信息:")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    logger.info(f"总参数: {all_params:,}")
    
    # 开始训练
    logger.info("开始训练")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("训练完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
