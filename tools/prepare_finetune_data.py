#!/usr/bin/env python3
# prepare_finetune_data.py - 准备法律领域微调数据
import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="准备法律领域微调数据")
    parser.add_argument("--input_dir", type=str, default="data/processed",
                        help="输入数据目录")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="输出数据目录")
    parser.add_argument("--train_file", type=str, default="train_data.json",
                        help="训练数据文件名")
    parser.add_argument("--val_file", type=str, default="val_data.json",
                        help="验证数据文件名")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()

def load_law_data(input_dir: str) -> List[Dict[str, Any]]:
    """加载法律文本数据"""
    data = []
    
    # 加载chunks_with_summary.jsonl文件
    chunks_file = os.path.join(input_dir, "chunks_with_summary.jsonl")
    if os.path.exists(chunks_file):
        print(f"加载法律文本数据: {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
    else:
        print(f"警告: 找不到法律文本数据文件 {chunks_file}")
    
    print(f"共加载 {len(data)} 条法律文本数据")
    return data

def load_qa_data(input_dir: str) -> List[Dict[str, Any]]:
    """加载问答数据"""
    qa_data = []
    
    # 尝试加载问答数据文件
    qa_files = [
        os.path.join(input_dir, "qa_pairs.json"),
        os.path.join(input_dir, "qa_data.json"),
        os.path.join(input_dir, "law_qa.json"),
        os.path.join("data/lora", "legal-qa-v1.json"),
        os.path.join("data/lora", "legal-qa-v1.jsonl")
    ]
    
    for qa_file in qa_files:
        if os.path.exists(qa_file):
            print(f"加载问答数据: {qa_file}")
            if qa_file.endswith('.json'):
                with open(qa_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    qa_data.extend(items)
            elif qa_file.endswith('.jsonl'):
                with open(qa_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            qa_data.append(item)
    
    if not qa_data:
        print("警告: 未找到问答数据文件")
    else:
        print(f"共加载 {len(qa_data)} 条问答数据")
    
    return qa_data

def generate_instruction_data(law_data: List[Dict[str, Any]], qa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成指令微调数据"""
    instruction_data = []
    
    # 从法律文本数据生成指令
    print("从法律文本生成指令数据...")
    for item in tqdm(law_data):
        # 从文本生成解释任务
        if "text" in item and len(item["text"]) > 50:
            # 解释法律条文
            instruction_data.append({
                "query": f"请解释以下法律条文的含义：\n\n{item['text']}",
                "answer": item.get("meta", {}).get("summary", f"这是关于{item.get('meta', {}).get('law', '法律')}的条文，主要规定了相关的法律要求和责任。")
            })
            
            # 提取关键概念
            if random.random() < 0.3:  # 只为30%的条目生成此类任务
                instruction_data.append({
                    "query": f"请从以下法律条文中提取关键法律概念：\n\n{item['text']}",
                    "answer": f"根据提供的法律条文，关键法律概念包括{item.get('meta', {}).get('law', '相关法律')}中规定的权利义务关系。"
                })
    
    # 从问答数据生成指令
    print("从问答数据生成指令数据...")
    for item in qa_data:
        # 处理不同格式的问答数据
        query = None
        answer = None
        
        # 检查各种可能的字段名
        if "question" in item:
            query = item["question"]
        elif "query" in item:
            query = item["query"]
        
        if "answer" in item:
            answer = item["answer"]
        
        # 检查我们下载的数据集特定格式
        if query is None and "original" in item and "question" in item["original"]:
            query = item["original"]["question"]
            
        if answer is None and "original" in item and "answer" in item["original"]:
            answer = item["original"]["answer"]
        
        # 如果找到了问题和回答，添加到指令数据中
        if query and answer:
            instruction_data.append({
                "query": query,
                "answer": answer
            })
    
    # 如果没有问答数据，生成一些示例问答
    if not qa_data:
        print("生成示例法律问答数据...")
        example_qa = [
            {
                "query": "什么是民事行为能力？",
                "answer": "民事行为能力是指自然人通过自己的行为实施民事法律行为、取得民事权利和承担民事义务的资格。根据《中华人民共和国民法典》第十八条至第二十二条的规定，自然人的民事行为能力分为完全民事行为能力、限制民事行为能力和无民事行为能力三种情况。"
            },
            {
                "query": "合同成立的条件有哪些？",
                "answer": "根据《中华人民共和国民法典》第四百七十条的规定，合同成立的条件包括：1. 当事人具有相应的民事行为能力；2. 意思表示真实；3. 不违反法律、行政法规的强制性规定，不违背公序良俗。合同成立一般采取要约-承诺的方式，当承诺生效时合同成立。"
            },
            {
                "query": "侵权责任的构成要件是什么？",
                "answer": "根据《中华人民共和国民法典》的规定，侵权责任的构成要件一般包括：1. 行为人实施了侵权行为；2. 受害人的民事权益受到损害；3. 行为人的行为与损害后果之间存在因果关系；4. 行为人主观上有过错（特殊情况下可以适用无过错责任）。"
            }
        ]
        instruction_data.extend(example_qa)
    
    print(f"共生成 {len(instruction_data)} 条指令数据")
    return instruction_data

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    law_data = load_law_data(args.input_dir)
    qa_data = load_qa_data(args.input_dir)
    
    # 生成指令数据
    instruction_data = generate_instruction_data(law_data, qa_data)
    
    # 随机打乱数据
    random.shuffle(instruction_data)
    
    # 划分训练集和验证集
    val_size = int(len(instruction_data) * args.val_split)
    train_data = instruction_data[val_size:]
    val_data = instruction_data[:val_size]
    
    # 保存数据
    train_path = os.path.join(args.output_dir, args.train_file)
    val_path = os.path.join(args.output_dir, args.val_file)
    
    print(f"保存训练数据到: {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"保存验证数据到: {val_path}")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据准备完成! 训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"数据准备过程出错: {e}")
        import traceback
        traceback.print_exc()
