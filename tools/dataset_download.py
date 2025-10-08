#!/usr/bin/env python3
# dataset_download.py - 下载Hugging Face数据集
import os
import argparse
from datasets import load_dataset
import json
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="下载Hugging Face数据集")
    parser.add_argument("--dataset", type=str, default="dzunggg/legal-qa-v1",
                        help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="data/lora",
                        help="输出目录")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分割")
    parser.add_argument("--format", type=str, default="json",
                        choices=["json", "jsonl"], 
                        help="输出格式(json或jsonl)")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print(f"开始下载数据集: {args.dataset}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 下载数据集
    try:
        dataset = load_dataset(args.dataset, split=args.split)
        print(f"数据集加载成功，共 {len(dataset)} 条记录")
        
        # 检查数据集结构
        print("数据集特征:")
        for feature in dataset.features:
            print(f"- {feature}")
        
        # 转换为列表
        data_list = []
        for item in tqdm(dataset, desc="处理数据"):
            # 转换为字典
            item_dict = {}
            for key in item:
                # 如果是字节或其他非JSON可序列化类型，转换为字符串
                if isinstance(item[key], bytes):
                    item_dict[key] = item[key].decode("utf-8", errors="ignore")
                else:
                    item_dict[key] = item[key]
            
            # 将数据转换为我们的格式
            # 假设数据集有question和answer字段
            query = item_dict.get("question", item_dict.get("query", ""))
            answer = item_dict.get("answer", "")
            
            # 如果没有question/query和answer字段，尝试使用其他可能的字段名
            if not query and "input" in item_dict:
                query = item_dict["input"]
            if not answer and "output" in item_dict:
                answer = item_dict["output"]
            
            # 创建我们的格式
            formatted_item = {
                "query": query,
                "answer": answer,
                "original": item_dict  # 保留原始数据
            }
            
            data_list.append(formatted_item)
        
        # 保存数据
        if args.format == "json":
            output_path = os.path.join(args.output_dir, f"{args.dataset.split('/')[-1]}.json")
            print(f"保存数据到: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
        else:
            output_path = os.path.join(args.output_dir, f"{args.dataset.split('/')[-1]}.jsonl")
            print(f"保存数据到: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                for item in data_list:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"数据集下载并保存成功，共 {len(data_list)} 条记录")
        
        # 打印前5条记录示例
        print("\n数据示例:")
        for i, item in enumerate(data_list[:5]):
            print(f"示例 {i+1}:")
            print(f"问题: {item['query']}")
            print(f"回答: {item['answer'][:100]}..." if len(item['answer']) > 100 else f"回答: {item['answer']}")
            print("-" * 50)
    
    except Exception as e:
        print(f"下载数据集时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
