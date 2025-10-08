#!/usr/bin/env python3
# dataset_download.py - 下载Hugging Face数据集
import os
import argparse
from datasets import load_dataset
import json
from tqdm import tqdm
import requests
import shutil
from huggingface_hub import hf_hub_url, HfApi

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="下载Hugging Face数据集")
    parser.add_argument("--dataset", type=str, default="ShengbinYue/DISC-Law-SFT",
                        help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="data/lora",
                        help="输出目录")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分割")
    parser.add_argument("--format", type=str, default="json",
                        choices=["json", "jsonl"], 
                        help="输出格式(json或jsonl)")
    parser.add_argument("--direct_download", action="store_true",
                        help="直接下载文件而不是使用datasets库")
    return parser.parse_args()

def download_file(url, output_path):
    """下载文件到指定路径"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), 
                             desc=f"下载 {os.path.basename(output_path)}"):
                f.write(chunk)
    return output_path

def direct_download_dataset(dataset_name, output_dir):
    """直接下载数据集文件"""
    # 使用HfApi获取数据集中的所有文件
    api = HfApi()
    try:
        files = api.list_repo_files(dataset_name, repo_type="dataset")
        
        # 过滤出JSONL文件
        jsonl_files = [f for f in files if f.endswith('.jsonl')]
        
        if not jsonl_files:
            print(f"警告: 在数据集 {dataset_name} 中未找到JSONL文件")
            return []
        
        # 下载每个JSONL文件
        downloaded_files = []
        for file in jsonl_files:
            url = hf_hub_url(dataset_name, file, repo_type="dataset")
            output_path = os.path.join(output_dir, os.path.basename(file))
            
            print(f"下载文件: {file}")
            download_file(url, output_path)
            downloaded_files.append(output_path)
            
        return downloaded_files
    
    except Exception as e:
        print(f"直接下载数据集文件时出错: {e}")
        return []

def process_jsonl_file(file_path, output_dir, output_format="jsonl"):
    """处理单个JSONL文件"""
    print(f"处理文件: {file_path}")
    
    # 读取JSONL文件
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"解析 {os.path.basename(file_path)}"):
            if line.strip():
                try:
                    item = json.loads(line)
                    
                    # 提取query和answer字段
                    query = None
                    answer = None
                    
                    # 检查可能的字段名
                    if "input" in item:
                        query = item["input"]
                    elif "question" in item:
                        query = item["question"]
                    elif "instruction" in item:
                        query = item["instruction"]
                        
                    if "output" in item:
                        answer = item["output"]
                    elif "answer" in item:
                        answer = item["answer"]
                    elif "response" in item:
                        answer = item["response"]
                    
                    # 如果有reference字段，可以作为额外信息
                    reference = None
                    if "reference" in item:
                        if isinstance(item["reference"], list):
                            reference = " ".join(item["reference"])
                        else:
                            reference = str(item["reference"])
                    
                    # 如果找到了问题和回答
                    if query and answer:
                        # 如果有参考信息，添加到问题中
                        if reference:
                            formatted_query = f"{query}\n\n参考资料：\n{reference}"
                        else:
                            formatted_query = query
                            
                        # 创建格式化的项目
                        formatted_item = {
                            "query": formatted_query,
                            "answer": answer,
                            "original": item  # 保留原始数据
                        }
                        
                        data_list.append(formatted_item)
                except Exception as e:
                    print(f"解析行时出错: {e}")
                    continue
    
    # 保存处理后的数据
    base_name = os.path.basename(file_path).split('.')[0]
    if output_format == "json":
        output_path = os.path.join(output_dir, f"{base_name}_processed.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    else:
        output_path = os.path.join(output_dir, f"{base_name}_processed.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"处理完成: {len(data_list)} 条记录已保存到 {output_path}")
    return output_path, len(data_list)

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print(f"开始下载数据集: {args.dataset}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 使用直接下载方式
    if args.direct_download:
        print("使用直接下载方式获取数据集文件")
        downloaded_files = direct_download_dataset(args.dataset, args.output_dir)
        
        if not downloaded_files:
            print("未能下载任何文件，尝试使用datasets库")
            args.direct_download = False
        else:
            # 处理每个下载的文件
            total_records = 0
            processed_files = []
            
            for file_path in downloaded_files:
                output_path, record_count = process_jsonl_file(
                    file_path, args.output_dir, args.format)
                total_records += record_count
                processed_files.append(output_path)
            
            print(f"\n数据集处理完成，共 {total_records} 条记录")
            print(f"处理后的文件:")
            for file_path in processed_files:
                print(f"- {file_path}")
            
            # 打印数据示例
            if processed_files:
                sample_file = processed_files[0]
                if sample_file.endswith('.json'):
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        samples = json.load(f)[:5]
                else:
                    samples = []
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if i >= 5:
                                break
                            if line.strip():
                                samples.append(json.loads(line))
                
                print("\n数据示例:")
                for i, item in enumerate(samples):
                    print(f"示例 {i+1}:")
                    print(f"问题: {item['query'][:100]}..." if len(item['query']) > 100 else f"问题: {item['query']}")
                    print(f"回答: {item['answer'][:100]}..." if len(item['answer']) > 100 else f"回答: {item['answer']}")
                    print("-" * 50)
            
            return
    
    # 使用datasets库下载
    try:
        # 尝试加载整个数据集
        print("使用datasets库加载数据集")
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
        print("尝试使用直接下载方式...")
        
        # 如果使用datasets库失败，尝试直接下载
        downloaded_files = direct_download_dataset(args.dataset, args.output_dir)
        
        if not downloaded_files:
            print("未能下载任何文件")
            import traceback
            traceback.print_exc()
            return
        
        # 处理每个下载的文件
        total_records = 0
        processed_files = []
        
        for file_path in downloaded_files:
            output_path, record_count = process_jsonl_file(
                file_path, args.output_dir, args.format)
            total_records += record_count
            processed_files.append(output_path)
        
        print(f"\n数据集处理完成，共 {total_records} 条记录")
        print(f"处理后的文件:")
        for file_path in processed_files:
            print(f"- {file_path}")

if __name__ == "__main__":
    main()