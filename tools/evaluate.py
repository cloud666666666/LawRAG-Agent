#!/usr/bin/env python3
# evaluate.py - 评估检索增强生成的回答质量
import json
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None

def calculate_metrics(answers):
    """计算基本评估指标"""
    if not answers:
        return {}
    
    # 计算回答长度
    answer_lengths = [len(item["answer"]) for item in answers]
    
    # 计算生成时间
    generation_times = [item.get("time", 0) for item in answers]
    
    # 计算引用数量（简单估计）
    citation_counts = []
    for item in answers:
        answer = item["answer"]
        # 简单统计可能的引用标记
        citations = 0
        citations += answer.count("《")
        citations += answer.count("第") 
        citations += answer.count("条")
        citations += answer.count("款")
        citations += answer.count("项")
        citation_counts.append(citations)
    
    metrics = {
        "answer_length": {
            "mean": np.mean(answer_lengths),
            "median": np.median(answer_lengths),
            "min": min(answer_lengths),
            "max": max(answer_lengths)
        },
        "generation_time": {
            "mean": np.mean(generation_times),
            "median": np.median(generation_times),
            "min": min(generation_times),
            "max": max(generation_times),
            "total": sum(generation_times)
        },
        "citation_count": {
            "mean": np.mean(citation_counts),
            "median": np.median(citation_counts),
            "min": min(citation_counts),
            "max": max(citation_counts)
        }
    }
    
    return metrics

def main():
    """主函数"""
    print("\n=== 开始评估回答质量 ===")
    
    # 配置
    answers_path = "data/results/answers.json"
    output_path = "data/results/evaluation.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载回答
    print(f"加载回答: {answers_path}")
    answers = load_json_file(answers_path)
    if not answers:
        return
    
    print(f"共加载 {len(answers)} 个回答")
    
    # 计算指标
    print("计算评估指标...")
    metrics = calculate_metrics(answers)
    
    # 保存评估结果
    print(f"保存评估结果到: {output_path}")
    evaluation = {
        "metrics": metrics,
        "sample_count": len(answers)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    
    # 打印主要指标
    print("\n=== 评估结果摘要 ===")
    print(f"样本数量: {len(answers)}")
    print(f"平均回答长度: {metrics['answer_length']['mean']:.1f} 字符")
    print(f"平均生成时间: {metrics['generation_time']['mean']:.2f} 秒")
    print(f"总生成时间: {metrics['generation_time']['total']:.2f} 秒")
    print(f"平均引用数量: {metrics['citation_count']['mean']:.1f}")
    
    print("\n=== 评估完成 ===")
    print(f"详细结果已保存到: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
