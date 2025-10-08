#!/usr/bin/env python3
# generate.py - 使用检索结果和大语言模型生成回答
import json
import os
import sys
import time
from pathlib import Path
from models_load import llm_load
import torch
from tqdm import tqdm

def format_retrieved_context(results, max_items=5):
    """
    将检索结果格式化为上下文
    
    参数:
    - results: 检索结果列表
    - max_items: 最大使用的检索结果数量
    
    返回:
    - 格式化的上下文字符串
    """
    context_parts = []
    
    # 限制使用的检索结果数量
    results = results[:max_items]
    
    for i, item in enumerate(results, 1):
        doc = item["document"]
        meta = item["metadata"]
        law = meta.get("law", "未知法律")
        path = meta.get("path", "")
        doc_type = meta.get("type", "")
        
        # 根据类型添加不同的前缀
        prefix = f"[{law}] "
        if doc_type == "summary":
            prefix += "摘要: "
        
        context_parts.append(f"{i}. {prefix}{doc}")
    
    return "\n".join(context_parts)

def generate_answer(query, context, llm, tokenizer, max_new_tokens=512):
    """
    使用大语言模型生成回答
    
    参数:
    - query: 用户查询
    - context: 检索的上下文
    - llm: 大语言模型
    - tokenizer: 分词器
    - max_new_tokens: 最大生成的token数量
    
    返回:
    - 生成的回答
    """
    # 构建提示
    system_prompt = "你是一个专业的中文法律助手。请根据提供的法律条文回答用户的问题。回答应当准确、客观，并明确引用相关法条。"
    
    user_prompt = f"""请回答以下法律问题，并参考给出的法律条文：

问题：{query}

相关法律条文：
{context}

请基于上述法律条文回答问题，如果条文不足以回答问题，请明确指出。回答需要客观准确，并引用相关条文。
"""
    
    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 应用聊天模板
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096 - max_new_tokens
    )
    
    # 移动到正确的设备
    device = next(llm.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成回答
    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # 解码并提取回答
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return answer.strip()

def main():
    """主函数"""
    print("\n=== 开始生成回答 ===")
    
    # 配置
    results_path = "data/results/query_results.json"
    output_path = "data/results/answers.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载检索结果
    print(f"加载检索结果: {results_path}")
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            query_results = json.load(f)
    except Exception as e:
        print(f"加载检索结果失败: {e}")
        return
    
    print(f"共加载 {len(query_results)} 个查询结果")
    
    # 加载大语言模型
    print("加载大语言模型...")
    llm, tokenizer = llm_load()
    print("大语言模型加载完成")
    
    # 处理每个查询
    answers = []
    for query_item in tqdm(query_results, desc="生成回答"):
        query = query_item["query"]
        results = query_item["results"]
        
        # 格式化上下文
        context = format_retrieved_context(results)
        
        # 生成回答
        start_time = time.time()
        answer = generate_answer(query, context, llm, tokenizer)
        elapsed = time.time() - start_time
        
        # 保存结果
        answer_item = {
            "query": query,
            "answer": answer,
            "time": elapsed,
            "context": context
        }
        answers.append(answer_item)
        
        print(f"回答生成完成: {query}")
    
    # 保存所有回答
    print(f"保存回答到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    print("\n=== 回答生成完成 ===")
    print(f"共生成 {len(answers)} 个回答")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
