#!/usr/bin/env python3
# load_lora_model.py - 加载LoRA微调后的模型
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_lora_model(base_model_path="models/qwen2.5-7b-instruct", 
                   lora_model_path="models/qwen2.5-7b-law-lora",
                   device="cuda:0"):
    """
    加载LoRA微调后的模型
    
    参数:
    - base_model_path: 基础模型路径
    - lora_model_path: LoRA权重路径
    - device: 设备
    
    返回:
    - model: 加载了LoRA权重的模型
    - tokenizer: 分词器
    """
    print(f"加载基础模型: {base_model_path}")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 检查LoRA模型路径是否存在
    if not os.path.exists(lora_model_path):
        print(f"警告: LoRA模型路径不存在: {lora_model_path}")
        print("返回原始模型")
        return model, tokenizer
    
    # 加载LoRA权重
    print(f"加载LoRA权重: {lora_model_path}")
    model = PeftModel.from_pretrained(
        model,
        lora_model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # 合并LoRA权重（可选）
    # model = model.merge_and_unload()
    
    return model, tokenizer

def generate_with_lora(model, tokenizer, prompt, max_new_tokens=512, 
                      temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    """
    使用加载了LoRA权重的模型生成文本
    
    参数:
    - model: 模型
    - tokenizer: 分词器
    - prompt: 提示文本
    - max_new_tokens: 最大生成token数
    - temperature: 温度参数
    - top_p: top-p采样参数
    - repetition_penalty: 重复惩罚参数
    
    返回:
    - 生成的文本
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成文本
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取回答部分（去除输入提示）
    answer = generated_text[len(prompt):]
    
    return answer.strip()

if __name__ == "__main__":
    # 示例用法
    model, tokenizer = load_lora_model()
    
    # 测试生成
    test_prompts = [
        "什么是民事行为能力？",
        "合同成立的条件有哪些？",
        "侵权责任的构成要件是什么？"
    ]
    
    print("\n=== 测试LoRA微调模型 ===")
    for prompt in test_prompts:
        print(f"\n问题: {prompt}")
        answer = generate_with_lora(model, tokenizer, prompt)
        print(f"回答: {answer}")
    
    print("\n测试完成!")
