# debug_chunk.py
# 调试版本：用于测试和定位问题
from pathlib import Path
import re, json
import torch
import gc
import traceback
from transformers import GenerationConfig
from models_load import llm_load
from transformers.utils import logging
logging.set_verbosity_info()  

IN  = Path("data/origin/民法典.txt")
OUT = Path("data/processed/chunks_debug.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def test_model_loading():
    """测试模型加载"""
    print("=== 测试模型加载 ===")
    try:
        llm, tokenizer = llm_load()
        print("✓ 模型加载成功")
        
        # 测试基本功能
        test_text = "测试文本"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"✓ 分词器工作正常，输入形状: {inputs['input_ids'].shape}")
        
        return llm, tokenizer
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        traceback.print_exc()
        return None, None

def test_summarize_function(llm, tokenizer, test_cases):
    """测试摘要生成函数"""
    print("\n=== 测试摘要生成 ===")
    
    def summarize_simple(title: str, body: str) -> str:
        try:
            text = (body or title or "").strip()
            if not text:
                return ""

            user_prompt = f"请为以下法律条文生成简短摘要：\n标题：{title}\n内容：{text[:500]}"

            messages = [
                {"role": "system", "content": "你是法律摘要助手。"},
                {"role": "user", "content": user_prompt},
            ]

            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                chat_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024  # 使用较小的长度
            )
            
            device = next(llm.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = llm.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            gen_ids = out[0, input_len:]
            result = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            return result

        except Exception as e:
            print(f"摘要生成失败: {e}")
            return ""
        finally:
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    for i, (title, body) in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {title[:30]}...")
        try:
            result = summarize_simple(title, body)
            print(f"✓ 摘要生成成功: {result[:50]}...")
        except Exception as e:
            print(f"✗ 摘要生成失败: {e}")
            traceback.print_exc()

def test_memory_usage():
    """测试内存使用情况"""
    print("\n=== 测试内存使用 ===")
    
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"设备 {i}: {props.name}, 内存: {props.total_memory / 1024**3:.2f}GB")
        
        # 测试内存分配
        print("\n测试内存分配...")
        initial_memory = torch.cuda.memory_allocated()
        print(f"初始内存使用: {initial_memory / 1024**3:.2f}GB")
        
        # 创建一些张量
        test_tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000).cuda()
            test_tensors.append(tensor)
            current_memory = torch.cuda.memory_allocated()
            print(f"分配 {i+1} 个张量后: {current_memory / 1024**3:.2f}GB")
        
        # 清理
        del test_tensors
        torch.cuda.empty_cache()
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated()
        print(f"清理后内存使用: {final_memory / 1024**3:.2f}GB")
    else:
        print("CUDA不可用，使用CPU模式")

def test_text_parsing():
    """测试文本解析"""
    print("\n=== 测试文本解析 ===")
    
    try:
        txt = IN.read_text(encoding="utf-8", errors="ignore")
        print(f"✓ 文本读取成功，长度: {len(txt)} 字符")
        
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        print(f"✓ 文本预处理成功，有效行数: {len(lines)}")
        
        # 测试正则表达式
        pat_tiao = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+条\b.*$')
        tiao_count = sum(1 for ln in lines if pat_tiao.match(ln))
        print(f"✓ 找到条文数量: {tiao_count}")
        
        # 显示前几个条文
        print("\n前5个条文:")
        count = 0
        for ln in lines:
            if pat_tiao.match(ln) and count < 5:
                print(f"  {count+1}. {ln[:50]}...")
                count += 1
        
        return lines
    except Exception as e:
        print(f"✗ 文本解析失败: {e}")
        traceback.print_exc()
        return []

def main():
    print("开始调试民法典分块处理程序...")
    
    # 1. 测试模型加载
    llm, tokenizer = test_model_loading()
    if llm is None:
        print("模型加载失败，无法继续")
        return
    
    # 2. 测试内存使用
    test_memory_usage()
    
    # 3. 测试文本解析
    lines = test_text_parsing()
    if not lines:
        print("文本解析失败，无法继续")
        return
    
    # 4. 准备测试案例
    pat_tiao = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+条\b.*$')
    test_cases = []
    count = 0
    for i, ln in enumerate(lines):
        if pat_tiao.match(ln) and count < 3:  # 只测试前3个条文
            # 找到条文内容
            body_lines = []
            for j in range(i+1, min(i+10, len(lines))):
                if pat_tiao.match(lines[j]):
                    break
                body_lines.append(lines[j])
            test_cases.append((ln, "\n".join(body_lines)))
            count += 1
    
    # 5. 测试摘要生成
    test_summarize_function(llm, tokenizer, test_cases)
    
    print("\n=== 调试完成 ===")
    print("如果所有测试都通过，可以尝试运行完整版本")
    print("如果出现问题，请检查错误信息并相应调整代码")

if __name__ == "__main__":
    main()