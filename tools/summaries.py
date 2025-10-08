#!/usr/bin/env python3
# generate_summaries.py - 单独的摘要生成脚本
import json
import argparse
import torch
import os
import signal
import time
import traceback
import psutil
import sys
from pathlib import Path
from transformers import AutoTokenizer, GenerationConfig
from models_load import llm_load
import re
import gc

def summarize(title: str, body: str, limit: int = 50) -> str:
    """
    使用本地指令模型 + chat template 生成单句要点摘要。
    失败即抛错（不回退）。依赖全局 llm, tokenizer = llm_load()。
    """
    text = (body or title or "").strip()
    if not text:
        raise ValueError("summarize(): empty text")

    # 保守地截断文本
    if len(text) > 4000:
        text_for_llm = text[:1500] + "\n……\n" + text[-400:]
    elif len(text) > 2000:
        text_for_llm = text[:1000] + "\n……\n" + text[-300:]
    else:
        text_for_llm = text

    user_prompt = "你是一名严谨的中文法律助理。仅依据【正文】提炼**单句**要点摘要，要求：\n" + \
        "1) 客观中立，不新增原文之外的事实或数值；\n" + \
        "2) 优先保留义务/权利/适用范围/例外等关键信息；\n" + \
        "3) 不复述条号或标题；不使用可能/建议等评价词；\n" + \
        f"4) 摘要长度≤{limit}字，且为**单句**。\n\n" + \
        f"【标题】\n{title}\n\n【正文】\n{text_for_llm}\n"

    # —— 强制 chat template —— #
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("summarize(): tokenizer has no apply_chat_template; chat template is required.")

    messages = [
        {"role": "system", "content": "你是严谨的中文法律摘要助手。"},
        {"role": "user", "content": user_prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not chat_text or not isinstance(chat_text, str):
        raise RuntimeError("summarize(): apply_chat_template returned empty/invalid text.")

    # 截断到模型允许的最大长度，避免上下文溢出
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(tokenizer, "model_max_length", 4096)
    )
    device = next(llm.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    GEN_STRICT = GenerationConfig(
        do_sample=False,
        max_new_tokens=64,
        repetition_penalty=1.05,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        use_cache=True
    )

    with torch.inference_mode():
        out = llm.generate(
            **inputs,
            generation_config=GEN_STRICT,  # ← 关键
        )

    # 仅解码新生成部分，避免混入提示内容
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, input_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("summarize(): LLM returned empty text.")

    # 取末行，清理前缀符号
    cand = raw.splitlines()[-1].strip()
    cand = re.sub(r"^[\-\d\.\)（）\s]+", "", cand)

    # 强制"单句"：切到第一个句末标点
    cand = re.split(r"[。！？!？]\s*", cand, maxsplit=1)[0].strip()
    if not cand:
        raise RuntimeError("summarize(): empty after single-sentence enforcement.")

    # 限字
    if len(cand) > limit:
        cand = cand[:limit].rstrip()
        if not cand:
            raise RuntimeError("summarize(): empty after length trimming.")

    return cand

def process_chunk(chunk_idx, chunk_data, timeout=300):
    """处理单个chunk，生成摘要，带超时控制"""
    import threading
    import queue
    
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def worker():
        try:
            # 分离标题和正文
            text = chunk_data["text"]
            parts = text.split("\n", 1)
            title = parts[0]
            body = parts[1] if len(parts) > 1 else ""
            
            # 生成摘要
            summary = summarize(title, body)
            chunk_data["meta"]["summary"] = summary
            result_queue.put(chunk_data)
        except Exception as e:
            error_msg = f"警告: 条目 {chunk_idx} 摘要生成失败: {e}\n{traceback.format_exc()}"
            error_queue.put((e, error_msg))
    
    # 创建并启动工作线程
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # 等待线程完成或超时
    thread.join(timeout)
    
    if thread.is_alive():
        # 超时，记录错误并返回原始数据
        print(f"错误: 条目 {chunk_idx} 处理超时（{timeout}秒）")
        # 无法直接终止线程，但我们可以不使用其结果
        chunk_data["meta"]["summary_error"] = f"处理超时（{timeout}秒）"
        return chunk_data
    
    # 检查是否有错误
    if not error_queue.empty():
        _, error_msg = error_queue.get()
        print(error_msg)
        chunk_data["meta"]["summary_error"] = str(error_msg)
        return chunk_data
    
    # 获取结果
    if not result_queue.empty():
        return result_queue.get()
    
    # 如果既没有结果也没有错误（不应该发生），返回原始数据
    print(f"警告: 条目 {chunk_idx} 处理结果异常（既无结果也无错误）")
    chunk_data["meta"]["summary_error"] = "处理结果异常"
    return chunk_data

# 删除断点续传相关功能

def print_memory_usage():
    """打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # 系统内存
    system_memory = psutil.virtual_memory()
    
    # CUDA内存
    if torch.cuda.is_available():
        cuda_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        cuda_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        cuda_max = torch.cuda.max_memory_allocated() / (1024 ** 3)    # GB
        cuda_info = f"CUDA: 已分配={cuda_allocated:.2f}GB, 已保留={cuda_reserved:.2f}GB, 峰值={cuda_max:.2f}GB"
    else:
        cuda_info = "CUDA: 不可用"
    
    print(f"内存使用情况: RSS={memory_info.rss/(1024**3):.2f}GB, VMS={memory_info.vms/(1024**3):.2f}GB")
    print(f"系统内存: 已用={system_memory.percent}%, 可用={system_memory.available/(1024**3):.2f}GB")
    print(cuda_info)

def reset_cuda_device():
    """重置CUDA设备，彻底清理显存"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 强制同步CUDA流，确保所有操作完成
        torch.cuda.synchronize()

def handle_signal(signum, frame):
    """处理信号（如Ctrl+C）"""
    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM",
        signal.SIGABRT: "SIGABRT",
        signal.SIGSEGV: "SIGSEGV",
        signal.SIGFPE: "SIGFPE",
        signal.SIGILL: "SIGILL",
        signal.SIGQUIT: "SIGQUIT"
    }
    signal_name = signal_names.get(signum, f"未知信号 {signum}")
    print(f"\n收到中断信号 {signal_name}，正在安全退出...")
    
    # 打印当前堆栈跟踪，帮助诊断中断位置
    import traceback
    print("中断时的堆栈跟踪:")
    traceback.print_stack(frame)
    
    # 打印当前内存状态
    print_memory_usage()
    
    global interrupted
    interrupted = True

def safe_main():
    """包装主函数，捕获所有异常"""
    try:
        return main()
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        return 1
    except Exception as e:
        print(f"\n发生严重错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(description='为法律文本生成摘要')
    parser.add_argument('--input', type=str, default='data/processed/chunks.jsonl',
                        help='输入的JSONL文件路径')
    parser.add_argument('--output', type=str, default='data/processed/chunks_with_summary.jsonl',
                        help='输出的JSONL文件路径')
    parser.add_argument('--start', type=int, default=0,
                        help='起始索引（从0开始）')
    parser.add_argument('--count', type=int, default=None,
                        help='处理条目数量，不指定则处理全部')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='批处理大小，每处理这么多条记录保存一次')
    parser.add_argument('--reset-interval', type=int, default=100,
                        help='每处理多少条记录重置一次CUDA环境（默认100条）')
    parser.add_argument('--timeout', type=int, default=300,
                        help='单条记录处理超时时间（秒），默认300秒')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # 设置信号处理
    global interrupted
    interrupted = False
    
    # 注册更多信号处理器，捕获各种可能的中断信号
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGABRT, 
                signal.SIGSEGV, signal.SIGFPE, signal.SIGILL]:
        try:
            signal.signal(sig, handle_signal)
        except (ValueError, OSError) as e:
            print(f"无法注册信号 {sig}: {e}")
    
    # 创建日志文件
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"summary_generation_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"日志将保存到: {log_file}")
    
    # 记录进程ID，便于后续分析
    print(f"进程ID: {os.getpid()}")
    
    # 使用指定的起始索引
    start_idx = args.start
    print(f"从索引 {start_idx} 开始处理")
    
    # 设置日志重定向
    import sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    
    # 创建双输出流
    class TeeOutput:
        def __init__(self, file, original):
            self.file = file
            self.original = original
        
        def write(self, message):
            self.file.write(message)
            self.original.write(message)
            self.file.flush()  # 确保立即写入文件
            
        def flush(self):
            self.file.flush()
            self.original.flush()
    
    # 重定向标准输出和错误输出
    sys.stdout = TeeOutput(log_file_handle, original_stdout)
    sys.stderr = TeeOutput(log_file_handle, original_stderr)
    
    # 记录系统信息
    print("\n" + "="*50)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"系统信息: {os.uname() if hasattr(os, 'uname') else 'N/A'}")
    print(f"Python版本: {sys.version}")
    print("="*50 + "\n")
    
    # 加载模型
    global llm, tokenizer
    try:
        print("开始加载模型...")
        llm, tokenizer = llm_load()
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        traceback.print_exc()
        return 1
    
    # 统一设置 pad_token_id，避免生成时早停/填充异常
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.count is not None:
        print(f"处理范围: {start_idx} 到 {start_idx + args.count - 1}")
    else:
        print(f"处理范围: 从索引 {start_idx} 开始处理全部记录")
    
    # 读取输入文件
    total_count = 0
    with input_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if args.count is not None and total_count >= args.count:
                break
            total_count += 1
    
    print(f"共找到 {total_count} 条记录待处理")
    
    # 分批处理
    batch_size = args.batch_size
    processed_count = 0
    batch_count = 0
    
    # 清空输出文件
    with output_path.open('w', encoding='utf-8') as out_f:
        pass  # 清空文件内容
    
    # 处理数据
    with input_path.open('r', encoding='utf-8') as f:
        batch_results = []
        
        for i, line in enumerate(f):
            if i < start_idx:
                continue
                
            if args.count is not None and processed_count >= args.count:
                break
                
            if interrupted:
                print("检测到中断信号，正在保存当前批次并退出...")
                break
                
            try:
                chunk_data = json.loads(line)
                print(f"处理条目 {i}...")
                
                # 定期打印内存使用情况
                if processed_count % 10 == 0:
                    print_memory_usage()
                
                # 处理当前chunk，带超时控制
                result = process_chunk(i, chunk_data, timeout=args.timeout)
                batch_results.append(result)
                processed_count += 1
                
                # 每处理一个就清理内存
                gc.collect()
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 定期重置CUDA环境，彻底清理显存
                if processed_count % args.reset_interval == 0:
                    print(f"执行第 {processed_count} 条记录后的CUDA环境重置...")
                    reset_cuda_device()
                
                # 批量保存
                if len(batch_results) >= batch_size:
                    batch_count += 1
                    print(f"保存批次 {batch_count}（{len(batch_results)}条记录）...")
                    
                    # 写入结果 - 使用追加模式，因为前面已经清空了文件
                    with output_path.open('a', encoding='utf-8') as out_f:
                        for res in batch_results:
                            out_f.write(json.dumps(res, ensure_ascii=False) + '\n')
                    
                    # 清空批次
                    batch_results = []
                    print(f"已处理 {processed_count}/{total_count} 条记录")
                    
            except Exception as e:
                print(f"处理条目 {i} 时出错: {e}")
        
        # 保存最后一批
        if batch_results:
            print(f"保存最后一批（{len(batch_results)}条记录）...")
            with output_path.open('a', encoding='utf-8') as out_f:
                for res in batch_results:
                    out_f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    # 恢复标准输出
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
    
    if interrupted:
        print(f"已中断! 成功处理并保存了 {processed_count} 条记录到 {output_path}")
        print(f"详细日志已保存到 {log_file}")
    else:
        print(f"完成! 处理了 {processed_count} 条记录，结果保存到 {output_path}")
        print(f"详细日志已保存到 {log_file}")

if __name__ == "__main__":
    try:
        # 安装必要的依赖
        import importlib
        try:
            importlib.import_module('psutil')
        except ImportError:
            print("安装必要的依赖: psutil")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            
        # 设置CUDA错误处理
        if torch.cuda.is_available():
            # 启用CUDA异常的详细信息
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            # 设置CUDA OOM行为 - 尝试先清理缓存而不是直接崩溃
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # 增加CUDA错误监控
            previous_device = torch.cuda.current_device()
            try:
                # 测试CUDA设备可用性
                torch.cuda.synchronize()
                print(f"CUDA设备 {previous_device} 正常")
            except Exception as e:
                print(f"CUDA设备 {previous_device} 测试失败: {e}")
                print("尝试重置CUDA设备...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        # 打印系统信息
        print(f"Python 版本: {sys.version}")
        print(f"PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        
        # 运行主函数
        sys.exit(safe_main())
    except Exception as e:
        print(f"启动失败: {e}")
        traceback.print_exc()
        sys.exit(1)
