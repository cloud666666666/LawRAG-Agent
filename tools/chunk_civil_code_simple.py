# chunk_civil_code_simple.py
# 简化版本：专注于解决内存泄漏问题
from pathlib import Path
import re, json
import torch
import gc
from transformers import GenerationConfig
from models_load import llm_load
from transformers.utils import logging
logging.set_verbosity_info()  
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

IN  = Path("data/origin/民法典.txt")
OUT = Path("data/processed/chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 加载模型
print("正在加载模型...")
llm, tokenizer = llm_load()
print("模型加载完成")

# 统一设置 pad_token_id
if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
LAW_NAME = IN.stem

# 读入与规范化
print("正在读取和预处理文本...")
txt = IN.read_text(encoding="utf-8", errors="ignore")
txt = txt.replace("\u3000", " ")
txt = re.sub(r"[ \t]+", " ", txt).replace("\r", "")
txt = re.sub(r"\n{2,}", "\n", txt).strip()
txt = re.sub(r"目\s*录.*?(?=^第[一二三四五六七八九十百千零\d]+\s*编.*$)", "", txt, flags=re.S|re.M)

lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
print(f"共读取 {len(lines)} 行文本")

# 标题/条文匹配
pat_bian    = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+编\b.*$')
pat_zhang   = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+章\b.*$')
pat_tiao    = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+条\b.*$')
pat_tiao_no = re.compile(r'第\s*([一二三四五六七八九十百千零\d\s]+)\s*条')

def clean_no(s: str) -> str:
    return re.sub(r"\s+", "", s)

def clear_gpu_memory():
    """强制清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def summarize_with_memory_management(title: str, body: str, limit: int = 80) -> str:
    """
    带内存管理的摘要生成函数
    """
    try:
        text = (body or title or "").strip()
        if not text:
            return ""

        # 文本截断
        if len(text) > 4000:
            text_for_llm = text[:3000] + "\n……\n" + text[-800:]
        else:
            text_for_llm = text

        user_prompt = (
            f"你是一名严谨的中文法律助理。仅依据【正文】提炼**单句**要点摘要，要求：\n"
            f"1) 客观中立，不新增原文之外的事实或数值；\n"
            f"2) 优先保留义务/权利/适用范围/例外等关键信息；\n"
            f"3) 不复述条号或标题；不使用"可能/建议"等评价词；\n"
            f"4) 摘要长度≤{limit}字，且为**单句**。\n\n"
            f"【标题】\n{title}\n\n【正文】\n{text_for_llm}\n"
        )

        messages = [
            {"role": "system", "content": "你是严谨的中文法律摘要助手。"},
            {"role": "user", "content": user_prompt},
        ]

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 使用更短的max_length来减少内存使用
        inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # 减少到2048
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
                generation_config=GEN_STRICT,
            )

        # 解码
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, input_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        if not raw:
            return ""

        # 处理结果
        cand = raw.splitlines()[-1].strip()
        cand = re.sub(r"^[\-\d\.\)（）\s]+", "", cand)
        cand = re.split(r"[。！？!？]\s*", cand, maxsplit=1)[0].strip()
        
        if not cand:
            return ""

        if len(cand) > limit:
            cand = cand[:limit].rstrip()

        return cand

    except Exception as e:
        print(f"摘要生成失败: {e}")
        return ""
    finally:
        # 强制清理内存
        clear_gpu_memory()

# 主处理逻辑
cur_bian = None
cur_zhang = None
article_title = None
article_body = []
articles = 0

# 清空旧文件
if OUT.exists(): 
    OUT.unlink()

def flush():
    global article_title, article_body, articles, cur_bian, cur_zhang
    if not article_title:
        return
    
    body = "\n".join(article_body).strip()
    full_text = (article_title + ("\n" + body if body else "")).strip()

    # 生成路径
    m = pat_tiao_no.search(article_title)
    art_cn = clean_no(m.group(1)) if m else None
    parts = []
    if cur_bian:  parts.append(cur_bian)
    if cur_zhang: parts.append(cur_zhang)
    parts.append(f"第{art_cn}条" if art_cn else "未知条")
    path_str = "/".join(parts)

    # 生成摘要（带内存管理）
    summary_text = summarize_with_memory_management(article_title, body)

    obj = {
        "text": full_text,
        "meta": {
            "law": LAW_NAME,
            "path": path_str,
            "summary": summary_text
        }
    }

    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    articles += 1
    article_title, article_body[:] = None, []

print("开始处理条文...")

# 逐行解析
for i, ln in enumerate(tqdm(lines, desc=f"解析 {LAW_NAME}", unit="行", dynamic_ncols=True)):
    try:
        if pat_bian.match(ln):
            cur_bian = ln
            continue
        if pat_zhang.match(ln):
            cur_zhang = ln
            continue
        if pat_tiao.match(ln):
            flush()
            article_title = ln
            article_body = []
            continue
        if article_title:
            article_body.append(ln)
        
        # 每处理20条清理一次内存
        if i % 20 == 0:
            clear_gpu_memory()
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"已处理 {articles} 条，GPU内存使用: {memory_used:.2f}GB")
    
    except Exception as e:
        print(f"处理第 {i} 行时出错: {e}")
        continue

# 收尾
flush()

# 最终清理
clear_gpu_memory()

# 统计
def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u=="B" else f"{n:.2f} {u}"
        n /= 1024

size = OUT.stat().st_size
print(f"处理完成！")
print(f"输出文件: {OUT}")
print(f"处理条文数: {articles}")
print(f"文件大小: {size} bytes ({human(size)})")