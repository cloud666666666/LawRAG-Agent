# chunk_civil_code_improved.py
# 改进版本：解决内存泄漏、异常处理和进度保存问题
from pathlib import Path
import re, json
import torch
import gc
import time
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
PROGRESS_FILE = Path("data/processed/progress.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 加载模型
print("正在加载模型...")
llm, tokenizer = llm_load()
print("模型加载完成")

# 统一设置 pad_token_id，避免生成时早停/填充异常
if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
LAW_NAME = IN.stem  # 动态：law=输入文件名（去后缀）

# 读入与规范化
print("正在读取和预处理文本...")
txt = IN.read_text(encoding="utf-8", errors="ignore")
txt = txt.replace("\u3000", " ")
txt = re.sub(r"[ \t]+", " ", txt).replace("\r", "")
txt = re.sub(r"\n{2,}", "\n", txt).strip()
# 去"目录"
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

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def save_progress(processed_count, current_bian, current_zhang, current_title, current_body):
    """保存进度"""
    progress = {
        "processed_count": processed_count,
        "current_bian": current_bian,
        "current_zhang": current_zhang,
        "current_title": current_title,
        "current_body": current_body,
        "timestamp": time.time()
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_progress():
    """加载进度"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return None

def summarize_safe(title: str, body: str, limit: int = 80, max_retries: int = 3) -> str:
    """
    安全的摘要生成函数，包含重试机制和错误处理
    """
    for attempt in range(max_retries):
        try:
            return summarize(title, body, limit)
        except Exception as e:
            print(f"摘要生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
                clear_gpu_cache()  # 清理GPU缓存
            else:
                print(f"跳过该条文的摘要生成: {title[:50]}...")
                return "摘要生成失败"
    
    return "摘要生成失败"

def summarize(title: str, body: str, limit: int = 80) -> str:
    """
    使用本地指令模型 + chat template 生成单句要点摘要。
    失败即抛错（不回退）。依赖全局 llm, tokenizer = llm_load()。
    """
    text = (body or title or "").strip()
    if not text:
        raise ValueError("summarize(): empty text")

    # 可选：极长文本做轻截断，避免上下文超限（如不需要可删除）
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

def flush(article_title, article_body, cur_bian, cur_zhang, articles):
    """处理单个条文并写入文件"""
    if not article_title:
        return articles
    
    body = "\n".join(article_body).strip()
    full_text = (article_title + ("\n" + body if body else "")).strip()

    # path（仅 编/章/条）
    m = pat_tiao_no.search(article_title)
    art_cn = clean_no(m.group(1)) if m else None
    parts = []
    if cur_bian:  parts.append(cur_bian)
    if cur_zhang: parts.append(cur_zhang)
    parts.append(f"第{art_cn}条" if art_cn else "未知条")
    path_str = "/".join(parts)

    # 使用安全的摘要生成
    summary_text = summarize_safe(article_title, body)

    obj = {
        "text": full_text,
        "meta": {
            "law": LAW_NAME,          # 这里用 IN.stem
            "path": path_str,         # 形如：第一编/第一章/第一条
            "summary": summary_text
        }
    }

    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    articles += 1
    return articles

def main():
    global cur_bian, cur_zhang, article_title, article_body, articles
    
    # 检查是否有进度文件
    progress = load_progress()
    start_line = 0
    cur_bian = None
    cur_zhang = None
    article_title = None
    article_body = []
    articles = 0
    
    if progress:
        print(f"发现进度文件，从第 {progress['processed_count']} 条开始恢复...")
        start_line = progress.get('start_line', 0)
        cur_bian = progress.get('current_bian')
        cur_zhang = progress.get('current_zhang')
        article_title = progress.get('current_title')
        article_body = progress.get('current_body', [])
        articles = progress.get('processed_count', 0)
    
    # 清空旧文件（仅在从头开始时）
    if not progress and OUT.exists(): 
        OUT.unlink()
    
    print(f"开始处理，从第 {articles} 条开始...")
    
    # 逐行解析（进度条）
    try:
        for i, ln in enumerate(tqdm(lines[start_line:], desc=f"解析 {LAW_NAME}", unit="行", dynamic_ncols=True)):
            line_num = start_line + i
            
            if pat_bian.match(ln):
                cur_bian = ln
                continue
            if pat_zhang.match(ln):
                cur_zhang = ln
                continue
            if pat_tiao.match(ln):
                # 处理前一个条文
                articles = flush(article_title, article_body, cur_bian, cur_zhang, articles)
                # 开始新条文
                article_title = ln
                article_body = []
                
                # 每处理10条保存一次进度
                if articles % 10 == 0:
                    save_progress(articles, cur_bian, cur_zhang, article_title, article_body)
                    clear_gpu_cache()  # 定期清理GPU缓存
                    print(f"已处理 {articles} 条，当前GPU内存: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                
                continue
            if article_title:
                article_body.append(ln)
        
        # 收尾
        articles = flush(article_title, article_body, cur_bian, cur_zhang, articles)
        
        # 删除进度文件
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        
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
        
    except KeyboardInterrupt:
        print("\n程序被用户中断，保存当前进度...")
        save_progress(articles, cur_bian, cur_zhang, article_title, article_body)
        print(f"已保存进度，下次运行将从第 {articles} 条开始")
    except Exception as e:
        print(f"\n程序出现错误: {e}")
        print("保存当前进度...")
        save_progress(articles, cur_bian, cur_zhang, article_title, article_body)
        print(f"已保存进度，下次运行将从第 {articles} 条开始")
        raise

if __name__ == "__main__":
    main()