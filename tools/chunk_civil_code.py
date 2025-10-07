# chunk_civil_code.py
# {"text": fulltext, "meta": {"law": IN.stem, "path": "第一编/第一章/第一条", "summary": "..."}}
from pathlib import Path
import re, json
import torch
from transformers import AutoTokenizer,GenerationConfig
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
llm, tokenizer = llm_load()
# 统一设置 pad_token_id，避免生成时早停/填充异常
if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
LAW_NAME = IN.stem  # 动态：law=输入文件名（去后缀）

# 读入与规范化
txt = IN.read_text(encoding="utf-8", errors="ignore")
txt = txt.replace("\u3000", " ")
txt = re.sub(r"[ \t]+", " ", txt).replace("\r", "")
txt = re.sub(r"\n{2,}", "\n", txt).strip()
# 去“目录”
txt = re.sub(r"目\s*录.*?(?=^第[一二三四五六七八九十百千零\d]+\s*编.*$)", "", txt, flags=re.S|re.M)

lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

# 标题/条文匹配
pat_bian    = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+编\b.*$')
pat_zhang   = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+章\b.*$')
pat_tiao    = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+条\b.*$')
pat_tiao_no = re.compile(r'第\s*([一二三四五六七八九十百千零\d\s]+)\s*条')

def clean_no(s: str) -> str:
    return re.sub(r"\s+", "", s)

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
        f"3) 不复述条号或标题；不使用“可能/建议”等评价词；\n"
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
        max_new_tokens=160,
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

    # 强制“单句”：切到第一个句末标点
    cand = re.split(r"[。！？!？]\s*", cand, maxsplit=1)[0].strip()
    if not cand:
        raise RuntimeError("summarize(): empty after single-sentence enforcement.")

    # 限字
    if len(cand) > limit:
        cand = cand[:limit].rstrip()
        if not cand:
            raise RuntimeError("summarize(): empty after length trimming.")

    return cand


cur_bian = None
cur_zhang = None
article_title = None
article_body  = []
articles = 0

# 清空旧文件
if OUT.exists(): OUT.unlink()

def flush():
    global article_title, article_body, articles, cur_bian, cur_zhang
    if not article_title:
        return
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

    obj = {
        "text": full_text,
        "meta": {
            "law": LAW_NAME,          # 这里用 IN.stem
            "path": path_str,         # 形如：第一编/第一章/第一条
            "summary": summarize(article_title, body)
        }
    }

    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    articles += 1
    article_title, article_body[:] = None, []

# 逐行解析（进度条）
for ln in tqdm(lines, desc=f"解析 {LAW_NAME}", unit="行", dynamic_ncols=True):
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

# 收尾
flush()

# 统计
def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u=="B" else f"{n:.2f} {u}"
        n /= 1024

size = OUT.stat().st_size
print(f"OK -> {OUT} (articles={articles}, chunks={articles}, size={size} bytes, {human(size)})")
