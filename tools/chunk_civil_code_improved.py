# chunk_civil_code_improved.py
# 改进版本：解决内存泄漏、添加错误处理、断点续传等功能
# {"text": fulltext, "meta": {"law": IN.stem, "path": "第一编/第一章/第一条", "summary": "..."}}
from pathlib import Path
import re, json
import torch
import gc
import logging
from transformers import GenerationConfig
from models_load import llm_load
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_info()  
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chunk_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

IN  = Path("data/origin/民法典.txt")
OUT = Path("data/processed/chunks.jsonl")
CHECKPOINT_FILE = Path("data/processed/checkpoint.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 加载模型
logger.info("正在加载模型...")
llm, tokenizer = llm_load()
# 统一设置 pad_token_id，避免生成时早停/填充异常
if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
LAW_NAME = IN.stem  # 动态：law=输入文件名（去后缀）

# 读入与规范化
logger.info("正在读取和预处理文本...")
txt = IN.read_text(encoding="utf-8", errors="ignore")
txt = txt.replace("\u3000", " ")
txt = re.sub(r"[ \t]+", " ", txt).replace("\r", "")
txt = re.sub(r"\n{2,}", "\n", txt).strip()
# 去"目录"
txt = re.sub(r"目\s*录.*?(?=^第[一二三四五六七八九十百千零\d]+\s*编.*$)", "", txt, flags=re.S|re.M)

lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
logger.info(f"总共 {len(lines)} 行文本")

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
    添加了错误处理和内存管理。
    """
    text = (body or title or "").strip()
    if not text:
        return ""

    try:
        # 可选：极长文本做轻截断，避免上下文超限
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

        # 强制 chat template
        if not hasattr(tokenizer, "apply_chat_template"):
            logger.warning("tokenizer has no apply_chat_template, using fallback")
            return f"法律条文摘要：{title[:50]}..."

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
            logger.warning("apply_chat_template returned empty/invalid text")
            return f"法律条文摘要：{title[:50]}..."

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
                generation_config=GEN_STRICT,
            )

        # 仅解码新生成部分，避免混入提示内容
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, input_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        if not raw:
            logger.warning("LLM returned empty text")
            return f"法律条文摘要：{title[:50]}..."

        # 取末行，清理前缀符号
        cand = raw.splitlines()[-1].strip()
        cand = re.sub(r"^[\-\d\.\)（）\s]+", "", cand)

        # 强制"单句"：切到第一个句末标点
        cand = re.split(r"[。！？!？]\s*", cand, maxsplit=1)[0].strip()
        if not cand:
            logger.warning("empty after single-sentence enforcement")
            return f"法律条文摘要：{title[:50]}..."

        # 限字
        if len(cand) > limit:
            cand = cand[:limit].rstrip()
            if not cand:
                logger.warning("empty after length trimming")
                return f"法律条文摘要：{title[:50]}..."

        return cand

    except Exception as e:
        logger.error(f"摘要生成失败: {e}")
        return f"法律条文摘要：{title[:50]}..."
    finally:
        # 清理内存
        if 'inputs' in locals():
            del inputs
        if 'out' in locals():
            del out
        if 'gen_ids' in locals():
            del gen_ids
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def load_checkpoint():
    """加载检查点"""
    if CHECKPOINT_FILE.exists():
        try:
            with CHECKPOINT_FILE.open('r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            logger.info(f"从检查点恢复: 已处理 {checkpoint.get('articles', 0)} 条")
            return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
    return {"articles": 0, "cur_bian": None, "cur_zhang": None, "processed_lines": 0}

def save_checkpoint(articles, cur_bian, cur_zhang, processed_lines):
    """保存检查点"""
    checkpoint = {
        "articles": articles,
        "cur_bian": cur_bian,
        "cur_zhang": cur_zhang,
        "processed_lines": processed_lines
    }
    try:
        with CHECKPOINT_FILE.open('w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存检查点失败: {e}")

def flush(article_title, article_body, articles, cur_bian, cur_zhang):
    """处理单个条文"""
    if not article_title:
        return articles, None, []
    
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

    try:
        summary_text = summarize(article_title, body)
    except Exception as e:
        logger.error(f"摘要生成失败: {e}")
        summary_text = ""

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
    if articles % 10 == 0:  # 每10条保存一次检查点
        logger.info(f"已处理 {articles} 条")
    
    return articles, None, []

def main():
    """主处理函数"""
    # 加载检查点
    checkpoint = load_checkpoint()
    start_articles = checkpoint.get("articles", 0)
    cur_bian = checkpoint.get("cur_bian")
    cur_zhang = checkpoint.get("cur_zhang")
    start_line = checkpoint.get("processed_lines", 0)
    
    article_title = None
    article_body = []
    articles = start_articles

    # 如果从检查点恢复，清空输出文件重新开始
    if start_articles > 0:
        logger.info("从检查点恢复，清空输出文件重新开始")
        if OUT.exists(): 
            OUT.unlink()

    logger.info(f"开始处理，从第 {start_line} 行开始，已处理 {start_articles} 条")

    try:
        # 逐行解析（进度条）
        for i, ln in enumerate(tqdm(lines[start_line:], desc=f"解析 {LAW_NAME}", unit="行", dynamic_ncols=True, initial=start_line, total=len(lines))):
            current_line = start_line + i
            
            if pat_bian.match(ln):
                cur_bian = ln
                continue
            if pat_zhang.match(ln):
                cur_zhang = ln
                continue
            if pat_tiao.match(ln):
                articles, article_title, article_body = flush(article_title, article_body, articles, cur_bian, cur_zhang)
                article_title = ln
                article_body = []
                continue
            if article_title:
                article_body.append(ln)
            
            # 每100行保存一次检查点
            if current_line % 100 == 0:
                save_checkpoint(articles, cur_bian, cur_zhang, current_line + 1)

        # 收尾
        articles, _, _ = flush(article_title, article_body, articles, cur_bian, cur_zhang)

        # 清理检查点文件
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("处理完成，清理检查点文件")

        # 统计
        def human(n):
            for u in ["B","KB","MB","GB","TB"]:
                if n < 1024 or u == "TB":
                    return f"{n:.0f} {u}" if u=="B" else f"{n:.2f} {u}"
                n /= 1024

        size = OUT.stat().st_size
        logger.info(f"处理完成 -> {OUT} (articles={articles}, chunks={articles}, size={size} bytes, {human(size)})")
        print(f"OK -> {OUT} (articles={articles}, chunks={articles}, size={size} bytes, {human(size)})")

    except KeyboardInterrupt:
        logger.info("用户中断，保存检查点...")
        save_checkpoint(articles, cur_bian, cur_zhang, start_line + len([l for l in lines[start_line:] if l.strip()]))
        logger.info(f"已保存检查点，已处理 {articles} 条")
        raise
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        save_checkpoint(articles, cur_bian, cur_zhang, start_line + len([l for l in lines[start_line:] if l.strip()]))
        logger.info(f"已保存检查点，已处理 {articles} 条")
        raise

if __name__ == "__main__":
    main()