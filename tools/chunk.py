# chunk_civil_code.py
# {"text": fulltext, "meta": {"law": IN.stem, "path": "第一编/第一章/第一条", "summary": "..."}}
from pathlib import Path
import re, json
# 不再需要导入模型加载模块
# 不再需要transformers相关导入
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable
IN  = Path("data/origin/民法典.txt")
OUT = Path("data/processed/chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
# 不再需要加载模型，移至generate_summaries.py
LAW_NAME = IN.stem  # 动态：law=输入文件名（去后缀）

# 读入与规范化
txt = IN.read_text(encoding="utf-8", errors="ignore")
txt = txt.replace("\u3000", " ")
txt = re.sub(r"[ \t]+", " ", txt).replace("\r", "")
txt = re.sub(r"\n{2,}", "\n", txt).strip()
# 去"目录"
txt = re.sub(r"目\s*录.*?(?=^第[一二三四五六七八九十百千零\d]+\s*编.*$)", "", txt, flags=re.S|re.M)

lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

# 标题/条文匹配
pat_bian    = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+编\b.*$')
pat_zhang   = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+章\b.*$')
pat_tiao    = re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+条\b.*$')
pat_tiao_no = re.compile(r'第\s*([一二三四五六七八九十百千零\d\s]+)\s*条')

def clean_no(s: str) -> str:
    return re.sub(r"\s+", "", s)

# summarize函数已移至generate_summaries.py


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
            "summary": ""            # 先留空，后续批量补充
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
print(f"解析完成 -> {OUT} (articles={articles}, chunks={articles}, size={size} bytes, {human(size)})")

# 不在这里生成摘要，改为使用单独的脚本
print("解析完成,如需生成摘要请移至generate_summaries.py")