# pip install pdfplumber
from pathlib import Path
import re, pdfplumber

for pdf in Path("data/origin").glob("*.pdf"):
    with pdfplumber.open(pdf) as doc:
        text = "\n".join((p.extract_text() or "") for p in doc.pages)

    # 基础清洗：页码/目录/空白
    text = re.sub(r"[—–\-－]\s*\d+\s*[—–\-－]", "", text)                    # 去 “— 12 —”
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.M)                     # 去纯数字行
    text = re.sub(r"目\s*录.*?(?=^第[一二三四五六七八九十百]+编.*$)", "", text,
                  flags=re.S|re.M)                                          # 去目录段
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text).replace("\r","")
    text = re.sub(r"\n{2,}", "\n", text).strip()

    out = pdf.with_suffix(".txt")
    out.write_text(text, encoding="utf-8")
    print("OK ->", out)
