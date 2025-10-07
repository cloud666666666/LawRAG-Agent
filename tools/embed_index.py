# pip install sentence-transformers faiss-cpu
# 读取 data/processed/chunks.jsonl -> 生成 data/index/faiss.index
from pathlib import Path
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

CHUNKS = Path("data/processed/chunks.jsonl")
INDEX  = Path("data/index/faiss.index")
INDEX.parent.mkdir(parents=True, exist_ok=True)

docs  = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
texts = [d["text"] for d in docs]

model = SentenceTransformer("models/bge-m3")  # 本地目录
vecs  = model.encode(texts, batch_size=32, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, str(INDEX))
print("OK ->", INDEX, f"({len(texts)} vectors)")
