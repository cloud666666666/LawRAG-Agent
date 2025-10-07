from huggingface_hub import snapshot_download
import os

MODELS = [
  "BAAI/bge-m3",                  # Embedding
  "BAAI/bge-reranker-v2-m3",      # Reranker
  "Qwen/Qwen2.5-7B-Instruct"      # LLM
]

out = "models"
os.makedirs(out, exist_ok=True)
for repo in MODELS:
    name = repo.split("/")[-1].lower()
    snapshot_download(repo_id=repo, local_dir=f"{out}/{name}", local_dir_use_symlinks=False, resume_download=True)
    print("✓", repo, "->", f"{out}/{name}")
