from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import os
import torch

llm_path="models/qwen2.5-7b-instruct"
embedding_path="models/bge-m3"
rerank_path="models/bge-reranker-v2-m3"
lora_path="models/qwen2.5-7b-law-lora"
device="cuda:0"

def llm_load(use_lora=True):
    """
    加载本地大语言模型
    参数:
    - use_lora: 是否使用LoRA微调模型
    返回加载后的模型对象和分词器
    """
    # 检查是否使用LoRA微调模型
    if use_lora and os.path.exists(lora_path):
        try:
            from peft import PeftModel, PeftConfig
            print(f"加载LoRA微调模型: {lora_path}")
            
            # 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                llm_path, 
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
            
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
            
            # 加载LoRA权重
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float16,
                device_map=device
            )
            
            print("LoRA微调模型加载成功")
            return model, tokenizer
        except Exception as e:
            print(f"加载LoRA微调模型失败: {e}")
            print("回退到基础模型")
    
    # 加载基础模型
    print(f"加载基础模型: {llm_path}")
    model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)
    # 加载本地分词器
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    # 将模型移动到指定设备
    model = model.to(device)

    return model, tokenizer # 返回模型对象和分词器
def embedding_load():
    """
    加载本地向量模型
    返回加载后的向量模型对象
    """
    model = SentenceTransformer(embedding_path, device=device)
    return model


def rerank_load():
    """
    加载本地重排模型
    返回加载后的重排模型对象
    """
    model = FlagReranker(rerank_path, device=device)
    return model


if __name__ == "__main__":
    print(rerank_load())
    print(embedding_load())
    print(llm_load())