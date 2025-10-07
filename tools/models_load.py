from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
llm_path="models/qwen2.5-7b-instruct"
embedding_path="models/bge-m3"
rerank_path="models/bge-reranker-v2-m3"
device="cuda:5"
def llm_load():
    """
    加载本地大语言模型
    返回加载后的模型对象
    """
    # 加载本地大语言模型
    model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)
    # 加载本地分词器
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    # 将模型移动到指定设备
    model = model.to(device)

    return model, tokenizer # 只返回模型对象llm，如需分词器可改为 return model, tokenizer 
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