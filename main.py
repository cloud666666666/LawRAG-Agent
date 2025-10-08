#!/usr/bin/env python3
# main.py - LawRAG-Agent主程序，使用LangChain集成所有组件
import os
import sys
import json
import time
import torch
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# LangChain导入
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# 导入自定义模块
from tools.models_load import embedding_load, rerank_load, llm_load

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomEmbeddings(Embeddings):
    """自定义嵌入模型封装，兼容LangChain"""
    
    def __init__(self, model_name: str = "models/bge-m3"):
        """初始化嵌入模型"""
        self.model = embedding_load()
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        embedding = self.model.encode(text)
        return embedding.tolist()

class CustomReranker:
    """自定义重排序模型封装，兼容LangChain"""
    
    def __init__(self):
        """初始化重排序模型"""
        try:
            self.reranker = rerank_load()
        except Exception as e:
            print(f"重排序模型加载失败: {e}")
            self.reranker = None
    
    def compress_documents(
        self, documents: List[Document], query: str
    ) -> List[Document]:
        """重排序文档"""
        if not documents or not self.reranker:
            return documents
        
        # 准备文档对
        pairs = []
        for doc in documents:
            pairs.append([query, doc.page_content])
        
        try:
            # 计算重排序分数
            scores = self.reranker.compute_score(pairs, process_num=1)
            
            # 为文档添加重排序分数
            for i, doc in enumerate(documents):
                doc.metadata["rerank_score"] = float(scores[i])
            
            # 按重排序分数排序
            sorted_docs = sorted(documents, key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
            return sorted_docs
        except Exception as e:
            print(f"重排序过程出错: {e}")
            return documents

class CustomVectorStore:
    """自定义向量存储，加载预构建的FAISS索引"""
    
    def __init__(self, vector_path: str, name: str):
        """初始化向量存储"""
        self.vector_path = vector_path
        self.name = name
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # 加载索引和元数据
        self._load()
    
    def _load(self) -> bool:
        """从磁盘加载向量存储"""
        index_path = os.path.join(self.vector_path, self.name, "index.faiss")
        metadata_path = os.path.join(self.vector_path, self.name, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # 加载索引
            self.index = faiss.read_index(index_path)
            
            # 加载元数据
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                
            self.ids = metadata["ids"]
            self.documents = metadata["documents"]
            self.metadatas = metadata["metadatas"]
            
            return True
        return False
    
    def similarity_search_with_score(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Tuple[Document, float]]:
        """相似度搜索"""
        if self.index is None:
            return []
        
        # 转换查询向量
        query_np = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # 查询FAISS索引
        top_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_np, top_k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # 确保索引有效
                doc = Document(
                    page_content=self.documents[idx],
                    metadata=self.metadatas[idx]
                )
                # FAISS返回的是内积，转换为相似度
                similarity = float(distances[0][i])
                results.append((doc, similarity))
        
        return results

class CustomRetriever:
    """自定义检索器，封装向量存储"""
    
    def __init__(self, vector_store: CustomVectorStore, k: int = 5):
        """初始化检索器"""
        self._vector_store = vector_store
        self._k = k
        self._embeddings = CustomEmbeddings()
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        query_embedding = self._embeddings.embed_query(query)
        docs_and_scores = self._vector_store.similarity_search_with_score(
            query_embedding, k=self._k
        )
        
        # 提取文档并添加分数
        docs = []
        for doc, score in docs_and_scores:
            doc.metadata["similarity"] = score
            docs.append(doc)
        
        return docs

def format_docs(docs: List[Document]) -> str:
    """格式化文档为上下文字符串"""
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        meta = doc.metadata
        law = meta.get("law", "未知法律")
        path = meta.get("path", "")
        doc_type = meta.get("type", "")
        
        # 根据类型添加不同的前缀
        prefix = f"[{law}] "
        if doc_type == "summary":
            prefix += "摘要: "
        
        formatted_docs.append(f"{i}. {prefix}{content}")
    
    return "\n".join(formatted_docs)

def create_law_rag_chain():
    """创建法律RAG链"""
    # 加载向量存储
    vector_path = "data/vectors"
    text_store = CustomVectorStore(vector_path, "text_vector")
    summary_store = CustomVectorStore(vector_path, "summary_vector")
    
    # 创建检索器
    text_retriever = CustomRetriever(text_store, k=5)
    summary_retriever = CustomRetriever(summary_store, k=5)
    
    # 创建重排序器
    reranker = CustomReranker()
    
    # 创建简单检索器列表，不使用EnsembleRetriever
    retrievers = [text_retriever, summary_retriever]
    weights = [0.5, 0.5]  # 权重相等
    
    # 加载大语言模型
    llm, tokenizer = llm_load()
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的中文法律助手。请根据提供的法律条文回答用户的问题。回答应当准确、客观，并明确引用相关法条。"),
        ("user", "请回答以下法律问题，并参考给出的法律条文：\n\n问题：{question}\n\n相关法律条文：\n{context}\n\n请基于上述法律条文回答问题，如果条文不足以回答问题，请明确指出。回答需要客观准确，并引用相关条文。")
    ])
    
    # 创建自定义生成函数
    def generate_answer(prompt_value):
        # 编码输入
        inputs = tokenizer(
            prompt_value.content,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - 512  # 留出生成空间
        )
        device = next(llm.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.inference_mode():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # 解码并提取回答
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_length:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer.strip()
    
    # 创建自定义检索函数
    def retrieve_and_rerank(query):
        # 从两个检索器获取结果
        text_docs = text_retriever._get_relevant_documents(query)
        summary_docs = summary_retriever._get_relevant_documents(query)
        
        # 合并结果
        all_docs = text_docs + summary_docs
        
        # 重排序
        reranked_docs = reranker.compress_documents(all_docs, query)
        
        # 取前5个结果
        return reranked_docs[:5]
    
    # 创建LangChain链
    chain = (
        {"context": lambda x: format_docs(retrieve_and_rerank(x)), "question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(generate_answer)
    )
    
    return chain

def process_query(query: str) -> Dict[str, Any]:
    """处理单个查询"""
    # 创建RAG链
    chain = create_law_rag_chain()
    
    # 执行查询
    start_time = time.time()
    answer = chain.invoke(query)
    elapsed = time.time() - start_time
    
    # 返回结果
    return {
        "query": query,
        "answer": answer,
        "time": elapsed
    }

def process_queries_from_file(file_path: str) -> List[Dict[str, Any]]:
    """从文件处理多个查询"""
    # 读取查询
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    
    # 处理查询
    results = []
    for query in tqdm(queries, desc="处理查询"):
        result = process_query(query)
        results.append(result)
    
    return results

def main():
    """主函数"""
    import time
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        # 从命令行参数获取查询
        query = sys.argv[1]
        result = process_query(query)
        print(f"\n问题: {result['query']}")
        print(f"\n回答: {result['answer']}")
        print(f"\n用时: {result['time']:.2f}秒")
    elif os.path.exists("example_queries.txt"):
        # 从文件处理查询
        print("从文件处理查询...")
        results = process_queries_from_file("example_queries.txt")
        
        # 保存结果
        output_path = "data/results/langchain_answers.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，结果已保存到: {output_path}")
    else:
        # 使用默认查询
        query = "民事行为能力"
        result = process_query(query)
        print(f"\n问题: {result['query']}")
        print(f"\n回答: {result['answer']}")
        print(f"\n用时: {result['time']:.2f}秒")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()