#!/usr/bin/env python3
# chroma_builder.py - 构建向量数据库
import json
import os
import time
import pickle
from pathlib import Path
from tqdm import tqdm
from models_load import embedding_load

# 使用FAISS向量存储，不依赖SQLite
from sentence_transformers import util
import numpy as np
import torch
import faiss

class VectorStore:
    """基于FAISS的向量存储，不依赖SQLite"""
    def __init__(self, name, vector_path):
        self.name = name
        self.vector_path = vector_path
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self.index = None
        self.dimension = None
        
    def add(self, ids, embeddings, documents, metadatas):
        """添加文档"""
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        # 确定维度
        if self.dimension is None and len(embeddings) > 0:
            if isinstance(embeddings[0], list):
                self.dimension = len(embeddings[0])
            else:
                self.dimension = len(embeddings[0].tolist())
                
            # 创建FAISS索引
            self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度（余弦相似度）
        
        # 转换为numpy数组并添加到索引
        if isinstance(embeddings[0], list):
            embeddings_np = np.array(embeddings).astype('float32')
        else:
            embeddings_np = np.array([emb.tolist() for emb in embeddings]).astype('float32')
            
        if self.index is not None:
            self.index.add(embeddings_np)
            
        self.embeddings.extend(embeddings_np)
        
    def save(self):
        """保存向量存储到磁盘"""
        os.makedirs(os.path.join(self.vector_path, self.name), exist_ok=True)
        
        # 保存索引
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.vector_path, self.name, "index.faiss"))
            
        # 保存元数据
        metadata = {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "dimension": self.dimension
        }
        
        with open(os.path.join(self.vector_path, self.name, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
            
    def load(self):
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
            self.dimension = metadata["dimension"]
            
            return True
        return False
        
    def query(self, query_embeddings, n_results=5, include=None):
        """查询文档"""
        if self.index is None or self.index.ntotal == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        # 转换查询向量
        if isinstance(query_embeddings[0], list):
            query_np = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)
        else:
            query_np = np.array(query_embeddings[0].tolist()).astype('float32').reshape(1, -1)
            
        # 查询FAISS索引
        top_k = min(n_results, self.index.ntotal)
        distances, indices = self.index.search(query_np, top_k)
        
        # 构建结果
        result_ids = []
        result_docs = []
        result_meta = []
        result_dist = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.ids):  # 确保索引有效
                result_ids.append(self.ids[idx])
                result_docs.append(self.documents[idx])
                result_meta.append(self.metadatas[idx])
                # FAISS返回的是内积，转换为距离
                result_dist.append(1 - distances[0][i])
            
        return {
            "ids": [result_ids],
            "documents": [result_docs],
            "metadatas": [result_meta],
            "distances": [result_dist]
        }
        
    def count(self):
        """返回文档数量"""
        return len(self.documents)

def main():
    """
    构建向量数据库主函数
    直接运行此脚本即可构建向量数据库
    """
    print("\n=== 开始构建向量数据库 ===")
    
    # 使用固定配置，无需命令行参数
    # 设置路径
    data_path = 'data/processed/chunks_with_summary.jsonl'
    vector_path = 'data/vectors'
    batch_size = 50
    rebuild = True  # 默认重建集合
    
    # 确保向量存储目录存在
    os.makedirs(vector_path, exist_ok=True)
    
    # 加载嵌入模型
    print("加载嵌入模型...")
    embedding_model = embedding_load()
    
    # 创建向量存储
    print(f"创建向量存储: {vector_path}")
    
    # 创建集合
    col_text = VectorStore("text_vector", vector_path)
    col_summary = VectorStore("summary_vector", vector_path)
    
    # 如果不是重建模式，尝试加载现有索引
    if not rebuild:
        text_loaded = col_text.load()
        summary_loaded = col_summary.load()
        if text_loaded and summary_loaded:
            print(f"已加载现有向量存储")
            print(f"文本集合: {col_text.count()} 条记录")
            print(f"摘要集合: {col_summary.count()} 条记录")
            
            # 执行示例查询
            return
    
    # 读取数据
    print(f"读取数据: {data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    total = len(data)
    print(f"共读取 {total} 条记录")
    
    # 检查是否所有记录都有摘要
    missing_summary = [i for i, item in enumerate(data) if "summary" not in item["meta"]]
    if missing_summary:
        print(f"警告: 有 {len(missing_summary)} 条记录缺少摘要")
        print(f"缺少摘要的记录索引: {missing_summary[:10]}...")
    
    # 批量添加到向量数据库
    print("开始构建向量数据库...")
    start_time = time.time()
    
    # 准备批处理
    for i in tqdm(range(0, total, batch_size)):
        batch = data[i:i+batch_size]
        
        # 准备批次数据
        ids = []
        texts = []
        metadatas = []
        
        summary_ids = []
        summaries = []
        summary_metadatas = []
        
        for j, item in enumerate(batch):
            doc_id = f"doc_{i+j}"
            
            # 原文本
            ids.append(doc_id)
            texts.append(item["text"])
            metadatas.append({
                "law": item["meta"].get("law", ""),
                "path": item["meta"].get("path", ""),
                "has_summary": "summary" in item["meta"]
            })
            
            # 摘要
            if "summary" in item["meta"]:
                summary_ids.append(f"sum_{i+j}")
                summaries.append(item["meta"]["summary"])
                summary_metadatas.append({
                    "law": item["meta"].get("law", ""),
                    "path": item["meta"].get("path", ""),
                    "doc_id": doc_id
                })
        
        # 计算嵌入并添加到集合
        if texts:
            # 计算文本嵌入
            text_embeddings = embedding_model.encode(texts)
            col_text.add(
                ids=ids,
                embeddings=text_embeddings,
                documents=texts,
                metadatas=metadatas
            )
        
        if summaries:
            # 计算摘要嵌入
            summary_embeddings = embedding_model.encode(summaries)
            col_summary.add(
                ids=summary_ids,
                embeddings=summary_embeddings,
                documents=summaries,
                metadatas=summary_metadatas
            )
    
    # 保存向量存储
    print("保存向量存储...")
    col_text.save()
    col_summary.save()
    
    # 统计信息
    elapsed = time.time() - start_time
    print(f"向量数据库构建完成!")
    print(f"总耗时: {elapsed:.2f}秒")
    print(f"文本集合: {col_text.count()} 条记录")
    print(f"摘要集合: {col_summary.count()} 条记录")
    
    print("\n=== 向量数据库构建完成 ===")
    print(f"向量数据保存在: {vector_path}")
    print("您可以使用 python tools/batch_query.py 进行查询")


if __name__ == "__main__":
    try:
        # 确保目录存在
        os.makedirs('data/vectors', exist_ok=True)
        os.makedirs('data/results', exist_ok=True)
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()