#!/usr/bin/env python3
# batch_query.py - 批处理查询脚本，支持重排序和RRF融合
import json
import os
import pickle
import faiss
import time
import sys
from pathlib import Path
from models_load import embedding_load, rerank_load
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 导入重排序模型修复
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.reranker_fix import apply_fixes

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
        
    def query(self, query_embeddings, n_results=10, include=None):
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

def reciprocal_rank_fusion(results_list, k=60):
    """
    实现Reciprocal Rank Fusion (RRF)算法
    
    参数:
    - results_list: 列表，每个元素是一个包含文档和分数的列表
    - k: RRF公式中的常数，默认为60
    
    返回:
    - 融合后的结果列表，按RRF分数排序
    """
    # 初始化RRF分数字典
    rrf_scores = defaultdict(float)
    
    # 处理每个结果列表
    for results in results_list:
        # 为每个结果分配排名
        for rank, (doc, meta) in enumerate(results):
            # 计算RRF分数: 1/(k + rank)
            doc_key = (doc, meta.get('law', ''), meta.get('path', ''))
            rrf_scores[doc_key] += 1.0 / (k + rank)
    
    # 将结果转换为列表并排序
    fused_results = []
    for (doc, law, path), score in rrf_scores.items():
        meta = {'law': law, 'path': path, 'rrf_score': score}
        fused_results.append((doc, meta))
    
    # 按RRF分数降序排序
    fused_results.sort(key=lambda x: x[1]['rrf_score'], reverse=True)
    
    return fused_results

def main():
    """
    批量查询主函数
    直接运行此脚本即可执行批量查询
    """
    print("\n=== 开始执行批量查询 ===")
    
    # 使用固定配置，无需命令行参数
    class Args:
        def __init__(self):
            self.queries = 'example_queries.txt'
            self.output = 'data/results/query_results.json'
            self.vector_path = 'data/vectors'
            self.top_k = 10
            # 启用重排序功能，这是实验重点
            self.use_rerank = True
            self.rerank_top_k = 5
            # 启用RRF融合
            self.use_rrf = True
            self.rrf_k = 60
    
    args = Args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载向量存储
    vector_path = args.vector_path
    print(f"加载向量存储: {vector_path}")
    
    col_text = VectorStore("text_vector", vector_path)
    col_summary = VectorStore("summary_vector", vector_path)
    
    # 加载向量存储
    text_loaded = col_text.load()
    summary_loaded = col_summary.load()
    
    if not text_loaded or not summary_loaded:
        print("向量存储不存在，请先运行 vectordb_builder.py 构建向量数据库")
        return
        
    print(f"文本集合: {col_text.count()} 条记录")
    print(f"摘要集合: {col_summary.count()} 条记录")
    
    # 加载嵌入模型
    print("加载嵌入模型...")
    embedding_model = embedding_load()
    
    # 加载重排序模型（如果需要）
    reranker = None
    if args.use_rerank:
        try:
            print("加载重排序模型...")
            # 设置环境变量，避免多进程问题已在reranker_fix中处理
            reranker = rerank_load()
            print("重排序模型加载成功")
        except Exception as e:
            print(f"重排序模型加载失败: {e}")
            print("将使用向量检索结果，不进行重排序")
            args.use_rerank = False
    
    # 读取查询
    print(f"读取查询文件: {args.queries}")
    queries = []
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    
    print(f"共读取 {len(queries)} 个查询")
    
    # 处理查询
    results = []
    for i, query in enumerate(tqdm(queries, desc="处理查询")):
        query_result = {"query": query, "results": []}
        start_time = time.time()
        
        # 计算查询向量
        query_embedding = embedding_model.encode(query)
        
        # 查询摘要集合
        summary_results = col_summary.query(
            query_embeddings=[query_embedding],
            n_results=args.top_k
        )
        
        # 查询文本集合
        text_results = col_text.query(
            query_embeddings=[query_embedding],
            n_results=args.top_k
        )
        
        # 准备结果列表
        summary_list = []
        text_list = []
        
        # 添加摘要结果
        for doc, meta, dist in zip(
            summary_results["documents"][0],
            summary_results["metadatas"][0],
            summary_results["distances"][0]
        ):
            meta_copy = meta.copy()
            meta_copy["type"] = "summary"
            meta_copy["initial_score"] = 1 - dist
            summary_list.append((doc, meta_copy))
        
        # 添加文本结果
        for doc, meta, dist in zip(
            text_results["documents"][0],
            text_results["metadatas"][0],
            text_results["distances"][0]
        ):
            meta_copy = meta.copy()
            meta_copy["type"] = "text"
            meta_copy["initial_score"] = 1 - dist
            text_list.append((doc, meta_copy))
        
        # 如果使用重排序
        if args.use_rerank and reranker:
            print(f"对查询 '{query}' 进行重排序...")
            
            # 合并结果用于重排序
            combined_docs = []
            combined_meta = []
            
            # 添加摘要和文本结果
            for doc, meta in summary_list + text_list:
                combined_docs.append(doc)
                combined_meta.append(meta)
            
            # 准备重排序
            pairs = []
            for doc in combined_docs:
                pairs.append([query, doc])
            
            # 执行重排序 - 使用修复后的单进程模式
            try:
                # 重排序计算
                rerank_scores = reranker.compute_score(pairs)
                
                # 更新分数
                for j, score in enumerate(rerank_scores):
                    combined_meta[j]["rerank_score"] = float(score)
                    
                print(f"查询 '{query}' 重排序完成")
            except Exception as e:
                print(f"重排序过程出错: {e}")
                # 使用初始分数作为备选
                for j in range(len(combined_docs)):
                    combined_meta[j]["rerank_score"] = combined_meta[j]["initial_score"]
            
            # 按重排序分数排序
            sorted_results = sorted(zip(combined_docs, combined_meta), 
                                   key=lambda x: x[1]["rerank_score"], 
                                   reverse=True)
            
            # 取前K个结果
            top_results = sorted_results[:args.rerank_top_k]
            
            # 如果使用RRF融合
            if args.use_rrf:
                print(f"对查询 '{query}' 应用RRF融合...")
                
                # 分离重排序后的摘要和文本结果
                reranked_summary = [(doc, meta) for doc, meta in sorted_results if meta["type"] == "summary"]
                reranked_text = [(doc, meta) for doc, meta in sorted_results if meta["type"] == "text"]
                
                # 应用RRF融合
                fused_results = reciprocal_rank_fusion(
                    [reranked_summary, reranked_text], 
                    k=args.rrf_k
                )
                
                # 取前K个融合结果
                top_results = fused_results[:args.rerank_top_k]
                
                # 添加融合标记
                for _, meta in top_results:
                    meta["fusion"] = "rrf"
                
                print(f"查询 '{query}' RRF融合完成")
                
            # 使用融合或重排序结果
            combined_docs = [item[0] for item in top_results]
            combined_meta = [item[1] for item in top_results]
        else:
            # 如果不使用重排序，但使用RRF融合
            if args.use_rrf:
                print(f"对查询 '{query}' 应用RRF融合（无重排序）...")
                
                # 应用RRF融合
                fused_results = reciprocal_rank_fusion(
                    [summary_list, text_list], 
                    k=args.rrf_k
                )
                
                # 取前K个融合结果
                top_results = fused_results[:args.rerank_top_k]
                
                # 添加融合标记
                for _, meta in top_results:
                    meta["fusion"] = "rrf"
                
                combined_docs = [item[0] for item in top_results]
                combined_meta = [item[1] for item in top_results]
                
                print(f"查询 '{query}' RRF融合完成")
            else:
                # 不使用重排序也不使用RRF，简单合并结果
                combined_results = []
                
                # 交替添加摘要和文本结果
                for i in range(max(len(summary_list), len(text_list))):
                    if i < len(summary_list):
                        combined_results.append(summary_list[i])
                    if i < len(text_list):
                        combined_results.append(text_list[i])
                
                # 取前K个结果
                top_results = combined_results[:args.rerank_top_k]
                combined_docs = [item[0] for item in top_results]
                combined_meta = [item[1] for item in top_results]
        
        # 添加到结果
        for doc, meta in zip(combined_docs, combined_meta):
            result_item = {
                "document": doc,
                "metadata": meta
            }
            query_result["results"].append(result_item)
        
        query_result["time"] = time.time() - start_time
        results.append(query_result)
    
    # 保存结果
    print(f"保存结果到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n=== 查询处理完成 ===")
    print(f"共处理了 {len(queries)} 个查询")
    print(f"结果已保存到: {args.output}")

def cleanup_resources():
    """清理资源，避免多进程资源泄漏"""
    try:
        # 清理多进程资源
        import multiprocessing
        if hasattr(multiprocessing, 'resource_tracker'):
            rt = multiprocessing.resource_tracker._resource_tracker
            if rt is not None:
                rt._stop = True  # 停止资源跟踪器
                
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 如果有CUDA，清理CUDA缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
    except:
        pass

if __name__ == "__main__":
    try:
        # 确保目录存在
        os.makedirs('data/results', exist_ok=True)
        
        # 应用重排序模型修复
        print("应用重排序模型修复...")
        apply_fixes()
        
        # 运行主函数
        main()
        
        # 清理资源
        cleanup_resources()
    except Exception as e:
        print(f"\n错误: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
        
        # 即使出错也尝试清理资源
        cleanup_resources()