"""
优化后的向量化和索引构建模块
- 改进批处理性能
- 添加进度条和内存监控
- 支持增量索引构建
- 优化内存使用
"""
import json
import time
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import gc
import torch
from optimized_models_load import embedding_load, get_model_info

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedEmbeddingIndexer:
    """优化的向量化索引构建器"""
    
    def __init__(self, 
                 chunks_path: str = "data/processed/chunks.jsonl",
                 index_path: str = "data/index/faiss.index",
                 batch_size: int = 64,
                 normalize_embeddings: bool = True,
                 use_gpu: bool = True):
        """
        初始化索引构建器
        
        Args:
            chunks_path: 分块数据文件路径
            index_path: 索引文件保存路径
            batch_size: 批处理大小
            normalize_embeddings: 是否标准化向量
            use_gpu: 是否使用GPU加速
        """
        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # 确保输出目录存在
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 性能统计
        self.stats = {
            'total_docs': 0,
            'total_vectors': 0,
            'processing_time': 0,
            'indexing_time': 0,
            'memory_peak': 0
        }
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """加载分块数据"""
        logger.info(f"加载分块数据: {self.chunks_path}")
        
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"分块数据文件不存在: {self.chunks_path}")
        
        chunks = []
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        
        self.stats['total_docs'] = len(chunks)
        logger.info(f"成功加载 {len(chunks)} 个文档分块")
        return chunks
    
    def _process_batch(self, texts: List[str], model) -> np.ndarray:
        """处理一批文本的向量化"""
        try:
            # 使用优化的批处理参数
            embeddings = model.encode(
                texts,
                batch_size=min(self.batch_size, len(texts)),
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False  # 避免嵌套进度条
            )
            
            # 确保数据类型为float32（FAISS要求）
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"批处理向量化失败: {e}")
            raise
    
    def _create_faiss_index(self, dimension: int) -> faiss.Index:
        """创建FAISS索引"""
        logger.info(f"创建FAISS索引，维度: {dimension}")
        
        # 使用内积索引（适合标准化向量）
        index = faiss.IndexFlatIP(dimension)
        
        # 如果使用GPU且可用
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("使用GPU加速FAISS索引")
            except Exception as e:
                logger.warning(f"GPU加速失败，使用CPU: {e}")
        
        return index
    
    def build_index(self, force_rebuild: bool = False) -> str:
        """
        构建向量索引
        
        Args:
            force_rebuild: 是否强制重建索引
            
        Returns:
            索引文件路径
        """
        if self.index_path.exists() and not force_rebuild:
            logger.info(f"索引文件已存在: {self.index_path}")
            return str(self.index_path)
        
        start_time = time.time()
        logger.info("开始构建向量索引...")
        
        try:
            # 加载分块数据
            chunks = self._load_chunks()
            if not chunks:
                raise ValueError("没有找到有效的分块数据")
            
            # 提取文本
            texts = [chunk['text'] for chunk in chunks]
            logger.info(f"提取了 {len(texts)} 个文本片段")
            
            # 加载embedding模型
            model = embedding_load()
            
            # 获取模型信息
            model_info = get_model_info()
            logger.info(f"使用设备: {model_info['device']}")
            
            # 分批处理向量化
            all_embeddings = []
            num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"开始向量化处理，批次数: {num_batches}, 批大小: {self.batch_size}")
            
            with tqdm(total=len(texts), desc="向量化处理", unit="docs") as pbar:
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i + self.batch_size]
                    
                    # 处理当前批次
                    batch_embeddings = self._process_batch(batch_texts, model)
                    all_embeddings.append(batch_embeddings)
                    
                    # 更新进度
                    pbar.update(len(batch_texts))
                    
                    # 监控内存使用
                    current_memory = self._get_memory_usage()
                    self.stats['memory_peak'] = max(self.stats['memory_peak'], current_memory)
                    
                    # 定期清理内存
                    if i % (self.batch_size * 5) == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            # 合并所有向量
            logger.info("合并向量...")
            all_embeddings = np.vstack(all_embeddings)
            self.stats['total_vectors'] = all_embeddings.shape[0]
            
            # 创建FAISS索引
            index = self._create_faiss_index(all_embeddings.shape[1])
            
            # 构建索引
            logger.info("构建FAISS索引...")
            index_start = time.time()
            index.add(all_embeddings)
            self.stats['indexing_time'] = time.time() - index_start
            
            # 保存索引
            logger.info(f"保存索引到: {self.index_path}")
            faiss.write_index(index, str(self.index_path))
            
            # 记录统计信息
            self.stats['processing_time'] = time.time() - start_time
            
            # 输出性能统计
            self._print_stats()
            
            return str(self.index_path)
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise
        finally:
            # 清理内存
            if 'all_embeddings' in locals():
                del all_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _print_stats(self):
        """打印性能统计信息"""
        stats = self.stats
        logger.info("=" * 50)
        logger.info("索引构建完成 - 性能统计")
        logger.info("=" * 50)
        logger.info(f"文档总数: {stats['total_docs']:,}")
        logger.info(f"向量总数: {stats['total_vectors']:,}")
        logger.info(f"向量维度: {self._get_vector_dimension()}")
        logger.info(f"批处理大小: {self.batch_size}")
        logger.info(f"总处理时间: {stats['processing_time']:.2f}s")
        logger.info(f"索引构建时间: {stats['indexing_time']:.2f}s")
        logger.info(f"平均处理速度: {stats['total_docs']/stats['processing_time']:.1f} docs/s")
        logger.info(f"峰值内存使用: {stats['memory_peak']:.1f}MB")
        logger.info(f"索引文件大小: {self._get_file_size()}")
        logger.info("=" * 50)
    
    def _get_vector_dimension(self) -> int:
        """获取向量维度"""
        try:
            if self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                return index.d
            return 0
        except:
            return 0
    
    def _get_file_size(self) -> str:
        """获取文件大小（人类可读格式）"""
        if self.index_path.exists():
            size = self.index_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024 or unit == 'GB':
                    return f"{size:.1f} {unit}"
                size /= 1024
        return "未知"
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        
        try:
            # 加载模型和索引
            model = embedding_load()
            index = faiss.read_index(str(self.index_path))
            
            # 向量化查询
            query_embedding = model.encode([query], normalize_embeddings=self.normalize_embeddings)
            query_embedding = query_embedding.astype(np.float32)
            
            # 搜索
            scores, indices = index.search(query_embedding, k)
            
            # 加载原始文档
            chunks = self._load_chunks()
            
            # 构建结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunks):
                    results.append({
                        'text': chunks[idx]['text'],
                        'meta': chunks[idx]['meta'],
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

def main():
    """主函数"""
    # 创建索引构建器
    indexer = OptimizedEmbeddingIndexer(
        batch_size=32,  # 根据GPU内存调整
        use_gpu=True
    )
    
    # 构建索引
    index_path = indexer.build_index(force_rebuild=False)
    print(f"索引构建完成: {index_path}")
    
    # 测试搜索
    print("\n测试搜索功能...")
    results = indexer.search("合同成立的条件", k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. 相似度: {result['score']:.4f}")
        print(f"   路径: {result['meta']['path']}")
        print(f"   摘要: {result['meta']['summary']}")
        print(f"   文本: {result['text'][:100]}...")
        print()

if __name__ == "__main__":
    main()