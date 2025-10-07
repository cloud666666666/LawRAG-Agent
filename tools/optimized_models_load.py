"""
优化后的模型加载模块
- 实现单例模式，避免重复加载
- 支持延迟加载和内存管理
- 添加性能监控和缓存机制
"""
import os
import gc
import time
import logging
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器，实现单例模式和延迟加载"""
    
    _instance = None
    _models = {}
    _device = "cuda:5" if torch.cuda.is_available() else "cpu"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.model_paths = {
                'llm': "models/qwen2.5-7b-instruct",
                'embedding': "models/bge-m3", 
                'rerank': "models/bge-reranker-v2-m3"
            }
            self._load_times = {}
            self._memory_usage = {}
    
    def get_device(self) -> str:
        """获取当前设备"""
        return self._device
    
    def set_device(self, device: str):
        """设置设备"""
        self._device = device
        logger.info(f"设备设置为: {device}")
    
    def _log_performance(self, model_name: str, load_time: float, memory_usage: float = None):
        """记录性能指标"""
        self._load_times[model_name] = load_time
        if memory_usage:
            self._memory_usage[model_name] = memory_usage
        logger.info(f"{model_name} 加载耗时: {load_time:.2f}s, 内存使用: {memory_usage:.2f}MB" if memory_usage else f"{model_name} 加载耗时: {load_time:.2f}s")
    
    def _get_memory_usage(self) -> float:
        """获取当前GPU内存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def load_llm(self, force_reload: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载大语言模型（单例模式）"""
        if 'llm' in self._models and not force_reload:
            return self._models['llm']
        
        start_time = time.time()
        logger.info("开始加载LLM模型...")
        
        try:
            # 检查模型路径是否存在
            if not os.path.exists(self.model_paths['llm']):
                raise FileNotFoundError(f"LLM模型路径不存在: {self.model_paths['llm']}")
            
            # 加载模型和分词器
            model = AutoModelForCausalLM.from_pretrained(
                self.model_paths['llm'], 
                trust_remote_code=True,
                torch_dtype=torch.float16,  # 使用半精度减少内存
                device_map="auto" if torch.cuda.is_available() else None
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_paths['llm'])
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 移动到指定设备
            if not torch.cuda.is_available() or self._device != "cpu":
                model = model.to(self._device)
            
            self._models['llm'] = (model, tokenizer)
            
            load_time = time.time() - start_time
            memory_usage = self._get_memory_usage()
            self._log_performance('llm', load_time, memory_usage)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"LLM模型加载失败: {e}")
            raise
    
    def load_embedding(self, force_reload: bool = False) -> SentenceTransformer:
        """加载向量模型（单例模式）"""
        if 'embedding' in self._models and not force_reload:
            return self._models['embedding']
        
        start_time = time.time()
        logger.info("开始加载Embedding模型...")
        
        try:
            # 检查模型路径是否存在
            if not os.path.exists(self.model_paths['embedding']):
                raise FileNotFoundError(f"Embedding模型路径不存在: {self.model_paths['embedding']}")
            
            model = SentenceTransformer(
                self.model_paths['embedding'], 
                device=self._device
            )
            
            self._models['embedding'] = model
            
            load_time = time.time() - start_time
            memory_usage = self._get_memory_usage()
            self._log_performance('embedding', load_time, memory_usage)
            
            return model
            
        except Exception as e:
            logger.error(f"Embedding模型加载失败: {e}")
            raise
    
    def load_rerank(self, force_reload: bool = False) -> FlagReranker:
        """加载重排模型（单例模式）"""
        if 'rerank' in self._models and not force_reload:
            return self._models['rerank']
        
        start_time = time.time()
        logger.info("开始加载Rerank模型...")
        
        try:
            # 检查模型路径是否存在
            if not os.path.exists(self.model_paths['rerank']):
                raise FileNotFoundError(f"Rerank模型路径不存在: {self.model_paths['rerank']}")
            
            model = FlagReranker(self.model_paths['rerank'], device=self._device)
            
            self._models['rerank'] = model
            
            load_time = time.time() - start_time
            memory_usage = self._get_memory_usage()
            self._log_performance('rerank', load_time, memory_usage)
            
            return model
            
        except Exception as e:
            logger.error(f"Rerank模型加载失败: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """卸载指定模型以释放内存"""
        if model_name in self._models:
            del self._models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"已卸载 {model_name} 模型")
    
    def unload_all_models(self):
        """卸载所有模型"""
        self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("已卸载所有模型")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'loaded_models': list(self._models.keys()),
            'device': self._device,
            'load_times': self._load_times,
            'memory_usage': self._memory_usage,
            'cuda_available': torch.cuda.is_available()
        }

# 全局模型管理器实例
model_manager = ModelManager()

# 便捷函数
def llm_load(force_reload: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载LLM模型"""
    return model_manager.load_llm(force_reload)

def embedding_load(force_reload: bool = False) -> SentenceTransformer:
    """加载Embedding模型"""
    return model_manager.load_embedding(force_reload)

def rerank_load(force_reload: bool = False) -> FlagReranker:
    """加载Rerank模型"""
    return model_manager.load_rerank(force_reload)

def unload_model(model_name: str):
    """卸载指定模型"""
    model_manager.unload_model(model_name)

def unload_all_models():
    """卸载所有模型"""
    model_manager.unload_all_models()

def get_model_info() -> Dict[str, Any]:
    """获取模型信息"""
    return model_manager.get_model_info()

# 优化的生成配置缓存
@lru_cache(maxsize=1)
def get_optimized_generation_config(tokenizer) -> GenerationConfig:
    """获取优化的生成配置（缓存）"""
    return GenerationConfig(
        max_new_tokens=100,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.05,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        use_cache=True  # 启用缓存以提高性能
    )

if __name__ == "__main__":
    # 测试模型加载
    print("测试模型加载...")
    
    # 加载LLM
    llm, tokenizer = llm_load()
    print(f"LLM加载完成: {type(llm).__name__}")
    
    # 加载Embedding
    embedding = embedding_load()
    print(f"Embedding加载完成: {type(embedding).__name__}")
    
    # 加载Rerank
    rerank = rerank_load()
    print(f"Rerank加载完成: {type(rerank).__name__}")
    
    # 显示模型信息
    info = get_model_info()
    print(f"模型信息: {info}")