"""
优化后的主程序
- 整合所有性能优化功能
- 实现内存管理和资源清理
- 添加性能监控
- 提供配置选项
"""
import os
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# 导入优化模块
from tools.optimized_models_load import model_manager, get_model_info
from tools.optimized_embed_index import OptimizedEmbeddingIndexer
from tools.optimized_chunk_civil_code import OptimizedCivilCodeChunker
from tools.performance_monitor import PerformanceMonitor, ModelPerformanceProfiler, create_performance_report

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedApp:
    """优化的应用程序主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化应用程序
        
        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        
        # 初始化性能监控
        self.monitor = PerformanceMonitor("performance_log.json")
        self.profiler = ModelPerformanceProfiler(self.monitor)
        
        # 初始化组件
        self.chunker = None
        self.indexer = None
        self.models_loaded = False
        
        # 设置设备
        device = self.config.get('device', 'cuda:5' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        model_manager.set_device(device)
        
        logger.info(f"应用程序初始化完成，使用设备: {device}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'device': 'cuda:5',
            'batch_size': 32,
            'summary_limit': 80,
            'max_text_length': 4000,
            'chunk_batch_size': 1,
            'use_gpu': True,
            'force_rebuild': False,
            'enable_caching': True,
            'memory_limit_mb': 8000
        }
    
    def load_models(self, force_reload: bool = False):
        """加载所有模型"""
        if self.models_loaded and not force_reload:
            logger.info("模型已加载，跳过重复加载")
            return
        
        logger.info("开始加载模型...")
        start_time = time.time()
        
        try:
            # 加载LLM模型
            logger.info("加载LLM模型...")
            self.profiler.profile_model_loading(
                "llm",
                model_manager.load_llm,
                force_reload
            )
            
            # 加载Embedding模型
            logger.info("加载Embedding模型...")
            self.profiler.profile_model_loading(
                "embedding",
                model_manager.load_embedding,
                force_reload
            )
            
            # 加载Rerank模型
            logger.info("加载Rerank模型...")
            self.profiler.profile_model_loading(
                "rerank",
                model_manager.load_rerank,
                force_reload
            )
            
            self.models_loaded = True
            load_time = time.time() - start_time
            logger.info(f"所有模型加载完成，耗时: {load_time:.2f}s")
            
            # 记录模型信息
            model_info = get_model_info()
            logger.info(f"模型信息: {model_info}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def process_civil_code(self, force_rebuild: bool = False):
        """处理民法典文件"""
        logger.info("开始处理民法典文件...")
        
        try:
            # 创建分块处理器
            self.chunker = OptimizedCivilCodeChunker(
                input_path=self.config['input_path'],
                output_path=self.config['output_path'],
                summary_limit=self.config['summary_limit'],
                max_text_length=self.config['max_text_length'],
                batch_size=self.config['chunk_batch_size']
            )
            
            # 处理文件
            output_path = self.chunker.process(force_rebuild=force_rebuild)
            logger.info(f"民法典处理完成: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"民法典处理失败: {e}")
            raise
    
    def build_index(self, force_rebuild: bool = False):
        """构建向量索引"""
        logger.info("开始构建向量索引...")
        
        try:
            # 创建索引构建器
            self.indexer = OptimizedEmbeddingIndexer(
                chunks_path=self.config['output_path'],
                index_path=self.config['index_path'],
                batch_size=self.config['batch_size'],
                use_gpu=self.config['use_gpu']
            )
            
            # 构建索引
            index_path = self.indexer.build_index(force_rebuild=force_rebuild)
            logger.info(f"向量索引构建完成: {index_path}")
            
            return index_path
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise
    
    def search(self, query: str, k: int = 5):
        """搜索相似文档"""
        if not self.indexer:
            logger.error("索引未构建，请先运行 build_index()")
            return None
        
        logger.info(f"搜索查询: {query}")
        
        try:
            results = self.indexer.search(query, k=k)
            logger.info(f"找到 {len(results)} 个相关文档")
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def cleanup(self):
        """清理资源"""
        logger.info("开始清理资源...")
        
        try:
            # 卸载所有模型
            model_manager.unload_all_models()
            
            # 清理缓存
            if hasattr(self.chunker, '_summary_cache'):
                self.chunker._summary_cache.clear()
            
            # 保存性能指标
            self.monitor.save_metrics()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")
    
    def run_full_pipeline(self, force_rebuild: bool = False):
        """运行完整管道"""
        logger.info("开始运行完整管道...")
        
        try:
            # 1. 加载模型
            self.load_models()
            
            # 2. 处理民法典
            self.process_civil_code(force_rebuild=force_rebuild)
            
            # 3. 构建索引
            self.build_index(force_rebuild=force_rebuild)
            
            # 4. 测试搜索
            test_queries = [
                "合同成立的条件",
                "违约责任",
                "物权保护",
                "婚姻家庭关系"
            ]
            
            logger.info("测试搜索功能...")
            for query in test_queries:
                results = self.search(query, k=3)
                if results:
                    logger.info(f"查询 '{query}' 找到 {len(results)} 个结果")
                    for i, result in enumerate(results[:2], 1):
                        logger.info(f"  {i}. {result['meta']['path']} (相似度: {result['score']:.4f})")
            
            logger.info("完整管道运行完成")
            
        except Exception as e:
            logger.error(f"管道运行失败: {e}")
            raise
        finally:
            # 清理资源
            self.cleanup()
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        return create_performance_report(self.monitor)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化的民法典处理应用程序")
    parser.add_argument("--device", default="cuda:5", help="设备 (cuda:5, cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建")
    parser.add_argument("--input-path", default="data/origin/民法典.txt", help="输入文件路径")
    parser.add_argument("--output-path", default="data/processed/chunks.jsonl", help="输出文件路径")
    parser.add_argument("--index-path", default="data/index/faiss.index", help="索引文件路径")
    parser.add_argument("--mode", choices=["full", "chunk", "index", "search"], default="full", help="运行模式")
    parser.add_argument("--query", help="搜索查询（仅用于search模式）")
    parser.add_argument("--k", type=int, default=5, help="搜索结果数量")
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'device': args.device,
        'batch_size': args.batch_size,
        'force_rebuild': args.force_rebuild,
        'input_path': args.input_path,
        'output_path': args.output_path,
        'index_path': args.index_path,
        'summary_limit': 80,
        'max_text_length': 4000,
        'chunk_batch_size': 1,
        'use_gpu': args.device != 'cpu'
    }
    
    # 创建应用程序
    app = OptimizedApp(config)
    
    try:
        if args.mode == "full":
            app.run_full_pipeline(force_rebuild=args.force_rebuild)
        elif args.mode == "chunk":
            app.load_models()
            app.process_civil_code(force_rebuild=args.force_rebuild)
        elif args.mode == "index":
            app.load_models()
            app.build_index(force_rebuild=args.force_rebuild)
        elif args.mode == "search":
            app.load_models()
            app.build_index()
            if args.query:
                results = app.search(args.query, k=args.k)
                if results:
                    print(f"\n搜索结果 (查询: {args.query}):")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['meta']['path']} (相似度: {result['score']:.4f})")
                        print(f"   摘要: {result['meta']['summary']}")
                        print(f"   文本: {result['text'][:100]}...")
                        print()
        
        # 生成性能报告
        report = app.generate_performance_report()
        print("\n" + report)
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        raise
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()