"""
性能基准测试脚本
- 测试各种配置下的性能
- 生成性能对比报告
- 提供优化建议
"""
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml

from optimized_main import OptimizedApp
from tools.performance_monitor import PerformanceMonitor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, config_file: str = "optimization_config.yaml"):
        self.config_file = Path(config_file)
        self.results = []
        self.baseline_config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _create_test_config(self, **overrides) -> Dict[str, Any]:
        """创建测试配置"""
        config = self.baseline_config.copy()
        
        # 更新配置
        for key, value in overrides.items():
            if '.' in key:
                # 处理嵌套键
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        return config
    
    def run_single_test(self, test_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个测试"""
        logger.info(f"运行测试: {test_name}")
        
        # 创建应用程序
        app = OptimizedApp(config)
        
        # 创建性能监控器
        monitor = PerformanceMonitor(f"benchmark_{test_name}.json")
        
        start_time = time.time()
        
        try:
            # 测试模型加载
            model_load_start = time.time()
            app.load_models()
            model_load_time = time.time() - model_load_start
            
            # 测试文本处理
            chunk_start = time.time()
            app.process_civil_code(force_rebuild=True)
            chunk_time = time.time() - chunk_start
            
            # 测试索引构建
            index_start = time.time()
            app.build_index(force_rebuild=True)
            index_time = time.time() - index_start
            
            # 测试搜索
            search_start = time.time()
            test_queries = ["合同成立的条件", "违约责任", "物权保护"]
            search_times = []
            for query in test_queries:
                query_start = time.time()
                results = app.search(query, k=5)
                search_times.append(time.time() - query_start)
            
            total_time = time.time() - start_time
            
            # 获取性能指标
            memory_usage = monitor.get_memory_usage()
            model_info = app.generate_performance_report()
            
            result = {
                'test_name': test_name,
                'config': config,
                'timing': {
                    'total_time': total_time,
                    'model_load_time': model_load_time,
                    'chunk_time': chunk_time,
                    'index_time': index_time,
                    'avg_search_time': sum(search_times) / len(search_times),
                    'search_times': search_times
                },
                'memory': memory_usage,
                'model_info': model_info
            }
            
            logger.info(f"测试 {test_name} 完成，总耗时: {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"测试 {test_name} 失败: {e}")
            return {
                'test_name': test_name,
                'config': config,
                'error': str(e),
                'timing': {'total_time': time.time() - start_time}
            }
        finally:
            app.cleanup()
    
    def run_batch_tests(self, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """运行批量测试"""
        results = []
        
        for i, test_config in enumerate(test_configs):
            test_name = test_config.get('name', f'test_{i+1}')
            config = test_config.get('config', {})
            
            result = self.run_single_test(test_name, config)
            results.append(result)
            
            # 清理内存
            import gc
            gc.collect()
            
            # 等待一段时间
            time.sleep(2)
        
        return results
    
    def run_optimization_tests(self) -> List[Dict[str, Any]]:
        """运行优化测试"""
        test_configs = [
            {
                'name': 'baseline',
                'config': self._create_test_config()
            },
            {
                'name': 'small_batch',
                'config': self._create_test_config(
                    batch_processing={'embedding_batch_size': 16}
                )
            },
            {
                'name': 'large_batch',
                'config': self._create_test_config(
                    batch_processing={'embedding_batch_size': 64}
                )
            },
            {
                'name': 'cpu_only',
                'config': self._create_test_config(
                    models={'device': 'cpu'},
                    indexing={'use_gpu': False}
                )
            },
            {
                'name': 'float16',
                'config': self._create_test_config(
                    models={'precision': 'float16'}
                )
            },
            {
                'name': 'no_cache',
                'config': self._create_test_config(
                    models={'use_cache': False},
                    text_processing={'enable_caching': False}
                )
            }
        ]
        
        return self.run_batch_tests(test_configs)
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """生成测试报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("性能基准测试报告")
        report_lines.append("=" * 80)
        
        # 过滤有效结果
        valid_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        if failed_results:
            report_lines.append(f"\n失败的测试 ({len(failed_results)}):")
            for result in failed_results:
                report_lines.append(f"  - {result['test_name']}: {result['error']}")
        
        if not valid_results:
            report_lines.append("\n没有成功的测试结果")
            return "\n".join(report_lines)
        
        # 性能对比
        report_lines.append(f"\n性能对比 ({len(valid_results)} 个测试):")
        report_lines.append("-" * 80)
        report_lines.append(f"{'测试名称':<20} {'总时间(s)':<10} {'模型加载(s)':<12} {'分块(s)':<10} {'索引(s)':<10} {'搜索(s)':<10}")
        report_lines.append("-" * 80)
        
        for result in valid_results:
            timing = result['timing']
            report_lines.append(
                f"{result['test_name']:<20} "
                f"{timing['total_time']:<10.2f} "
                f"{timing['model_load_time']:<12.2f} "
                f"{timing['chunk_time']:<10.2f} "
                f"{timing['index_time']:<10.2f} "
                f"{timing['avg_search_time']:<10.2f}"
            )
        
        # 找出最佳配置
        best_total = min(valid_results, key=lambda x: x['timing']['total_time'])
        best_chunk = min(valid_results, key=lambda x: x['timing']['chunk_time'])
        best_index = min(valid_results, key=lambda x: x['timing']['index_time'])
        best_search = min(valid_results, key=lambda x: x['timing']['avg_search_time'])
        
        report_lines.append("\n最佳配置:")
        report_lines.append(f"  总时间最短: {best_total['test_name']} ({best_total['timing']['total_time']:.2f}s)")
        report_lines.append(f"  分块最快: {best_chunk['test_name']} ({best_chunk['timing']['chunk_time']:.2f}s)")
        report_lines.append(f"  索引最快: {best_index['test_name']} ({best_index['timing']['index_time']:.2f}s)")
        report_lines.append(f"  搜索最快: {best_search['test_name']} ({best_search['timing']['avg_search_time']:.2f}s)")
        
        # 内存使用对比
        report_lines.append("\n内存使用对比:")
        report_lines.append("-" * 60)
        report_lines.append(f"{'测试名称':<20} {'RSS内存(MB)':<15} {'内存使用率(%)':<15}")
        report_lines.append("-" * 60)
        
        for result in valid_results:
            memory = result['memory']
            report_lines.append(
                f"{result['test_name']:<20} "
                f"{memory['rss']:<15.1f} "
                f"{memory['percent']:<15.1f}"
            )
        
        # 优化建议
        report_lines.append("\n优化建议:")
        report_lines.append("-" * 40)
        
        # 分析性能差异
        total_times = [r['timing']['total_time'] for r in valid_results]
        avg_time = sum(total_times) / len(total_times)
        
        for result in valid_results:
            timing = result['timing']
            if timing['total_time'] < avg_time * 0.9:
                report_lines.append(f"✓ {result['test_name']} 表现优秀")
            elif timing['total_time'] > avg_time * 1.1:
                report_lines.append(f"⚠ {result['test_name']} 性能较差，建议优化")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "benchmark_results.json"):
        """保存测试结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"测试结果已保存到: {filename}")

def main():
    """主函数"""
    benchmark = PerformanceBenchmark()
    
    logger.info("开始性能基准测试...")
    
    # 运行优化测试
    results = benchmark.run_optimization_tests()
    
    # 生成报告
    report = benchmark.generate_report(results)
    print(report)
    
    # 保存结果
    benchmark.save_results(results)
    
    # 保存报告
    with open("benchmark_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("基准测试完成")

if __name__ == "__main__":
    main()