"""
性能监控和优化工具
- 监控内存使用
- 记录性能指标
- 提供优化建议
- 生成性能报告
"""
import os
import time
import psutil
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_file: str = "performance_log.json"):
        self.log_file = Path(log_file)
        self.metrics = []
        self.start_time = time.time()
        
        # 确保日志目录存在
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'python_version': os.sys.version,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况（MB）"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics = {
            'rss': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024
        }
        
        # GPU内存使用
        if torch.cuda.is_available():
            metrics['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024
            metrics['gpu_cached'] = torch.cuda.memory_reserved() / 1024 / 1024
            metrics['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return metrics
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """获取CPU使用情况"""
        return {
            'percent': psutil.cpu_percent(interval=1),
            'per_cpu': psutil.cpu_percent(percpu=True),
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    def record_metric(self, operation: str, duration: float, 
                     memory_before: Optional[Dict] = None,
                     memory_after: Optional[Dict] = None,
                     additional_info: Optional[Dict] = None):
        """记录性能指标"""
        metric = {
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'memory_before': memory_before,
            'memory_after': memory_after,
            'additional_info': additional_info or {}
        }
        
        self.metrics.append(metric)
        logger.info(f"记录指标: {operation} - {duration:.2f}s")
    
    def save_metrics(self):
        """保存性能指标到文件"""
        report = {
            'system_info': self.get_system_info(),
            'total_runtime': time.time() - self.start_time,
            'metrics': self.metrics,
            'summary': self._generate_summary()
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能指标已保存到: {self.log_file}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成性能摘要"""
        if not self.metrics:
            return {}
        
        durations = [m['duration'] for m in self.metrics]
        operations = [m['operation'] for m in self.metrics]
        
        # 按操作类型分组
        operation_groups = {}
        for metric in self.metrics:
            op = metric['operation']
            if op not in operation_groups:
                operation_groups[op] = []
            operation_groups[op].append(metric['duration'])
        
        summary = {
            'total_operations': len(self.metrics),
            'total_duration': sum(durations),
            'average_duration': np.mean(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'operation_breakdown': {
                op: {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': np.mean(times),
                    'max_time': max(times)
                }
                for op, times in operation_groups.items()
            }
        }
        
        return summary
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        # 检查内存使用
        memory = self.get_memory_usage()
        if memory['percent'] > 80:
            suggestions.append("内存使用率过高，建议减少批处理大小或使用更小的模型")
        
        if torch.cuda.is_available() and memory.get('gpu_allocated', 0) > 0:
            gpu_usage = memory['gpu_allocated'] / (memory['gpu_allocated'] + memory['gpu_cached'])
            if gpu_usage < 0.5:
                suggestions.append("GPU内存利用率较低，可以增加批处理大小")
        
        # 检查操作耗时
        if self.metrics:
            long_operations = [m for m in self.metrics if m['duration'] > 10]
            if long_operations:
                suggestions.append(f"发现 {len(long_operations)} 个耗时操作，建议优化")
        
        # 检查重复操作
        operation_counts = {}
        for metric in self.metrics:
            op = metric['operation']
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        frequent_ops = [op for op, count in operation_counts.items() if count > 10]
        if frequent_ops:
            suggestions.append(f"发现频繁操作: {frequent_ops}，建议添加缓存机制")
        
        return suggestions

class ModelPerformanceProfiler:
    """模型性能分析器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.model_load_times = {}
        self.inference_times = []
    
    def profile_model_loading(self, model_name: str, load_func, *args, **kwargs):
        """分析模型加载性能"""
        memory_before = self.monitor.get_memory_usage()
        start_time = time.time()
        
        try:
            result = load_func(*args, **kwargs)
            duration = time.time() - start_time
            memory_after = self.monitor.get_memory_usage()
            
            self.model_load_times[model_name] = duration
            
            self.monitor.record_metric(
                f"load_{model_name}",
                duration,
                memory_before,
                memory_after,
                {'model_name': model_name}
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_metric(
                f"load_{model_name}_failed",
                duration,
                memory_before,
                None,
                {'error': str(e), 'model_name': model_name}
            )
            raise
    
    def profile_inference(self, operation_name: str, inference_func, *args, **kwargs):
        """分析推理性能"""
        memory_before = self.monitor.get_memory_usage()
        start_time = time.time()
        
        try:
            result = inference_func(*args, **kwargs)
            duration = time.time() - start_time
            memory_after = self.monitor.get_memory_usage()
            
            self.inference_times.append(duration)
            
            self.monitor.record_metric(
                f"inference_{operation_name}",
                duration,
                memory_before,
                memory_after,
                {'operation': operation_name}
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_metric(
                f"inference_{operation_name}_failed",
                duration,
                memory_before,
                None,
                {'error': str(e), 'operation': operation_name}
            )
            raise
    
    def get_inference_stats(self) -> Dict[str, float]:
        """获取推理统计信息"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'count': len(times),
            'total_time': float(np.sum(times)),
            'average_time': float(np.mean(times)),
            'median_time': float(np.median(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times))
        }

def create_performance_report(monitor: PerformanceMonitor) -> str:
    """创建性能报告"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("性能分析报告")
    report_lines.append("=" * 60)
    
    # 系统信息
    system_info = monitor.get_system_info()
    report_lines.append(f"系统信息:")
    report_lines.append(f"  CPU核心数: {system_info['cpu_count']}")
    report_lines.append(f"  总内存: {system_info['memory_total'] / 1024 / 1024 / 1024:.1f} GB")
    report_lines.append(f"  可用内存: {system_info['memory_available'] / 1024 / 1024 / 1024:.1f} GB")
    report_lines.append(f"  CUDA可用: {system_info['cuda_available']}")
    if system_info['cuda_available']:
        report_lines.append(f"  GPU数量: {system_info['cuda_device_count']}")
    
    # 当前内存使用
    memory = monitor.get_memory_usage()
    report_lines.append(f"\n当前内存使用:")
    report_lines.append(f"  RSS内存: {memory['rss']:.1f} MB")
    report_lines.append(f"  内存使用率: {memory['percent']:.1f}%")
    if 'gpu_allocated' in memory:
        report_lines.append(f"  GPU内存: {memory['gpu_allocated']:.1f} MB")
    
    # 性能摘要
    if monitor.metrics:
        summary = monitor._generate_summary()
        report_lines.append(f"\n性能摘要:")
        report_lines.append(f"  总操作数: {summary['total_operations']}")
        report_lines.append(f"  总耗时: {summary['total_duration']:.2f}s")
        report_lines.append(f"  平均耗时: {summary['average_duration']:.2f}s")
        report_lines.append(f"  最长耗时: {summary['max_duration']:.2f}s")
        
        # 操作分解
        report_lines.append(f"\n操作分解:")
        for op, stats in summary['operation_breakdown'].items():
            report_lines.append(f"  {op}:")
            report_lines.append(f"    次数: {stats['count']}")
            report_lines.append(f"    总时间: {stats['total_time']:.2f}s")
            report_lines.append(f"    平均时间: {stats['average_time']:.2f}s")
    
    # 优化建议
    suggestions = monitor.get_optimization_suggestions()
    if suggestions:
        report_lines.append(f"\n优化建议:")
        for i, suggestion in enumerate(suggestions, 1):
            report_lines.append(f"  {i}. {suggestion}")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)

# 使用示例
if __name__ == "__main__":
    # 创建性能监控器
    monitor = PerformanceMonitor()
    
    # 记录一些示例操作
    monitor.record_metric("test_operation", 1.5, additional_info={"test": True})
    
    # 生成报告
    report = create_performance_report(monitor)
    print(report)
    
    # 保存指标
    monitor.save_metrics()