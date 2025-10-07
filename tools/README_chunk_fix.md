# 民法典分块处理程序修复说明

## 问题分析

原程序在处理一百多条数据后自动关闭的可能原因：

1. **GPU内存泄漏**：每次调用`summarize`函数时，GPU内存没有正确释放
2. **异常处理不足**：某个条文处理失败时程序崩溃
3. **内存累积**：长时间运行导致内存不足
4. **模型资源管理不当**：没有定期清理GPU缓存

## 解决方案

### 1. 简化版本 (`chunk_civil_code_simple.py`)

**主要改进：**
- 添加了`clear_gpu_memory()`函数，定期清理GPU缓存
- 减少了`max_length`从4096到2048，降低内存使用
- 每处理20条条文清理一次内存
- 添加了异常处理，单个条文失败不影响整体处理
- 添加了内存使用监控

**使用方法：**
```bash
cd /workspace
python3 tools/chunk_civil_code_simple.py
```

### 2. 完整版本 (`chunk_civil_code_improved.py`)

**主要改进：**
- 添加了进度保存和恢复功能
- 实现了安全的摘要生成（带重试机制）
- 添加了完整的异常处理
- 支持程序中断后继续处理
- 更详细的内存管理和监控

**使用方法：**
```bash
cd /workspace
python3 tools/chunk_civil_code_improved.py
```

### 3. 调试版本 (`debug_chunk.py`)

**功能：**
- 测试模型加载是否正常
- 检查GPU内存使用情况
- 测试文本解析功能
- 验证摘要生成功能

**使用方法：**
```bash
cd /workspace
python3 tools/debug_chunk.py
```

## 建议的处理步骤

1. **首先运行调试版本**，确认环境和模型都正常：
   ```bash
   python3 tools/debug_chunk.py
   ```

2. **如果调试通过，运行简化版本**：
   ```bash
   python3 tools/chunk_civil_code_simple.py
   ```

3. **如果简化版本仍有问题，可以尝试完整版本**：
   ```bash
   python3 tools/chunk_civil_code_improved.py
   ```

## 关键改进点

### 内存管理
```python
def clear_gpu_memory():
    """强制清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
```

### 异常处理
```python
def summarize_with_memory_management(title: str, body: str, limit: int = 80) -> str:
    try:
        # 摘要生成逻辑
        return result
    except Exception as e:
        print(f"摘要生成失败: {e}")
        return ""
    finally:
        clear_gpu_memory()  # 确保内存被清理
```

### 进度监控
```python
# 每处理20条清理一次内存
if i % 20 == 0:
    clear_gpu_memory()
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"已处理 {articles} 条，GPU内存使用: {memory_used:.2f}GB")
```

## 故障排除

如果程序仍然提前终止，请检查：

1. **GPU内存是否足够**：运行`nvidia-smi`查看GPU内存使用情况
2. **系统内存是否足够**：运行`free -h`查看系统内存
3. **模型是否正确加载**：运行调试版本确认
4. **是否有权限问题**：确保有写入`data/processed/`目录的权限

## 性能优化建议

1. **减少批处理大小**：如果内存不足，可以减少每批处理的条文数量
2. **使用更小的模型**：如果可能，使用更小的语言模型
3. **增加清理频率**：如果内存使用仍然很高，可以增加`clear_gpu_memory()`的调用频率
4. **使用CPU模式**：如果GPU内存严重不足，可以考虑使用CPU模式（虽然会很慢）