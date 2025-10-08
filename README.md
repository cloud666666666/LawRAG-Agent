# LawRAG-Agent

LawRAG-Agent 是一个基于检索增强生成（RAG）技术的中文法律助手系统，专门设计用于回答法律相关问题，并提供准确的法律条文引用。

## 项目概述

本项目利用大语言模型（LLM）和向量检索技术，构建了一个能够理解和回答中文法律问题的智能系统。系统具有以下特点：

- 基于 LangChain 框架构建，集成多种组件
- 使用双向量检索策略（原文 + 摘要）提高检索准确性
- 应用重排序技术优化检索结果
- 支持批量查询和评估功能

## 系统架构

系统主要由以下几个核心组件组成：

1. **嵌入模型**：使用 BGE-M3 模型将文本转换为向量表示
2. **重排序模型**：使用 BGE-Reranker-V2-M3 模型对检索结果进行重排序
3. **大语言模型**：使用 Qwen2.5-7B-Instruct 模型生成回答
4. **向量存储**：基于 FAISS 的自定义向量存储，支持高效相似度搜索
5. **RAG 链**：集成检索和生成的完整处理流程

## 主要功能

- **法律文本检索**：从法律文本库中检索相关条文
- **法律摘要检索**：从法律摘要中检索相关信息
- **结果重排序**：使用专业重排序模型提高检索质量
- **RRF 融合**：通过 Reciprocal Rank Fusion 算法融合多种检索结果
- **法律问答生成**：基于检索结果生成专业的法律回答

## 目录结构

```
LawRAG-Agent/
├── data/                  # 数据目录
│   ├── index/            # 索引数据
│   ├── origin/           # 原始法律文本
│   ├── processed/        # 处理后的数据
│   ├── results/          # 查询结果和评估
│   └── vectors/          # 向量存储
├── models/                # 模型目录
│   ├── bge-m3/           # 嵌入模型
│   ├── bge-reranker-v2-m3/ # 重排序模型
│   └── qwen2.5-7b-instruct/ # 大语言模型
├── tools/                 # 工具脚本
│   ├── batch_query.py    # 批量查询处理
│   ├── chunk.py          # 文本分块处理
│   ├── download_models.py # 模型下载
│   ├── embed_index.py    # 嵌入索引
│   ├── evaluate.py       # 评估工具
│   ├── extract_pdf.py    # PDF提取
│   ├── generate.py       # 生成工具
│   ├── models_load.py    # 模型加载
│   ├── summaries.py      # 摘要生成
│   └── vectordb_builder.py # 向量数据库构建
├── .gitignore            # Git忽略文件
├── example_queries.txt   # 示例查询
├── main.py               # 主程序
├── requirements.txt      # 依赖包
├── README.md             # 项目说明
└── run.sh                # 运行脚本
```

## 安装与配置

### 环境要求

- Python 3.8+
- CUDA 支持（推荐用于模型推理）
- 16GB+ 内存

### 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 下载模型

项目使用以下模型：

- 嵌入模型：BGE-M3
- 重排序模型：BGE-Reranker-V2-M3
- 大语言模型：Qwen2.5-7B-Instruct

可以使用 `tools/download_models.py` 脚本下载模型。

## 使用方法

### 数据准备

1. 将原始法律文本放入 `data/origin/` 目录
2. 运行文本处理脚本：
   ```bash
   python tools/extract_pdf.py  # 如果源文件是PDF
   python tools/chunk.py        # 文本分块
   ```

### 构建向量数据库

```bash
python tools/vectordb_builder.py
```

### 生成法律摘要

```bash
python tools/generate_summaries.py
```

### 运行查询

单个查询：
```bash
python main.py "民事主体的权利有哪些"
```

批量查询：
```bash
python tools/batch_query.py
```

### 评估结果

```bash
python tools/evaluate.py
```

## 示例查询

项目包含以下示例查询（见 `example_queries.txt`）：

- 民事主体的权利有哪些
- 如何处理民事纠纷
- 未成年人的民事行为能力
- 法人的民事责任
- 民事法律关系的主体
- 婚姻家庭关系的法律规定
- 继承权的行使
- 合同的订立和履行
- 侵权责任的认定
- 物权的保护
- 知识产权的保护
- 人格权的保护

## 高级功能

### 重排序和融合

系统支持两种高级检索优化技术：

1. **重排序**：使用 BGE-Reranker-V2-M3 模型对初始检索结果进行重新排序，提高相关性
2. **RRF融合**：使用 Reciprocal Rank Fusion 算法融合来自不同检索源的结果

这些功能可以在 `tools/batch_query.py` 中配置：

```python
# 启用重排序
args.use_rerank = True
args.rerank_top_k = 5

# 启用RRF融合
args.use_rrf = True
args.rrf_k = 60
```

## 性能优化

为提高系统性能，项目实现了以下优化：

- 使用 FAISS 进行高效向量检索
- 批处理查询和嵌入计算
- 多进程资源管理和清理
- 缓存机制减少重复计算

## 贡献指南

欢迎对项目进行贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[待添加许可证信息]

## 联系方式

[待添加联系方式]
