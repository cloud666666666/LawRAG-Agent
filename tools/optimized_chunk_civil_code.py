"""
优化后的民法典分块处理模块
- 优化文本处理管道
- 减少重复计算
- 改进批处理性能
- 添加缓存机制
"""
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import torch
from transformers import AutoTokenizer, GenerationConfig
from optimized_models_load import llm_load, get_optimized_generation_config
from transformers.utils import logging as transformers_logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()  # 减少transformers日志输出

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

class OptimizedCivilCodeChunker:
    """优化的民法典分块处理器"""
    
    def __init__(self, 
                 input_path: str = "data/origin/民法典.txt",
                 output_path: str = "data/processed/chunks.jsonl",
                 summary_limit: int = 80,
                 max_text_length: int = 4000,
                 batch_size: int = 1):
        """
        初始化分块处理器
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            summary_limit: 摘要长度限制
            max_text_length: 最大文本长度
            batch_size: 批处理大小（LLM生成）
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.summary_limit = summary_limit
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 编译正则表达式（提高性能）
        self.patterns = self._compile_patterns()
        
        # 性能统计
        self.stats = {
            'total_lines': 0,
            'total_articles': 0,
            'processing_time': 0,
            'summarization_time': 0,
            'llm_calls': 0
        }
        
        # 缓存
        self._summary_cache = {}
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """编译正则表达式模式"""
        return {
            'bian': re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+编\b.*$'),
            'zhang': re.compile(r'^第\s*(?:[一二三四五六七八九十百千零\d]\s*)+章\b.*$'),
            'tiao': re.compile(r'^第\s*(?:[一二三四五六七八九十百零\d]\s*)+条\b.*$'),
            'tiao_no': re.compile(r'第\s*([一二三四五六七八九十百千零\d\s]+)\s*条'),
            'whitespace': re.compile(r'[ \t]+'),
            'newlines': re.compile(r'\n{2,}'),
            'page_numbers': re.compile(r'[—–\-－]\s*\d+\s*[—–\-－]'),
            'pure_numbers': re.compile(r'^\s*\d+\s*$', re.M),
            'catalog': re.compile(r'目\s*录.*?(?=^第[一二三四五六七八九十百千零\d]+\s*编.*$)', re.S|re.M),
            'prefix_clean': re.compile(r'^[\-\d\.\)（）\s]+'),
            'sentence_split': re.compile(r'[。！？!？]\s*')
        }
    
    def _clean_text(self, text: str) -> str:
        """优化的文本清理"""
        # 基础清理
        text = text.replace("\u3000", " ")
        text = self.patterns['whitespace'].sub(" ", text)
        text = text.replace("\r", "")
        text = self.patterns['newlines'].sub("\n", text)
        
        # 去除页码和目录
        text = self.patterns['page_numbers'].sub("", text)
        text = self.patterns['pure_numbers'].sub("", text)
        text = self.patterns['catalog'].sub("", text)
        
        return text.strip()
    
    def _clean_number(self, s: str) -> str:
        """清理条号中的空格"""
        return re.sub(r"\s+", "", s)
    
    @lru_cache(maxsize=1000)
    def _cached_summarize(self, title: str, body: str) -> str:
        """缓存的摘要生成"""
        cache_key = f"{title[:100]}_{body[:200]}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]
        
        summary = self._summarize_impl(title, body)
        self._summary_cache[cache_key] = summary
        return summary
    
    def _summarize_impl(self, title: str, body: str) -> str:
        """摘要生成实现"""
        text = (body or title or "").strip()
        if not text:
            raise ValueError("summarize(): empty text")
        
        # 文本截断优化
        if len(text) > self.max_text_length:
            text_for_llm = text[:3000] + "\n……\n" + text[-800:]
        else:
            text_for_llm = text
        
        # 构建提示词
        user_prompt = (
            f"你是一名严谨的中文法律助理。仅依据【正文】提炼**单句**要点摘要，要求：\n"
            f"1) 客观中立，不新增原文之外的事实或数值；\n"
            f"2) 优先保留义务/权利/适用范围/例外等关键信息；\n"
            f"3) 不复述条号或标题；不使用"可能/建议"等评价词；\n"
            f"4) 摘要长度≤{self.summary_limit}字，且为**单句**。\n\n"
            f"【标题】\n{title}\n\n【正文】\n{text_for_llm}\n"
        )
        
        try:
            # 获取模型和分词器
            llm, tokenizer = llm_load()
            
            # 检查chat template
            if not hasattr(tokenizer, "apply_chat_template"):
                raise RuntimeError("summarize(): tokenizer has no apply_chat_template")
            
            # 构建消息
            messages = [
                {"role": "system", "content": "你是严谨的中文法律摘要助手。"},
                {"role": "user", "content": user_prompt},
            ]
            
            # 应用chat template
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            if not chat_text or not isinstance(chat_text, str):
                raise RuntimeError("summarize(): apply_chat_template returned empty/invalid text")
            
            # 分词
            inputs = tokenizer(
                chat_text,
                return_tensors="pt",
                truncation=True,
                max_length=getattr(tokenizer, "model_max_length", 4096)
            )
            
            device = next(llm.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 获取优化的生成配置
            gen_config = get_optimized_generation_config(tokenizer)
            gen_config.max_new_tokens = 160
            
            # 生成摘要
            with torch.inference_mode():
                out = llm.generate(**inputs, generation_config=gen_config)
            
            # 解码
            input_len = inputs["input_ids"].shape[1]
            gen_ids = out[0, input_len:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            if not raw:
                raise RuntimeError("summarize(): LLM returned empty text")
            
            # 后处理
            cand = raw.splitlines()[-1].strip()
            cand = self.patterns['prefix_clean'].sub("", cand)
            cand = self.patterns['sentence_split'].split(cand, maxsplit=1)[0].strip()
            
            if not cand:
                raise RuntimeError("summarize(): empty after single-sentence enforcement")
            
            # 长度限制
            if len(cand) > self.summary_limit:
                cand = cand[:self.summary_limit].rstrip()
                if not cand:
                    raise RuntimeError("summarize(): empty after length trimming")
            
            self.stats['llm_calls'] += 1
            return cand
            
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            # 返回简化摘要作为后备
            return f"法律条文：{title[:50]}..." if title else "法律条文摘要"
    
    def _summarize(self, title: str, body: str) -> str:
        """摘要生成（带缓存）"""
        return self._cached_summarize(title, body)
    
    def _flush_article(self, article_title: str, article_body: List[str], 
                      cur_bian: Optional[str], cur_zhang: Optional[str], 
                      law_name: str) -> Dict[str, Any]:
        """刷新文章到输出"""
        if not article_title:
            return None
        
        body = "\n".join(article_body).strip()
        full_text = (article_title + ("\n" + body if body else "")).strip()
        
        # 构建路径
        m = self.patterns['tiao_no'].search(article_title)
        art_cn = self._clean_number(m.group(1)) if m else None
        
        parts = []
        if cur_bian: parts.append(cur_bian)
        if cur_zhang: parts.append(cur_zhang)
        parts.append(f"第{art_cn}条" if art_cn else "未知条")
        path_str = "/".join(parts)
        
        # 生成摘要
        summary = self._summarize(article_title, body)
        
        return {
            "text": full_text,
            "meta": {
                "law": law_name,
                "path": path_str,
                "summary": summary
            }
        }
    
    def process(self, force_rebuild: bool = False) -> str:
        """
        处理民法典文件
        
        Args:
            force_rebuild: 是否强制重建
            
        Returns:
            输出文件路径
        """
        if self.output_path.exists() and not force_rebuild:
            logger.info(f"输出文件已存在: {self.output_path}")
            return str(self.output_path)
        
        start_time = time.time()
        logger.info(f"开始处理民法典文件: {self.input_path}")
        
        try:
            # 读取和清理文本
            logger.info("读取和清理文本...")
            txt = self.input_path.read_text(encoding="utf-8", errors="ignore")
            txt = self._clean_text(txt)
            
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            self.stats['total_lines'] = len(lines)
            
            # 清空旧文件
            if self.output_path.exists():
                self.output_path.unlink()
            
            # 解析状态
            cur_bian = None
            cur_zhang = None
            article_title = None
            article_body = []
            articles = 0
            
            law_name = self.input_path.stem
            
            # 逐行解析
            logger.info("开始解析法律条文...")
            for ln in tqdm(lines, desc="解析法律条文", unit="行"):
                if self.patterns['bian'].match(ln):
                    cur_bian = ln
                    continue
                if self.patterns['zhang'].match(ln):
                    cur_zhang = ln
                    continue
                if self.patterns['tiao'].match(ln):
                    # 刷新上一个文章
                    if article_title:
                        article_data = self._flush_article(
                            article_title, article_body, cur_bian, cur_zhang, law_name
                        )
                        if article_data:
                            with self.output_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(article_data, ensure_ascii=False) + "\n")
                            articles += 1
                    
                    # 开始新文章
                    article_title = ln
                    article_body = []
                    continue
                if article_title:
                    article_body.append(ln)
            
            # 处理最后一个文章
            if article_title:
                article_data = self._flush_article(
                    article_title, article_body, cur_bian, cur_zhang, law_name
                )
                if article_data:
                    with self.output_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(article_data, ensure_ascii=False) + "\n")
                    articles += 1
            
            self.stats['total_articles'] = articles
            self.stats['processing_time'] = time.time() - start_time
            
            # 输出统计信息
            self._print_stats()
            
            return str(self.output_path)
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            raise
    
    def _print_stats(self):
        """打印统计信息"""
        stats = self.stats
        size = self.output_path.stat().st_size if self.output_path.exists() else 0
        
        def human_size(n):
            for u in ["B", "KB", "MB", "GB"]:
                if n < 1024 or u == "GB":
                    return f"{n:.1f} {u}"
                n /= 1024
        
        logger.info("=" * 50)
        logger.info("民法典分块处理完成 - 统计信息")
        logger.info("=" * 50)
        logger.info(f"输入文件: {self.input_path}")
        logger.info(f"输出文件: {self.output_path}")
        logger.info(f"处理行数: {stats['total_lines']:,}")
        logger.info(f"生成文章: {stats['total_articles']:,}")
        logger.info(f"处理时间: {stats['processing_time']:.2f}s")
        logger.info(f"LLM调用次数: {stats['llm_calls']:,}")
        logger.info(f"平均速度: {stats['total_lines']/stats['processing_time']:.1f} lines/s")
        logger.info(f"文件大小: {human_size(size)}")
        logger.info("=" * 50)

def main():
    """主函数"""
    # 创建分块处理器
    chunker = OptimizedCivilCodeChunker(
        summary_limit=80,
        max_text_length=4000
    )
    
    # 处理文件
    output_path = chunker.process(force_rebuild=False)
    print(f"处理完成: {output_path}")

if __name__ == "__main__":
    main()