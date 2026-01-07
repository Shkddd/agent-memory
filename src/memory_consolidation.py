"""
Memory Consolidation: 内存聚合与压缩
支持相似度聚类、时间窗口聚合、优先级保护等
"""
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from embedding_util import EmbeddingUtil
from memory_summarizer import MemorySummarizer, RuleBasedSummarizer

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """压缩配置"""
    
    # 聚合触发条件
    max_memories_per_agent: int = 1000  # 单agent最大记忆数
    max_memory_size_mb: int = 500  # 最大内存占用（MB）
    consolidation_interval_hours: int = 24  # 压缩间隔（小时）
    
    # 聚合策略
    similarity_threshold: float = 0.75  # 相似度阈值（0-1）
    time_window_days: int = 7  # 时间窗口（天）
    min_cluster_size: int = 3  # 最小聚合单位
    
    # 保护规则
    protect_priority_levels: List[str] = None  # 保护的优先级（如['HIGH', 'CRITICAL']）
    protect_memory_types: List[str] = None  # 保护的记忆类型（如['fact', 'rule']）
    protect_recent_days: int = 7  # 最近N天的记忆不压缩
    
    # 摘要配置
    summary_max_length: int = 300  # 摘要最大长度
    
    def __post_init__(self):
        if self.protect_priority_levels is None:
            self. protect_priority_levels = ['HIGH', 'CRITICAL']
        if self.protect_memory_types is None:
            self.protect_memory_types = ['fact', 'rule', 'user_preference']


class MemoryCluster:
    """相似记忆簇"""
    
    def __init__(self, cluster_id: int, memories: List[Dict], 
                 centroid:  np.ndarray, avg_similarity: float):
        self.cluster_id = cluster_id
        self.memories = memories
        self.centroid = centroid
        self.avg_similarity = avg_similarity
        self.created_at = datetime.now()
    
    def __repr__(self):
        return (
            f"Cluster(id={self.cluster_id}, size={len(self.memories)}, "
            f"similarity={self.avg_similarity:.2f})"
        )


class MemoryConsolidator:
    """记忆聚合和压缩引擎"""
    
    def __init__(self, config: ConsolidationConfig = None, 
                 summarizer: MemorySummarizer = None):
        """
        初始化压缩引擎
        
        Args:
            config: 压缩配置
            summarizer: 摘要器
        """
        self.config = config or ConsolidationConfig()
        self.summarizer = summarizer or MemorySummarizer(RuleBasedSummarizer())
        self.embedding_util = EmbeddingUtil()
        self.consolidation_records: List[Dict] = []
    
    def should_consolidate(self, memory_count: int, last_consolidation: Optional[datetime] = None) -> bool:
        """
        判断是否需要压缩
        
        Args: 
            memory_count: 当前记忆数
            last_consolidation: 上次压缩时间
            
        Returns:
            是否需要压缩
        """
        # 检查数量
        if memory_count > self.config.max_memories_per_agent:
            logger.info(f"Consolidation triggered: memory count ({memory_count}) exceeds limit")
            return True
        
        # 检查时间间隔
        if last_consolidation: 
            time_since = datetime.now() - last_consolidation
            if time_since > timedelta(hours=self.config.consolidation_interval_hours):
                logger.info(f"Consolidation triggered: {time_since} since last consolidation")
                return True
        
        return False
    
    def cluster_similar_memories(
        self,
        memories: List[Dict],
        embedding_dim: int = 384
    ) -> List[MemoryCluster]:
        """
        使用相似度聚类将相似记忆分组
        
        Args: 
            memories: 记忆列表
            embedding_dim: 向量维度
            
        Returns:
            记忆簇列表
        """
        if len(memories) < self.config.min_cluster_size:
            return []
        
        # 生成嵌入向量
        texts = [m.get('text', '') for m in memories]
        embeddings = np.array([
            self.embedding_util.get_embedding(text) for text in texts
        ])
        
        clusters = []
        cluster_id = 0
        used_indices = set()
        
        # 简单聚类：对每个未分配的记忆，找相似的邻居
        for i, embedding in enumerate(embeddings):
            if i in used_indices: 
                continue
            
            # 找相似的记忆
            similarities = np.dot(embeddings, embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding) + 1e-8
            )
            similar_indices = np.where(similarities >= self.config.similarity_threshold)[0]
            
            if len(similar_indices) >= self.config.min_cluster_size:
                cluster_memories = [memories[idx] for idx in similar_indices]
                centroid = np.mean(embeddings[similar_indices], axis=0)
                avg_sim = float(np.mean(similarities[similar_indices]))
                
                cluster = MemoryCluster(
                    cluster_id=cluster_id,
                    memories=cluster_memories,
                    centroid=centroid,
                    avg_similarity=avg_sim
                )
                clusters.append(cluster)
                used_indices.update(similar_indices)
                cluster_id += 1
        
        logger.info(f"Created {len(clusters)} memory clusters")
        return clusters
    
    def filter_consolidable_memories(self, memories: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        过滤可压缩的记忆（保护重要记忆）
        
        Args:
            memories: 记忆列表
            
        Returns:
            (可压缩的记忆, 受保护的记忆)
        """
        consolidable = []
        protected = []
        
        cutoff_date = datetime.now() - timedelta(days=self. config.protect_recent_days)
        
        for memory in memories:
            # 检查优先级
            priority = memory.get('metadata', {}).get('priority', 'MEDIUM')
            if priority in self. config.protect_priority_levels:
                protected.append(memory)
                continue
            
            # 检查类型
            mem_type = memory.get('metadata', {}).get('type', 'unknown')
            if mem_type in self.config.protect_memory_types:
                protected.append(memory)
                continue
            
            # 检查时间
            created_at_str = memory.get('created_at')
            if created_at_str: 
                created_at = datetime. fromisoformat(created_at_str)
                if created_at > cutoff_date:
                    protected.append(memory)
                    continue
            
            consolidable.append(memory)
        
        logger.info(
            f"Filtered memories: {len(consolidable)} consolidable, "
            f"{len(protected)} protected"
        )
        return consolidable, protected
    
    def consolidate_cluster(self, cluster: MemoryCluster) -> Dict:
        """
        压缩一个记忆簇，生成摘要
        
        Args: 
            cluster: 记忆簇
            
        Returns: 
            摘要结果
        """
        summary_result = self.summarizer.summarize_memories(
            cluster.memories,
            max_length=self. config.summary_max_length,
            topic=f"cluster_{cluster.cluster_id}"
        )
        
        # 记录压缩日志
        record = {
            'timestamp': datetime.now().isoformat(),
            'cluster_id':  cluster.cluster_id,
            'original_count': len(cluster.memories),
            'summary_length': len(summary_result['summary']),
            'compression_ratio': (
                summary_result['original_length'] / 
                max(1, len(summary_result['summary']))
            ),
            'source_ids': cluster.memories[0].get('id'),  # 用第一条记忆的ID作为来源
            'summary': summary_result['summary'][: 100] + "..."  # 截断用于日志
        }
        self.consolidation_records.append(record)
        
        logger. info(
            f"Consolidated cluster {cluster.cluster_id}:  "
            f"{record['original_count']} -> 1 memory "
            f"(compression:  {record['compression_ratio']:. 1f}x)"
        )
        
        return summary_result
    
    def run_consolidation(
        self,
        memories: List[Dict],
        agent_id: str = "default"
    ) -> Dict:
        """
        执行完整的压缩流程
        
        Args: 
            memories: 所有记忆列表
            agent_id: 代理ID（用于日志）
            
        Returns:
            压缩结果 {
                'consolidated_count': int,
                'consolidated_memories':  List[Dict],  # 新的摘要记忆
                'archived_memory_ids': List,  # 被归档的原记忆ID
                'summary_results': List[Dict]
            }
        """
        logger.info(f"Starting consolidation for agent {agent_id} ({len(memories)} memories)")
        
        # 过滤可压缩和受保护的记忆
        consolidable, protected = self. filter_consolidable_memories(memories)
        
        if len(consolidable) < self.config.min_cluster_size:
            logger.info(f"Not enough consolidable memories (need >= {self.config.min_cluster_size})")
            return {
                'consolidated_count': 0,
                'consolidated_memories': [],
                'archived_memory_ids': [],
                'summary_results': [],
                'message': 'Not enough memories to consolidate'
            }
        
        # 聚类相似记忆
        clusters = self.cluster_similar_memories(consolidable)
        
        if not clusters:
            logger.warning("No clusters found during consolidation")
            return {
                'consolidated_count': 0,
                'consolidated_memories':  [],
                'archived_memory_ids': [],
                'summary_results': [],
                'message':  'No clusters found'
            }
        
        # 压缩每个簇
        consolidated_memories = []
        archived_ids = []
        summary_results = []
        
        for cluster in clusters:
            summary = self.consolidate_cluster(cluster)
            consolidated_memories. append({
                'text': summary['summary'],
                'metadata': {
                    'type': 'consolidated_summary',
                    'priority': 'MEDIUM',
                    'source_cluster_id': cluster.cluster_id,
                    'source_count': summary['source_count'],
                    'source_ids':  summary['source_ids']
                },
                'created_at': datetime.now().isoformat(),
                'is_summary': True
            })
            
            # 记录被归档的ID
            archived_ids.extend([m. get('id') for m in cluster.memories if 'id' in m])
            summary_results.append(summary)
        
        logger.info(
            f"Consolidation complete: {len(clusters)} clusters -> "
            f"{len(consolidated_memories)} summaries, "
            f"{len(archived_ids)} memories archived"
        )
        
        return {
            'consolidated_count':  len(clusters),
            'consolidated_memories': consolidated_memories,
            'archived_memory_ids':  archived_ids,
            'summary_results': summary_results,
            'message': 'Consolidation successful'
        }
    
    def get_consolidation_records(self, limit: int = 10) -> List[Dict]:
        """获取最近的压缩记录"""
        return self.consolidation_records[-limit:]
    
    def get_consolidation_stats(self) -> Dict:
        """获取压缩统计"""
        if not self.consolidation_records:
            return {
                'total_consolidations': 0,
                'total_compressed_memories': 0,
                'avg_compression_ratio': 0
            }
        
        return {
            'total_consolidations': len(self.consolidation_records),
            'total_compressed_memories': sum(
                r['original_count'] for r in self.consolidation_records
            ),
            'avg_compression_ratio': np.mean([
                r['compression_ratio'] for r in self. consolidation_records
            ]),
            'last_consolidation':  self.consolidation_records[-1]['timestamp']
        }
