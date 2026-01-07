"""
压缩配置管理
支持从环境变量、配置文件或代码配置
"""
import os
import json
import logging
from typing import Optional, Dict
from pathlib import Path
from dataclasses import asdict

from memory_consolidation import ConsolidationConfig

logger = logging.getLogger(__name__)


class CompressionConfigManager:
    """压缩配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径（JSON格式）
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> ConsolidationConfig:
        """加载配置"""
        
        # 1. 从环境变量加载
        env_config = self._load_from_env()
        
        # 2. 从配置文件加载（覆盖环境变量）
        if self.config_file and Path(self.config_file).exists():
            file_config = self._load_from_file(self.config_file)
            env_config.update(file_config)
        
        return ConsolidationConfig(**env_config)
    
    def _load_from_env(self) -> Dict:
        """从环境变量加载配置"""
        return {
            'max_memories_per_agent': int(os.getenv('CONSOLIDATION_MAX_MEMORIES', '1000')),
            'max_memory_size_mb':  int(os.getenv('CONSOLIDATION_MAX_SIZE_MB', '500')),
            'consolidation_interval_hours': int(os.getenv('CONSOLIDATION_INTERVAL_HOURS', '24')),
            'similarity_threshold': float(os.getenv('CONSOLIDATION_SIMILARITY_THRESHOLD', '0.75')),
            'time_window_days': int(os.getenv('CONSOLIDATION_TIME_WINDOW_DAYS', '7')),
            'min_cluster_size': int(os.getenv('CONSOLIDATION_MIN_CLUSTER_SIZE', '3')),
            'protect_recent_days': int(os.getenv('CONSOLIDATION_PROTECT_RECENT_DAYS', '7')),
            'summary_max_length': int(os.getenv('CONSOLIDATION_SUMMARY_MAX_LENGTH', '300')),
        }
    
    def _load_from_file(self, config_file: str) -> Dict:
        """从JSON配置文件加载"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
            return {}
    
    def save_config(self, output_file: str):
        """保存当前配置到文件"""
        try: 
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self. config), f, indent=2, ensure_ascii=False)
            logger.info(f"Config saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self. config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger. warning(f"Unknown config key: {key}")
    
    def get_config(self) -> ConsolidationConfig:
        """获取当前配置"""
        return self.config
    
    def get_config_dict(self) -> Dict:
        """获取配置字典"""
        return asdict(self. config)
    
    def __repr__(self):
        return f"CompressionConfigManager(\n{json.dumps(self.get_config_dict(), indent=2)})"


# 全局配置实例
_config_manager = None


def get_compression_config(config_file: Optional[str] = None) -> CompressionConfig:
    """获取全局压缩配置"""
    global _config_manager
    if _config_manager is None: 
        _config_manager = CompressionConfigManager(config_file)
    return _config_manager.get_config()
