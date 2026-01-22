"""
ViViT 多日序列数据集

用于 Video Vision Transformer 模型训练的 20 日序列数据集
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Callable, Tuple
from pathlib import Path

# 依赖导入
try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# 常量
IMAGE_SHAPE = (15, 8, 8)
IMAGE_DTYPE = np.float32
DEFAULT_SEQ_LEN = 20


def check_dependencies():
    """检查依赖"""
    if not HAS_LMDB:
        raise ImportError("lmdb 未安装，请运行: pip install lmdb")
    if not HAS_LZ4:
        raise ImportError("lz4 未安装，请运行: pip install lz4")


def decompress_image(compressed: bytes) -> np.ndarray:
    """解压图像"""
    raw_bytes = lz4.frame.decompress(compressed)
    return np.frombuffer(raw_bytes, dtype=IMAGE_DTYPE).reshape(IMAGE_SHAPE).copy()


class ViViTDataset(Dataset):
    """
    ViViT 多日序列数据集
    
    为每个 (date, stock) 组合构建前 N 天的图像序列
    
    Example:
        >>> dates = ["20230101", "20230102", ..., "20230131"]
        >>> codes = ["600519.SH", "000001.SZ"]
        >>> dataset = ViViTDataset("/data/lmdb", dates, codes, seq_len=20)
        >>> sequence = dataset[0]  # torch.Tensor [20, 15, 8, 8]
    """
    
    def __init__(
        self,
        lmdb_dir: str,
        trade_dates: List[str],
        stock_codes: List[str],
        seq_len: int = DEFAULT_SEQ_LEN,
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
        preload_envs: bool = True,
    ):
        """
        Args:
            lmdb_dir: LMDB 文件目录
            trade_dates: 交易日期列表（已排序）
            stock_codes: 股票代码列表
            seq_len: 序列长度，默认 20 天
            labels: 标签数组，shape = [len(dates), len(codes)] 或展平的 1D
            transform: 数据增强函数，作用于单张图像
            return_meta: 是否返回元信息 (date, code)
            preload_envs: 是否预加载所有 LMDB 环境
        """
        check_dependencies()
        
        self.lmdb_dir = Path(lmdb_dir)
        self.dates = trade_dates
        self.codes = stock_codes
        self.seq_len = seq_len
        self.labels = labels
        self.transform = transform
        self.return_meta = return_meta
        
        # 验证标签形状
        if labels is not None:
            expected_size = len(trade_dates) * len(stock_codes)
            if labels.size != expected_size:
                raise ValueError(
                    f"标签数量 ({labels.size}) 与样本数量 ({expected_size}) 不匹配"
                )
            # 确保是 2D 形状
            if labels.ndim == 1:
                self.labels = labels.reshape(len(trade_dates), len(stock_codes))
        
        # LMDB 环境缓存
        self.envs: Dict[str, lmdb.Environment] = {}
        
        if preload_envs:
            self._preload_envs()
    
    def _preload_envs(self):
        """预加载所有 LMDB 环境"""
        for date in self.dates:
            lmdb_path = self.lmdb_dir / f"{date}.lmdb"
            if lmdb_path.exists():
                self.envs[date] = lmdb.open(
                    str(lmdb_path),
                    readonly=True,
                    lock=False,
                    readahead=False,
                )
    
    def _get_env(self, date: str) -> Optional[lmdb.Environment]:
        """获取指定日期的 LMDB 环境（懒加载）"""
        if date not in self.envs:
            lmdb_path = self.lmdb_dir / f"{date}.lmdb"
            if lmdb_path.exists():
                self.envs[date] = lmdb.open(
                    str(lmdb_path),
                    readonly=True,
                    lock=False,
                    readahead=False,
                )
            else:
                return None
        return self.envs.get(date)
    
    def _read_image(self, env: lmdb.Environment, code: str) -> np.ndarray:
        """从 LMDB 读取单张图像"""
        with env.begin() as txn:
            compressed = txn.get(code.encode('utf-8'))
            if compressed is None:
                return np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
            return decompress_image(compressed)
    
    def __len__(self) -> int:
        return len(self.dates) * len(self.codes)
    
    def __getitem__(self, idx: int):
        # 计算 date 和 code 索引
        date_idx = idx // len(self.codes)
        code_idx = idx % len(self.codes)
        
        target_date = self.dates[date_idx]
        code = self.codes[code_idx]
        
        # 获取前 seq_len 天的日期（包含当天）
        start_idx = max(0, date_idx - self.seq_len + 1)
        date_range = self.dates[start_idx:date_idx + 1]
        
        # 读取图像序列
        images = []
        for d in date_range:
            env = self._get_env(d)
            if env is not None:
                img = self._read_image(env, code)
            else:
                img = np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
            
            # 应用数据增强
            if self.transform is not None:
                img = self.transform(img)
            
            images.append(img)
        
        # 不足 seq_len 天，前面补零
        while len(images) < self.seq_len:
            zero_img = np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
            if self.transform is not None:
                zero_img = self.transform(zero_img)
            images.insert(0, zero_img)
        
        # Stack 成 [seq_len, 15, 8, 8]
        sequence = np.stack(images, axis=0)
        tensor = torch.from_numpy(sequence)
        
        # 构建返回值
        result = [tensor]
        
        if self.labels is not None:
            label = self.labels[date_idx, code_idx]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            elif isinstance(label, np.floating):
                label = torch.tensor(label.item())
            elif isinstance(label, (int, float)):
                label = torch.tensor(label)
            else:
                label = torch.tensor(label)
            result.append(label)
        
        if self.return_meta:
            result.append((target_date, code))
        
        if len(result) == 1:
            return result[0]
        return tuple(result)
    
    def get_sequence(
        self,
        stock_code: str,
        target_date: str,
    ) -> np.ndarray:
        """
        获取指定股票在指定日期的序列
        
        Args:
            stock_code: 股票代码
            target_date: 目标日期
        
        Returns:
            [seq_len, 15, 8, 8] 序列
        """
        if target_date not in self.dates:
            raise ValueError(f"日期 {target_date} 不在数据集中")
        
        date_idx = self.dates.index(target_date)
        start_idx = max(0, date_idx - self.seq_len + 1)
        date_range = self.dates[start_idx:date_idx + 1]
        
        images = []
        for d in date_range:
            env = self._get_env(d)
            if env is not None:
                img = self._read_image(env, stock_code)
            else:
                img = np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
            images.append(img)
        
        while len(images) < self.seq_len:
            images.insert(0, np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE))
        
        return np.stack(images, axis=0)
    
    def list_available_dates(self) -> List[str]:
        """列出有 LMDB 文件的日期"""
        available = []
        for date in self.dates:
            lmdb_path = self.lmdb_dir / f"{date}.lmdb"
            if lmdb_path.exists():
                available.append(date)
        return available
    
    def close(self):
        """关闭所有 LMDB 环境"""
        for env in self.envs.values():
            env.close()
        self.envs.clear()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class ViViTDatasetByStock(Dataset):
    """
    按股票组织的 ViViT 数据集
    
    返回单只股票的完整时间序列
    
    Example:
        >>> dataset = ViViTDatasetByStock("/data/lmdb", dates, codes)
        >>> sequences = dataset[0]  # torch.Tensor [T, 15, 8, 8]，T = len(dates)
    """
    
    def __init__(
        self,
        lmdb_dir: str,
        trade_dates: List[str],
        stock_codes: List[str],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            lmdb_dir: LMDB 文件目录
            trade_dates: 交易日期列表
            stock_codes: 股票代码列表
            labels: 标签数组，shape = [len(codes)]
            transform: 数据增强函数
        """
        check_dependencies()
        
        self.lmdb_dir = Path(lmdb_dir)
        self.dates = trade_dates
        self.codes = stock_codes
        self.labels = labels
        self.transform = transform
        
        # 预加载 LMDB 环境
        self.envs: Dict[str, lmdb.Environment] = {}
        for date in self.dates:
            lmdb_path = self.lmdb_dir / f"{date}.lmdb"
            if lmdb_path.exists():
                self.envs[date] = lmdb.open(
                    str(lmdb_path),
                    readonly=True,
                    lock=False,
                )
    
    def __len__(self) -> int:
        return len(self.codes)
    
    def __getitem__(self, idx: int):
        code = self.codes[idx]
        
        images = []
        for date in self.dates:
            if date in self.envs:
                with self.envs[date].begin() as txn:
                    compressed = txn.get(code.encode('utf-8'))
                    if compressed is not None:
                        img = decompress_image(compressed)
                    else:
                        img = np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
            else:
                img = np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
            
            if self.transform is not None:
                img = self.transform(img)
            
            images.append(img)
        
        # [T, 15, 8, 8]
        sequence = np.stack(images, axis=0)
        tensor = torch.from_numpy(sequence)
        
        if self.labels is not None:
            label = self.labels[idx]
            return tensor, torch.tensor(label)
        
        return tensor
    
    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs.clear()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


def create_vivit_dataloader(
    lmdb_dir: str,
    trade_dates: List[str],
    stock_codes: List[str],
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    labels: Optional[np.ndarray] = None,
    transform: Optional[Callable] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    创建 ViViT DataLoader
    
    Args:
        lmdb_dir: LMDB 文件目录
        trade_dates: 交易日期列表
        stock_codes: 股票代码列表
        seq_len: 序列长度
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        labels: 标签数组
        transform: 数据增强
        **kwargs: 传递给 DataLoader 的其他参数
    
    Returns:
        DataLoader
    """
    dataset = ViViTDataset(
        lmdb_dir=lmdb_dir,
        trade_dates=trade_dates,
        stock_codes=stock_codes,
        seq_len=seq_len,
        labels=labels,
        transform=transform,
        preload_envs=True,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )
