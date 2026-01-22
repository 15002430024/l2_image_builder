"""
ViT 单日数据集

用于 Vision Transformer 模型训练的单日 Level2 图像数据集
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Callable, Tuple

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


class ViTDataset(Dataset):
    """
    ViT 单日数据集
    
    从 LMDB 文件中读取图像数据，支持标签和数据增强
    
    Example:
        >>> dataset = ViTDataset("/data/lmdb/20230101.lmdb", stock_codes)
        >>> image = dataset[0]  # torch.Tensor [15, 8, 8]
        
        >>> # 带标签
        >>> dataset = ViTDataset(lmdb_path, codes, labels=labels)
        >>> image, label = dataset[0]
    """
    
    def __init__(
        self,
        lmdb_path: str,
        stock_codes: List[str],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        return_code: bool = False,
    ):
        """
        Args:
            lmdb_path: LMDB 文件路径
            stock_codes: 股票代码列表
            labels: 标签数组，长度与 stock_codes 一致
            transform: 数据增强函数
            return_code: 是否返回股票代码
        """
        check_dependencies()
        
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB 文件不存在: {lmdb_path}")
        
        self.lmdb_path = lmdb_path
        self.codes = stock_codes
        self.labels = labels
        self.transform = transform
        self.return_code = return_code
        
        if labels is not None and len(labels) != len(stock_codes):
            raise ValueError(f"标签数量 ({len(labels)}) 与股票数量 ({len(stock_codes)}) 不匹配")
        
        # 打开 LMDB 环境
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=128,
        )
        
        # 缓存 transaction
        self._txn = None
    
    @property
    def txn(self):
        """懒加载 transaction"""
        if self._txn is None:
            self._txn = self.env.begin()
        return self._txn
    
    def __len__(self) -> int:
        return len(self.codes)
    
    def __getitem__(self, idx: int):
        code = self.codes[idx]
        
        # 读取压缩数据
        compressed = self.txn.get(code.encode('utf-8'))
        
        if compressed is None:
            # 股票不存在，返回全零图像
            image = np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
        else:
            image = decompress_image(compressed)
        
        # 应用数据增强
        if self.transform is not None:
            image = self.transform(image)
        
        # 转为 Tensor
        tensor = torch.from_numpy(image)
        
        # 构建返回值
        result = [tensor]
        
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            elif isinstance(label, (int, float)):
                label = torch.tensor(label)
            result.append(label)
        
        if self.return_code:
            result.append(code)
        
        # 单个返回值时不用 tuple
        if len(result) == 1:
            return result[0]
        return tuple(result)
    
    def get_image(self, stock_code: str) -> Optional[np.ndarray]:
        """
        通过股票代码获取图像
        
        Args:
            stock_code: 股票代码
        
        Returns:
            图像数组或 None
        """
        compressed = self.txn.get(stock_code.encode('utf-8'))
        if compressed is None:
            return None
        return decompress_image(compressed)
    
    def close(self):
        """关闭 LMDB 环境"""
        if self._txn is not None:
            self._txn = None
        self.env.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class ViTDatasetWithMask(ViTDataset):
    """
    带有效性掩码的 ViT 数据集
    
    返回 (image, mask, label) 其中 mask 指示该图像是否有效
    """
    
    def __getitem__(self, idx: int):
        code = self.codes[idx]
        
        # 读取压缩数据
        compressed = self.txn.get(code.encode('utf-8'))
        
        if compressed is None:
            image = np.zeros(IMAGE_SHAPE, dtype=IMAGE_DTYPE)
            mask = False
        else:
            image = decompress_image(compressed)
            mask = True
        
        if self.transform is not None:
            image = self.transform(image)
        
        tensor = torch.from_numpy(image)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        
        result = [tensor, mask_tensor]
        
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            elif isinstance(label, (int, float)):
                label = torch.tensor(label)
            result.append(label)
        
        if self.return_code:
            result.append(code)
        
        return tuple(result)


def create_vit_dataloader(
    lmdb_path: str,
    stock_codes: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    labels: Optional[np.ndarray] = None,
    transform: Optional[Callable] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    创建 ViT DataLoader
    
    Args:
        lmdb_path: LMDB 文件路径
        stock_codes: 股票代码列表
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        labels: 标签数组
        transform: 数据增强
        **kwargs: 传递给 DataLoader 的其他参数
    
    Returns:
        DataLoader
    """
    dataset = ViTDataset(
        lmdb_path=lmdb_path,
        stock_codes=stock_codes,
        labels=labels,
        transform=transform,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )
