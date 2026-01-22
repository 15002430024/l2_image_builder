"""
LMDB 读取模块

支持并发读取的 LMDB 读取器
"""

import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple
from pathlib import Path

# LMDB 和 LZ4 导入
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
    """检查依赖是否安装"""
    if not HAS_LMDB:
        raise ImportError("lmdb 未安装，请运行: pip install lmdb")
    if not HAS_LZ4:
        raise ImportError("lz4 未安装，请运行: pip install lz4")


def decompress_image(compressed: bytes) -> np.ndarray:
    """
    解压 LZ4 压缩的图像数据
    
    Args:
        compressed: LZ4 压缩的 bytes
    
    Returns:
        [15, 8, 8] float32 数组
    """
    check_dependencies()
    
    raw_bytes = lz4.frame.decompress(compressed)
    image = np.frombuffer(raw_bytes, dtype=IMAGE_DTYPE).reshape(IMAGE_SHAPE)
    
    # 必须 copy，因为 frombuffer 返回的是 view
    return image.copy()


class LMDBReader:
    """
    LMDB 读取器
    
    支持并发读取和上下文管理
    
    Example:
        >>> with LMDBReader("/data/lmdb/20230101.lmdb") as reader:
        ...     image = reader.read("600519.SH")
        ...     keys = reader.list_keys()
        ...     print(len(reader))
    """
    
    def __init__(
        self,
        lmdb_path: str,
        readonly: bool = True,
        lock: bool = False,
        max_readers: int = 126,
    ):
        """
        Args:
            lmdb_path: LMDB 文件路径
            readonly: 是否只读模式
            lock: 是否使用文件锁
            max_readers: 最大并发读取数
        """
        check_dependencies()
        
        self.lmdb_path = Path(lmdb_path)
        
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB 文件不存在: {lmdb_path}")
        
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=readonly,
            lock=lock,
            max_readers=max_readers,
            readahead=False,  # 对随机访问更友好
        )
        
        # 缓存 key 列表
        self._keys_cache: Optional[List[str]] = None
    
    def read(self, stock_code: str) -> Optional[np.ndarray]:
        """
        读取单只股票图像
        
        Args:
            stock_code: 股票代码，如 "600519.SH"
        
        Returns:
            [15, 8, 8] float32 数组，如果不存在返回 None
        """
        with self.env.begin() as txn:
            value = txn.get(stock_code.encode('utf-8'))
            
            if value is None:
                return None
            
            return decompress_image(value)
    
    def read_batch(self, stock_codes: List[str]) -> Dict[str, np.ndarray]:
        """
        批量读取
        
        Args:
            stock_codes: 股票代码列表
        
        Returns:
            {stock_code: image}，不存在的股票不会出现在结果中
        """
        result = {}
        
        with self.env.begin() as txn:
            for code in stock_codes:
                value = txn.get(code.encode('utf-8'))
                if value is not None:
                    result[code] = decompress_image(value)
        
        return result
    
    def list_keys(self) -> List[str]:
        """
        列出所有股票代码
        
        Returns:
            股票代码列表
        """
        if self._keys_cache is not None:
            return self._keys_cache.copy()
        
        keys = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                keys.append(key.decode('utf-8'))
        
        self._keys_cache = keys
        return keys.copy()
    
    def has_key(self, stock_code: str) -> bool:
        """检查股票代码是否存在"""
        with self.env.begin() as txn:
            return txn.get(stock_code.encode('utf-8')) is not None
    
    def iter_items(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        迭代所有记录
        
        Yields:
            (stock_code, image) 元组
        """
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                code = key.decode('utf-8')
                image = decompress_image(value)
                yield code, image
    
    def get_stats(self) -> Dict:
        """获取 LMDB 统计信息"""
        stat = self.env.stat()
        info = self.env.info()
        
        return {
            'entries': stat['entries'],
            'psize': stat['psize'],
            'depth': stat['depth'],
            'map_size': info['map_size'],
        }
    
    def __len__(self) -> int:
        """返回记录数"""
        stat = self.env.stat()
        return stat['entries']
    
    def __contains__(self, stock_code: str) -> bool:
        """支持 in 操作符"""
        return self.has_key(stock_code)
    
    def __getitem__(self, stock_code: str) -> np.ndarray:
        """支持索引访问"""
        image = self.read(stock_code)
        if image is None:
            raise KeyError(f"股票代码不存在: {stock_code}")
        return image
    
    def close(self):
        """关闭 LMDB 环境"""
        self.env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def read_daily_lmdb(
    lmdb_path: str,
    stock_codes: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    读取 LMDB 文件中的图像
    
    便捷函数，自动打开和关闭 LMDB
    
    Args:
        lmdb_path: LMDB 文件路径
        stock_codes: 要读取的股票列表，None 表示读取全部
    
    Returns:
        {stock_code: image}
    """
    with LMDBReader(lmdb_path) as reader:
        if stock_codes is None:
            return dict(reader.iter_items())
        else:
            return reader.read_batch(stock_codes)


def read_single_stock(lmdb_path: str, stock_code: str) -> Optional[np.ndarray]:
    """
    读取单只股票图像
    
    便捷函数
    
    Args:
        lmdb_path: LMDB 文件路径
        stock_code: 股票代码
    
    Returns:
        图像数组或 None
    """
    with LMDBReader(lmdb_path) as reader:
        return reader.read(stock_code)


def get_lmdb_keys(lmdb_path: str) -> List[str]:
    """
    获取 LMDB 中所有股票代码
    
    便捷函数
    
    Args:
        lmdb_path: LMDB 文件路径
    
    Returns:
        股票代码列表
    """
    with LMDBReader(lmdb_path) as reader:
        return reader.list_keys()


class MultiDayLMDBReader:
    """
    多日 LMDB 读取器
    
    管理多个日期的 LMDB 文件
    
    Example:
        >>> reader = MultiDayLMDBReader("/data/lmdb")
        >>> reader.load_dates(["20230101", "20230102"])
        >>> image = reader.read("600519.SH", "20230101")
    """
    
    def __init__(self, lmdb_dir: str):
        """
        Args:
            lmdb_dir: LMDB 文件目录
        """
        self.lmdb_dir = Path(lmdb_dir)
        self._readers: Dict[str, LMDBReader] = {}
    
    def load_dates(self, dates: List[str]):
        """
        加载指定日期的 LMDB
        
        Args:
            dates: 日期列表，如 ["20230101", "20230102"]
        """
        for date in dates:
            lmdb_path = self.lmdb_dir / f"{date}.lmdb"
            if lmdb_path.exists():
                self._readers[date] = LMDBReader(str(lmdb_path))
    
    def list_available_dates(self) -> List[str]:
        """列出目录中所有可用的日期"""
        dates = []
        for path in self.lmdb_dir.glob("*.lmdb"):
            date = path.stem
            if date.isdigit() and len(date) == 8:
                dates.append(date)
        return sorted(dates)
    
    def read(self, stock_code: str, date: str) -> Optional[np.ndarray]:
        """
        读取指定日期的股票图像
        
        Args:
            stock_code: 股票代码
            date: 日期
        
        Returns:
            图像数组或 None
        """
        if date not in self._readers:
            lmdb_path = self.lmdb_dir / f"{date}.lmdb"
            if not lmdb_path.exists():
                return None
            self._readers[date] = LMDBReader(str(lmdb_path))
        
        return self._readers[date].read(stock_code)
    
    def close(self):
        """关闭所有 reader"""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
