"""
LMDB 写入模块

将 Level2 图像数据以 LZ4 压缩格式写入 LMDB
"""

import os
import numpy as np
from typing import Dict, Optional, List, Callable
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
IMAGE_SIZE_BYTES = 15 * 8 * 8 * 4  # 3840 bytes


def check_dependencies():
    """检查依赖是否安装"""
    if not HAS_LMDB:
        raise ImportError("lmdb 未安装，请运行: pip install lmdb")
    if not HAS_LZ4:
        raise ImportError("lz4 未安装，请运行: pip install lz4")


def compress_image(image: np.ndarray) -> bytes:
    """
    压缩图像为 LZ4 格式
    
    Args:
        image: [15, 8, 8] float32 数组
    
    Returns:
        LZ4 压缩后的 bytes
    """
    check_dependencies()
    
    # 确保类型和形状正确
    image = np.asarray(image, dtype=IMAGE_DTYPE)
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"Invalid image shape: {image.shape}, expected {IMAGE_SHAPE}")
    
    # 确保内存连续
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
    
    # 转为 bytes 并压缩
    raw_bytes = image.tobytes()
    compressed = lz4.frame.compress(raw_bytes)
    
    return compressed


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


def write_daily_lmdb(
    trade_date: str,
    stock_images: Dict[str, np.ndarray],
    output_dir: str,
    map_size: int = 100 * 1024 * 1024,
    overwrite: bool = False,
) -> str:
    """
    将一天所有股票的图像写入 LMDB
    
    Args:
        trade_date: 交易日期，如 "20230101"
        stock_images: {stock_code: image_array}，image 已归一化
        output_dir: 输出目录
        map_size: LMDB 映射大小，默认 100MB
        overwrite: 是否覆盖已存在的文件
    
    Returns:
        LMDB 文件路径
    
    Example:
        >>> images = {"600519.SH": np.random.rand(15, 8, 8).astype(np.float32)}
        >>> path = write_daily_lmdb("20230101", images, "/data/lmdb")
    """
    check_dependencies()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # LMDB 文件路径
    lmdb_path = output_path / f"{trade_date}.lmdb"
    
    # 检查是否已存在
    if lmdb_path.exists() and not overwrite:
        raise FileExistsError(f"LMDB 文件已存在: {lmdb_path}，设置 overwrite=True 以覆盖")
    
    # 如果覆盖，先删除
    if lmdb_path.exists() and overwrite:
        import shutil
        shutil.rmtree(lmdb_path)
    
    # 打开 LMDB 环境
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    
    written_count = 0
    errors = []
    
    try:
        with env.begin(write=True) as txn:
            for code, image in stock_images.items():
                try:
                    # 验证并压缩
                    compressed = compress_image(image)
                    
                    # 写入
                    key = code.encode('utf-8')
                    txn.put(key, compressed)
                    written_count += 1
                    
                except Exception as e:
                    errors.append((code, str(e)))
    finally:
        env.close()
    
    if errors:
        print(f"警告: {len(errors)} 只股票写入失败:")
        for code, err in errors[:5]:
            print(f"  {code}: {err}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors) - 5} 个错误")
    
    return str(lmdb_path)


def write_images_batch(
    trade_date: str,
    stock_codes: List[str],
    image_generator: Callable[[str], Optional[np.ndarray]],
    output_dir: str,
    map_size: int = 100 * 1024 * 1024,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    批量写入图像（使用生成器函数）
    
    适用于不想一次性将所有图像加载到内存的场景
    
    Args:
        trade_date: 交易日期
        stock_codes: 股票代码列表
        image_generator: 函数，接受 stock_code 返回图像或 None
        output_dir: 输出目录
        map_size: LMDB 映射大小
        progress_callback: 进度回调函数 (current, total)
    
    Returns:
        统计信息字典
    
    Example:
        >>> def gen(code):
        ...     return build_image(code)  # 你的图像构建函数
        >>> stats = write_images_batch("20230101", codes, gen, "/data/lmdb")
    """
    check_dependencies()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lmdb_path = output_path / f"{trade_date}.lmdb"
    
    # 删除已存在的
    if lmdb_path.exists():
        import shutil
        shutil.rmtree(lmdb_path)
    
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    
    stats = {
        'total': len(stock_codes),
        'written': 0,
        'skipped': 0,
        'errors': 0,
        'total_compressed_bytes': 0,
    }
    
    try:
        with env.begin(write=True) as txn:
            for i, code in enumerate(stock_codes):
                try:
                    image = image_generator(code)
                    
                    if image is None:
                        stats['skipped'] += 1
                        continue
                    
                    compressed = compress_image(image)
                    txn.put(code.encode('utf-8'), compressed)
                    
                    stats['written'] += 1
                    stats['total_compressed_bytes'] += len(compressed)
                    
                except Exception as e:
                    stats['errors'] += 1
                
                # 进度回调
                if progress_callback:
                    progress_callback(i + 1, len(stock_codes))
    finally:
        env.close()
    
    # 计算统计
    if stats['written'] > 0:
        stats['avg_compressed_bytes'] = stats['total_compressed_bytes'] / stats['written']
        stats['compression_ratio'] = IMAGE_SIZE_BYTES / stats['avg_compressed_bytes']
    
    return stats


def append_to_lmdb(
    lmdb_path: str,
    stock_images: Dict[str, np.ndarray],
    map_size: int = 100 * 1024 * 1024,
) -> int:
    """
    向已存在的 LMDB 追加数据
    
    Args:
        lmdb_path: LMDB 文件路径
        stock_images: {stock_code: image_array}
        map_size: 新的映射大小（如果需要扩容）
    
    Returns:
        新写入的记录数
    """
    check_dependencies()
    
    if not Path(lmdb_path).exists():
        raise FileNotFoundError(f"LMDB 文件不存在: {lmdb_path}")
    
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    written = 0
    try:
        with env.begin(write=True) as txn:
            for code, image in stock_images.items():
                compressed = compress_image(image)
                txn.put(code.encode('utf-8'), compressed)
                written += 1
    finally:
        env.close()
    
    return written


class LMDBWriter:
    """
    LMDB 写入器类
    
    支持增量写入和上下文管理
    """
    
    def __init__(
        self,
        lmdb_path: str,
        map_size: int = 100 * 1024 * 1024,
        create: bool = True,
    ):
        """
        Args:
            lmdb_path: LMDB 文件路径
            map_size: LMDB 映射大小
            create: 如果不存在是否创建
        """
        check_dependencies()
        
        self.lmdb_path = Path(lmdb_path)
        self.map_size = map_size
        
        if create:
            self.lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.env = lmdb.open(str(self.lmdb_path), map_size=map_size)
        self._written_count = 0
    
    def write(self, stock_code: str, image: np.ndarray) -> bool:
        """
        写入单只股票图像
        
        Returns:
            是否成功
        """
        try:
            compressed = compress_image(image)
            with self.env.begin(write=True) as txn:
                txn.put(stock_code.encode('utf-8'), compressed)
            self._written_count += 1
            return True
        except Exception:
            return False
    
    def write_batch(self, stock_images: Dict[str, np.ndarray]) -> int:
        """
        批量写入
        
        Returns:
            成功写入数量
        """
        written = 0
        with self.env.begin(write=True) as txn:
            for code, image in stock_images.items():
                try:
                    compressed = compress_image(image)
                    txn.put(code.encode('utf-8'), compressed)
                    written += 1
                except Exception:
                    pass
        self._written_count += written
        return written
    
    @property
    def written_count(self) -> int:
        """已写入的记录数"""
        return self._written_count
    
    def close(self):
        """关闭 LMDB 环境"""
        self.env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def get_lmdb_stats(lmdb_path: str) -> Dict:
    """
    获取 LMDB 文件统计信息
    
    Args:
        lmdb_path: LMDB 文件路径
    
    Returns:
        统计信息字典
    """
    check_dependencies()
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    try:
        stat = env.stat()
        info = env.info()
        
        # 计算存储效率
        total_compressed = 0
        record_count = 0
        
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                total_compressed += len(value)
                record_count += 1
        
        stats = {
            'entries': stat['entries'],
            'psize': stat['psize'],  # 页大小
            'depth': stat['depth'],  # B树深度
            'branch_pages': stat['branch_pages'],
            'leaf_pages': stat['leaf_pages'],
            'overflow_pages': stat['overflow_pages'],
            'map_size': info['map_size'],
            'total_compressed_bytes': total_compressed,
            'avg_compressed_bytes': total_compressed / record_count if record_count > 0 else 0,
            'compression_ratio': IMAGE_SIZE_BYTES * record_count / total_compressed if total_compressed > 0 else 0,
        }
        
        return stats
    finally:
        env.close()
