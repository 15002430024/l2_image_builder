"""
存储层 - LMDB + LZ4 压缩存储

提供高效的图像存储和读取功能：
- write_daily_lmdb: 写入一天所有股票图像
- read_daily_lmdb: 读取 LMDB 文件
- LMDBWriter: 支持增量写入的写入器
- LMDBReader: 支持并发读取的读取器
- MultiDayLMDBReader: 多日数据管理
"""

from .lmdb_writer import (
    write_daily_lmdb,
    write_images_batch,
    append_to_lmdb,
    compress_image,
    decompress_image,
    get_lmdb_stats,
    LMDBWriter,
    IMAGE_SHAPE,
    IMAGE_DTYPE,
    IMAGE_SIZE_BYTES,
)

from .lmdb_reader import (
    read_daily_lmdb,
    read_single_stock,
    get_lmdb_keys,
    LMDBReader,
    MultiDayLMDBReader,
)


__all__ = [
    # 写入函数
    'write_daily_lmdb',
    'write_images_batch',
    'append_to_lmdb',
    # 读取函数
    'read_daily_lmdb',
    'read_single_stock',
    'get_lmdb_keys',
    # 压缩/解压函数
    'compress_image',
    'decompress_image',
    # 统计函数
    'get_lmdb_stats',
    # 类
    'LMDBWriter',
    'LMDBReader',
    'MultiDayLMDBReader',
    # 常量
    'IMAGE_SHAPE',
    'IMAGE_DTYPE',
    'IMAGE_SIZE_BYTES',
]
