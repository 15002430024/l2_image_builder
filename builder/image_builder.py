"""
Level2 图像构建器

核心类，负责将 Level2 数据转换为 [15, 8, 8] 的三维图像

v3 架构更新：
- 上交所：委托表必须包含 IsAggressive 和 BizIndex 字段
- 深交所：自动构建 ActiveSeqs 实现主动/被动分流
- 新增通道约束验证（Ch7=Ch9+Ch11, Ch8=Ch10+Ch12）
"""

import logging
from typing import Dict, Optional, Tuple, Set, Union
import numpy as np

from ..config import Config, Channels
from ..data_loader.polars_utils import is_polars_df, DataFrame, POLARS_AVAILABLE
from ..calculator.quantile import (
    QuantileCalculator, 
    get_bin_index,
    compute_quantile_bins_sh_polars,
    compute_quantile_bins_sz_polars,
    compute_quantile_bins_sh_pandas,
    compute_quantile_bins_sz_pandas,
    compute_separate_quantile_bins_sh_polars,
    compute_separate_quantile_bins_sh_pandas,
    compute_separate_quantile_bins_sz_polars,
    compute_separate_quantile_bins_sz_pandas,
)
from ..calculator.big_order import BigOrderCalculator, compute_all
from ..cleaner.sz_cancel_enricher import enrich_sz_cancel_price
from ..diagnostics.reporter import validate_channel_constraints
from .normalizer import normalize_image
from .sh_builder import SHImageBuilder
from .sz_builder import SZImageBuilder, build_active_seqs_from_trade

# 配置日志
logger = logging.getLogger(__name__)

if POLARS_AVAILABLE:
    import polars as pl
import pandas as pd


class Level2ImageBuilder:
    """
    单只股票单日的图像构建器
    
    将 Level2 数据转换为标准化的三维图像格式 [15, 8, 8]
    
    通道定义：
        0: 全部成交
        1: 主动买入成交
        2: 主动卖出成交
        3: 大买单
        4: 大卖单
        5: 小买单
        6: 小卖单
        7: 买单委托
        8: 卖单委托
        9: 主动买入委托
        10: 主动卖出委托
        11: 非主动买入
        12: 非主动卖出
        13: 撤买
        14: 撤卖
    """
    
    def __init__(
        self,
        stock_code: str,
        trade_date: str,
        config: Optional[Config] = None,
    ):
        """
        Args:
            stock_code: 股票代码
            trade_date: 交易日期
            config: 配置对象
        """
        self.stock_code = stock_code
        self.trade_date = trade_date
        self.config = config or Config()
        
        # 初始化图像
        self.image = np.zeros(self.config.image_shape, dtype=np.float32)
        
        # 判断交易所
        self._is_sh = stock_code.endswith('.SH') or stock_code.startswith('6')
    
    def build(
        self,
        df_trade: DataFrame,
        df_order: DataFrame,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        buy_parent_amount: Dict[int, float],
        sell_parent_amount: Dict[int, float],
        threshold: float,
        active_seqs: Optional[Dict[str, Set[int]]] = None,
    ) -> np.ndarray:
        """
        构建图像
        
        Args:
            df_trade: 成交数据
            df_order: 委托数据
            price_bins: 价格分位数边界
            qty_bins: 量分位数边界
            buy_parent_amount: 买方母单金额
            sell_parent_amount: 卖方母单金额
            threshold: 大单阈值
            active_seqs: 主动委托序列号集合（深交所需要）
        
        Returns:
            归一化后的 [15, 8, 8] 图像
        """
        # 重置图像
        self.image.fill(0)
        
        if self._is_sh:
            self._build_sh(
                df_trade, df_order,
                price_bins, qty_bins,
                buy_parent_amount, sell_parent_amount,
                threshold,
            )
        else:
            self._build_sz(
                df_trade, df_order,
                price_bins, qty_bins,
                buy_parent_amount, sell_parent_amount,
                threshold,
                active_seqs or {'buy': set(), 'sell': set()},
            )
        
        return self.normalize()
    
    def _build_sh(
        self,
        df_trade: DataFrame,
        df_order: DataFrame,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        buy_parent_amount: Dict[int, float],
        sell_parent_amount: Dict[int, float],
        threshold: float,
    ):
        """上交所图像构建"""
        # 处理成交数据
        self._process_sh_trade(
            df_trade, price_bins, qty_bins,
            buy_parent_amount, sell_parent_amount, threshold,
        )
        
        # 处理委托数据（已预处理）
        self._process_sh_order(df_order, price_bins, qty_bins)
    
    def _process_sh_trade(
        self,
        df_trade: DataFrame,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        buy_parent_amount: Dict[int, float],
        sell_parent_amount: Dict[int, float],
        threshold: float,
    ):
        """处理上交所成交数据"""
        if is_polars_df(df_trade):
            iterator = df_trade.iter_rows(named=True)
        else:
            iterator = df_trade.iterrows()
            iterator = (row for _, row in iterator)
        
        for row in iterator:
            if is_polars_df(df_trade):
                price = row['Price']
                qty = row['Qty']
                bs_flag = row['TickBSFlag']
                buy_order_no = row['BuyOrderNO']
                sell_order_no = row['SellOrderNO']
            else:
                price = row['Price']
                qty = row['Qty']
                bs_flag = row['TickBSFlag']
                buy_order_no = row['BuyOrderNO']
                sell_order_no = row['SellOrderNO']
            
            # 计算 bin
            price_bin = get_bin_index(price, price_bins)
            qty_bin = get_bin_index(qty, qty_bins)
            
            # 通道0: 全部成交
            self.image[Channels.ALL_TRADE, price_bin, qty_bin] += 1
            
            # 通道1-2, 9-10: 根据主动方向
            if bs_flag == 'B':
                self.image[Channels.ACTIVE_BUY_TRADE, price_bin, qty_bin] += 1
                self.image[Channels.ACTIVE_BUY_ORDER, price_bin, qty_bin] += 1
            elif bs_flag == 'S':
                self.image[Channels.ACTIVE_SELL_TRADE, price_bin, qty_bin] += 1
                self.image[Channels.ACTIVE_SELL_ORDER, price_bin, qty_bin] += 1
            
            # 通道3-6: 大小单判定（与主动方向无关）
            # 买方
            buy_amount = buy_parent_amount.get(buy_order_no, 0)
            if buy_amount >= threshold:
                self.image[Channels.BIG_BUY, price_bin, qty_bin] += 1
            else:
                self.image[Channels.SMALL_BUY, price_bin, qty_bin] += 1
            
            # 卖方
            sell_amount = sell_parent_amount.get(sell_order_no, 0)
            if sell_amount >= threshold:
                self.image[Channels.BIG_SELL, price_bin, qty_bin] += 1
            else:
                self.image[Channels.SMALL_SELL, price_bin, qty_bin] += 1
    
    def _process_sh_order(
        self,
        df_order: DataFrame,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
    ):
        """
        处理上交所委托数据（已预处理）
        
        OrdType='New': 填充通道7/8和11/12
        OrdType='Cancel': 填充通道13/14
        """
        if is_polars_df(df_order):
            iterator = df_order.iter_rows(named=True)
        else:
            iterator = (row for _, row in df_order.iterrows())
        
        for row in iterator:
            if is_polars_df(df_order):
                ord_type = row['OrdType']
                side = row['Side']
                price = row['Price']
                qty = row['Qty']
            else:
                ord_type = row['OrdType']
                side = row['Side']
                price = row['Price']
                qty = row['Qty']
            
            # 跳过价格为0的记录（除非是特殊处理后）
            if price <= 0:
                continue
            
            price_bin = get_bin_index(price, price_bins)
            qty_bin = get_bin_index(qty, qty_bins)
            
            if ord_type == 'New':
                # 新增委托
                if side == 'B':
                    self.image[Channels.BUY_ORDER, price_bin, qty_bin] += 1
                    self.image[Channels.PASSIVE_BUY, price_bin, qty_bin] += 1
                elif side == 'S':
                    self.image[Channels.SELL_ORDER, price_bin, qty_bin] += 1
                    self.image[Channels.PASSIVE_SELL, price_bin, qty_bin] += 1
            
            elif ord_type == 'Cancel':
                # 撤单
                if side == 'B':
                    self.image[Channels.CANCEL_BUY, price_bin, qty_bin] += 1
                elif side == 'S':
                    self.image[Channels.CANCEL_SELL, price_bin, qty_bin] += 1
    
    def _build_sz(
        self,
        df_trade: DataFrame,
        df_order: DataFrame,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        buy_parent_amount: Dict[int, float],
        sell_parent_amount: Dict[int, float],
        threshold: float,
        active_seqs: Dict[str, Set[int]],
    ):
        """深交所图像构建"""
        # 处理成交数据（包括撤单）
        self._process_sz_trade(
            df_trade, price_bins, qty_bins,
            buy_parent_amount, sell_parent_amount, threshold,
        )
        
        # 处理委托数据
        self._process_sz_order(df_order, price_bins, qty_bins, active_seqs)
    
    def _process_sz_trade(
        self,
        df_trade: DataFrame,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        buy_parent_amount: Dict[int, float],
        sell_parent_amount: Dict[int, float],
        threshold: float,
    ):
        """处理深交所成交/撤单数据"""
        if is_polars_df(df_trade):
            iterator = df_trade.iter_rows(named=True)
        else:
            iterator = (row for _, row in df_trade.iterrows())
        
        for row in iterator:
            if is_polars_df(df_trade):
                exec_type = row['ExecType']
                price = row['LastPx']
                qty = row['LastQty']
                bid_seq = row['BidApplSeqNum']
                offer_seq = row['OfferApplSeqNum']
            else:
                exec_type = row['ExecType']
                price = row['LastPx']
                qty = row['LastQty']
                bid_seq = row['BidApplSeqNum']
                offer_seq = row['OfferApplSeqNum']
            
            if price <= 0:
                continue
            
            price_bin = get_bin_index(price, price_bins)
            qty_bin = get_bin_index(qty, qty_bins)
            
            if exec_type == '70':  # 成交
                # 通道0: 全部成交
                self.image[Channels.ALL_TRADE, price_bin, qty_bin] += 1
                
                # 通道1-2, 9-10: 根据主动方向
                if bid_seq > offer_seq:
                    # 买方主动
                    self.image[Channels.ACTIVE_BUY_TRADE, price_bin, qty_bin] += 1
                    self.image[Channels.ACTIVE_BUY_ORDER, price_bin, qty_bin] += 1
                elif offer_seq > bid_seq:
                    # 卖方主动
                    self.image[Channels.ACTIVE_SELL_TRADE, price_bin, qty_bin] += 1
                    self.image[Channels.ACTIVE_SELL_ORDER, price_bin, qty_bin] += 1
                
                # 通道3-6: 大小单
                buy_amount = buy_parent_amount.get(bid_seq, 0)
                if buy_amount >= threshold:
                    self.image[Channels.BIG_BUY, price_bin, qty_bin] += 1
                else:
                    self.image[Channels.SMALL_BUY, price_bin, qty_bin] += 1
                
                sell_amount = sell_parent_amount.get(offer_seq, 0)
                if sell_amount >= threshold:
                    self.image[Channels.BIG_SELL, price_bin, qty_bin] += 1
                else:
                    self.image[Channels.SMALL_SELL, price_bin, qty_bin] += 1
            
            elif exec_type == '52':  # 撤单
                # 通道13-14: 撤单
                if bid_seq > 0 and offer_seq == 0:
                    self.image[Channels.CANCEL_BUY, price_bin, qty_bin] += 1
                elif offer_seq > 0 and bid_seq == 0:
                    self.image[Channels.CANCEL_SELL, price_bin, qty_bin] += 1
    
    def _process_sz_order(
        self,
        df_order: DataFrame,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        active_seqs: Dict[str, Set[int]],
    ):
        """处理深交所委托数据"""
        if is_polars_df(df_order):
            iterator = df_order.iter_rows(named=True)
        else:
            iterator = (row for _, row in df_order.iterrows())
        
        for row in iterator:
            if is_polars_df(df_order):
                side = row['Side']
                price = row['Price']
                qty = row['Qty']  # v3: 深交所 OrderQty 已在 Loader 层归一化为 Qty
                appl_seq = row['ApplSeqNum']
            else:
                side = row['Side']
                price = row['Price']
                qty = row['Qty']  # v3: 深交所 OrderQty 已在 Loader 层归一化为 Qty
                appl_seq = row['ApplSeqNum']
            
            if price <= 0:
                continue
            
            price_bin = get_bin_index(price, price_bins)
            qty_bin = get_bin_index(qty, qty_bins)
            
            if side == '49':  # 买入
                self.image[Channels.BUY_ORDER, price_bin, qty_bin] += 1
                # 非主动买入：未作为主动方成交
                if appl_seq not in active_seqs['buy']:
                    self.image[Channels.PASSIVE_BUY, price_bin, qty_bin] += 1
            
            elif side == '50':  # 卖出
                self.image[Channels.SELL_ORDER, price_bin, qty_bin] += 1
                # 非主动卖出
                if appl_seq not in active_seqs['sell']:
                    self.image[Channels.PASSIVE_SELL, price_bin, qty_bin] += 1
    
    def normalize(self) -> np.ndarray:
        """
        归一化图像
        
        Returns:
            归一化后的图像
        """
        return normalize_image(self.image)
    
    def get_raw_image(self) -> np.ndarray:
        """获取原始（未归一化）图像"""
        return self.image.copy()
    
    def get_channel_stats(self) -> dict:
        """获取通道统计信息"""
        stats = {}
        for ch in range(self.config.num_channels):
            channel_data = self.image[ch]
            stats[self.config.channel_names[ch]] = {
                'sum': float(channel_data.sum()),
                'max': float(channel_data.max()),
                'nonzero': int(np.count_nonzero(channel_data)),
                'fill_rate': float(np.count_nonzero(channel_data) / 64),
            }
        return stats
    
    def build_single_stock(
        self,
        df_trade: DataFrame,
        df_order: DataFrame,
        use_vectorized: bool = True,
        validate_constraints: bool = True,
    ) -> Optional[np.ndarray]:
        """
        构建单只股票的图像（统一入口）
        
        自动完成：分位数计算 → 母单还原 → 阈值计算 → 图像构建 → 约束验证 → 归一化
        
        v3 架构变更：
        - 上交所：委托表必须包含 IsAggressive 和 BizIndex 字段
        - 深交所：自动构建 ActiveSeqs 实现主动/被动分流
        - 新增通道约束验证（Ch7=Ch9+Ch11, Ch8=Ch10+Ch12）
        
        Args:
            df_trade: 成交表（已清洗）
            df_order: 委托表（已清洗）
            use_vectorized: 是否使用向量化方法（推荐）
            validate_constraints: 是否验证 v3 通道约束（默认 True）
        
        Returns:
            归一化后的 [15, 8, 8] 图像，若数据为空返回 None
        
        Raises:
            ValueError: 上交所委托表缺少必需字段时抛出
        
        Example:
            >>> builder = Level2ImageBuilder("600519.SH", "2026-01-21")
            >>> image = builder.build_single_stock(df_trade, df_order)
            >>> image.shape
            (15, 8, 8)
        """
        # 检查数据是否为空
        trade_len = len(df_trade) if hasattr(df_trade, '__len__') else 0
        order_len = len(df_order) if hasattr(df_order, '__len__') else 0
        
        if trade_len == 0 and order_len == 0:
            return None
        
        is_polars = is_polars_df(df_trade) or is_polars_df(df_order)
        
        # v3: 上交所委托表字段验证（必须包含 IsAggressive 和 BizIndex）
        if self._is_sh:
            required_fields = ['IsAggressive', 'BizIndex']
            if is_polars:
                order_columns = df_order.columns
            else:
                order_columns = list(df_order.columns)
            missing = [f for f in required_fields if f not in order_columns]
            if missing:
                raise ValueError(
                    f"上交所委托表缺少必需字段: {missing}。"
                    f"请确保数据已预处理。BizIndex 是排序必需字段！"
                )
        
        # 1. 计算分位数
        # 根据配置选择分离或联合模式
        if self.config.separate_quantile_bins:
            # 分离模式：成交和委托独立计算分位数
            if self._is_sh:
                if is_polars:
                    tp_bins, tq_bins, op_bins, oq_bins = compute_separate_quantile_bins_sh_polars(df_trade, df_order)
                else:
                    tp_bins, tq_bins, op_bins, oq_bins = compute_separate_quantile_bins_sh_pandas(df_trade, df_order)
            else:
                if is_polars:
                    tp_bins, tq_bins, op_bins, oq_bins = compute_separate_quantile_bins_sz_polars(df_trade, df_order)
                else:
                    tp_bins, tq_bins, op_bins, oq_bins = compute_separate_quantile_bins_sz_pandas(df_trade, df_order)
            
            trade_price_bins, trade_qty_bins = tp_bins, tq_bins
            order_price_bins, order_qty_bins = op_bins, oq_bins
        else:
            # 联合模式：成交+委托统一分位数（原有逻辑）
            if self._is_sh:
                if is_polars:
                    price_bins, qty_bins = compute_quantile_bins_sh_polars(df_trade, df_order)
                else:
                    price_bins, qty_bins = compute_quantile_bins_sh_pandas(df_trade, df_order)
            else:
                if is_polars:
                    price_bins, qty_bins = compute_quantile_bins_sz_polars(df_trade, df_order)
                else:
                    price_bins, qty_bins = compute_quantile_bins_sz_pandas(df_trade, df_order)
            
            # 联合模式下，成交和委托使用相同的分位数
            trade_price_bins = order_price_bins = price_bins
            trade_qty_bins = order_qty_bins = qty_bins
        
        # 2. 母单还原 + 当日阈值
        exchange = 'sh' if self._is_sh else 'sz'
        buy_parent, sell_parent, threshold = compute_all(
            df_trade, exchange, self.config.threshold_std_multiplier
        )
        
        # 3. 构建图像
        if self._is_sh:
            # v3: 上交所使用 SHImageBuilder（内部使用 IsAggressive 字段）
            sh_builder = SHImageBuilder(
                trade_price_bins=trade_price_bins,
                trade_qty_bins=trade_qty_bins,
                order_price_bins=order_price_bins,
                order_qty_bins=order_qty_bins,
                buy_parent=buy_parent,
                sell_parent=sell_parent,
                threshold=threshold,
            )
            if use_vectorized:
                image = sh_builder.build_vectorized(df_trade, df_order)
            else:
                image = sh_builder.build(df_trade, df_order)
        else:
            # v3: 深交所先关联撤单价格，然后构建 ActiveSeqs
            df_trade_enriched = enrich_sz_cancel_price(df_trade, df_order)
            
            # v3: 构建主动委托索引（SZImageBuilder 内部会自动构建）
            sz_builder = SZImageBuilder(
                trade_price_bins=trade_price_bins,
                trade_qty_bins=trade_qty_bins,
                order_price_bins=order_price_bins,
                order_qty_bins=order_qty_bins,
                buy_parent=buy_parent,
                sell_parent=sell_parent,
                threshold=threshold,
            )
            if use_vectorized:
                image = sz_builder.build_vectorized(df_trade_enriched, df_order)
            else:
                image = sz_builder.build(df_trade_enriched, df_order)
        
        # 保存原始图像到实例
        self.image = image.astype(np.float32)
        
        # v3: 验证通道约束（Ch7=Ch9+Ch11, Ch8=Ch10+Ch12）
        if validate_constraints:
            constraint_result = validate_channel_constraints(self.image)
            if not constraint_result['valid']:
                logger.warning(
                    f"{self.stock_code}: 通道约束违反 - {constraint_result['errors']}"
                )
        
        # 4. 归一化
        normalized = normalize_image(image)
        
        return normalized
    
    @classmethod
    def build_image(
        cls,
        stock_code: str,
        df_trade: DataFrame,
        df_order: DataFrame,
        config: Optional[Config] = None,
        trade_date: str = "",
        use_vectorized: bool = True,
        validate_constraints: bool = True,
    ) -> Optional[np.ndarray]:
        """
        类方法：快速构建单只股票图像
        
        v3 架构变更：
        - 上交所：委托表必须包含 IsAggressive 和 BizIndex 字段
        - 深交所：自动构建 ActiveSeqs
        - 新增通道约束验证
        
        Args:
            stock_code: 股票代码，如 '600519.SH' 或 '000001.SZ'
            df_trade: 成交表（已清洗）
            df_order: 委托表（已清洗）
            config: 配置对象（可选）
            trade_date: 交易日期（可选）
            use_vectorized: 是否使用向量化方法
            validate_constraints: 是否验证 v3 通道约束（默认 True）
        
        Returns:
            归一化后的 [15, 8, 8] 图像
        
        Example:
            >>> image = Level2ImageBuilder.build_image(
            ...     "600519.SH", df_trade, df_order
            ... )
        """
        builder = cls(stock_code, trade_date, config)
        return builder.build_single_stock(
            df_trade, df_order, use_vectorized, validate_constraints
        )


def build_l2_image(
    stock_code: str,
    df_trade: DataFrame,
    df_order: DataFrame,
    config: Optional[Config] = None,
    use_vectorized: bool = True,
    validate_constraints: bool = True,
) -> Optional[np.ndarray]:
    """
    便捷函数：构建 Level2 图像
    
    v3 架构变更：
    - 上交所：委托表必须包含 IsAggressive 和 BizIndex 字段
    - 深交所：自动构建 ActiveSeqs
    - 新增通道约束验证
    
    Args:
        stock_code: 股票代码，如 '600519.SH' 或 '000001.SZ'
        df_trade: 成交表（已清洗）
        df_order: 委托表（已清洗）
        config: 配置对象（可选）
        use_vectorized: 是否使用向量化方法
        validate_constraints: 是否验证 v3 通道约束（默认 True）
    
    Returns:
        归一化后的 [15, 8, 8] 图像
    
    Example:
        >>> from l2_image_builder.builder import build_l2_image
        >>> image = build_l2_image("600519.SH", df_trade, df_order)
        >>> image.shape
        (15, 8, 8)
    """
    return Level2ImageBuilder.build_image(
        stock_code, df_trade, df_order, config, 
        use_vectorized=use_vectorized,
        validate_constraints=validate_constraints,
    )


def build_l2_image_with_stats(
    stock_code: str,
    df_trade: DataFrame,
    df_order: DataFrame,
    config: Optional[Config] = None,
    use_vectorized: bool = True,
    validate_constraints: bool = True,
) -> Tuple[Optional[np.ndarray], dict, np.ndarray]:
    """
    便捷函数：构建 Level2 图像并返回统计信息
    
    v3 架构变更：
    - 上交所：委托表必须包含 IsAggressive 和 BizIndex 字段
    - 深交所：自动构建 ActiveSeqs
    - 新增通道约束验证
    
    Args:
        stock_code: 股票代码
        df_trade: 成交表（已清洗）
        df_order: 委托表（已清洗）
        config: 配置对象（可选）
        use_vectorized: 是否使用向量化方法
        validate_constraints: 是否验证 v3 通道约束（默认 True）
    
    Returns:
        (normalized_image, channel_stats, raw_image)
        - normalized_image: 归一化后的图像
        - channel_stats: 通道统计信息
        - raw_image: 原始计数图像
    
    Example:
        >>> image, stats, raw = build_l2_image_with_stats(
        ...     "600519.SH", df_trade, df_order
        ... )
        >>> print(stats['all_trade']['sum'])
    """
    builder = Level2ImageBuilder(stock_code, "", config)
    normalized = builder.build_single_stock(
        df_trade, df_order, use_vectorized, validate_constraints
    )
    
    if normalized is None:
        return None, {}, np.zeros((15, 8, 8), dtype=np.float32)
    
    stats = builder.get_channel_stats()
    raw_image = builder.get_raw_image()
    
    return normalized, stats, raw_image
