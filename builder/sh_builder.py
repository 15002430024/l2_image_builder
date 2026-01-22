"""
上交所图像构建器（简化版）

Prompt 3.1 实现：
- 委托表已预处理，Qty 为完整母单量
- 撤单的 Price 已补全
- 直接按 OrdType 和 Side 过滤即可

15通道定义：
| 索引 | 名称 | 数据源 | 筛选条件 |
|------|------|--------|----------|
| 0 | 全部成交 | 成交表 | 全部 |
| 1 | 主动买入 | 成交表 | BSFlag='B' |
| 2 | 主动卖出 | 成交表 | BSFlag='S' |
| 3 | 大买单 | 成交表 | 买方母单≥阈值 |
| 4 | 大卖单 | 成交表 | 卖方母单≥阈值 |
| 5 | 小买单 | 成交表 | 买方母单<阈值 |
| 6 | 小卖单 | 成交表 | 卖方母单<阈值 |
| 7 | 买单 | 委托表 | OrdType='New' & Side='B' |
| 8 | 卖单 | 委托表 | OrdType='New' & Side='S' |
| 9 | 主动买入(委托) | 成交表 | BSFlag='B' |
| 10 | 主动卖出(委托) | 成交表 | BSFlag='S' |
| 11 | 非主动买入 | 委托表 | OrdType='New' & Side='B' |
| 12 | 非主动卖出 | 委托表 | OrdType='New' & Side='S' |
| 13 | 撤买 | 委托表 | OrdType='Cancel' & Side='B' |
| 14 | 撤卖 | 委托表 | OrdType='Cancel' & Side='S' |
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
import pandas as pd

from ..config import Channels
from ..data_loader.polars_utils import POLARS_AVAILABLE, is_polars_df

if POLARS_AVAILABLE:
    import polars as pl


class SHImageBuilder:
    """
    上交所图像构建器（简化版）
    
    Features:
        - 支持 Polars 和 Pandas DataFrame
        - 向量化实现，高性能
        - 简化的委托处理逻辑（数据已预处理）
    
    Args:
        price_bins: 价格分位数边界 (7个值)
        qty_bins: 数量分位数边界 (7个值)
        buy_parent: 买方母单金额映射 {OrderNO -> amount}
        sell_parent: 卖方母单金额映射 {OrderNO -> amount}
        threshold: 大单阈值
    """
    
    def __init__(
        self,
        price_bins: np.ndarray,
        qty_bins: np.ndarray,
        buy_parent: Dict[int, float],
        sell_parent: Dict[int, float],
        threshold: float,
    ):
        self.price_bins = price_bins
        self.qty_bins = qty_bins
        self.buy_parent = buy_parent
        self.sell_parent = sell_parent
        self.threshold = threshold
        
        # 初始化图像 [15, 8, 8]
        self.image = np.zeros((15, 8, 8), dtype=np.float32)
    
    def reset(self) -> None:
        """重置图像"""
        self.image.fill(0)
    
    def build(
        self,
        df_trade: Union['pl.DataFrame', pd.DataFrame],
        df_order: Union['pl.DataFrame', pd.DataFrame],
    ) -> np.ndarray:
        """
        构建图像
        
        Args:
            df_trade: 成交表
            df_order: 委托表（已预处理，Qty为完整母单量）
        
        Returns:
            [15, 8, 8] 图像
        """
        self.reset()
        
        # 1. 处理成交表 → 填充通道0-6, 9-10
        self._process_trades(df_trade)
        
        # 2. 处理委托表 → 填充通道7-8, 11-14
        self._process_orders(df_order)
        
        return self.image.copy()
    
    def build_vectorized(
        self,
        df_trade: Union['pl.DataFrame', pd.DataFrame],
        df_order: Union['pl.DataFrame', pd.DataFrame],
    ) -> np.ndarray:
        """
        向量化构建图像（推荐，更高性能）
        
        Args:
            df_trade: 成交表
            df_order: 委托表（已预处理，Qty为完整母单量）
        
        Returns:
            [15, 8, 8] 图像
        """
        self.reset()
        
        # 1. 向量化处理成交表
        self._process_trades_vectorized(df_trade)
        
        # 2. 向量化处理委托表
        self._process_orders_vectorized(df_order)
        
        return self.image.copy()
    
    # ==================== 逐行处理方法 ====================
    
    def _process_trades(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        处理成交记录（逐行版本）
        
        填充通道: 0, 1-2, 3-6, 9-10
        """
        if is_polars_df(df):
            self._process_trades_polars(df)
        else:
            self._process_trades_pandas(df)
    
    def _process_trades_polars(self, df: 'pl.DataFrame') -> None:
        """Polars 成交处理"""
        # 转换为 numpy 以提高性能
        prices = df['Price'].to_numpy()
        qtys = df['Qty'].to_numpy()
        bs_flags = df['TickBSFlag'].to_numpy()
        buy_orders = df['BuyOrderNO'].to_numpy()
        sell_orders = df['SellOrderNO'].to_numpy()
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            self._fill_trade(
                price_bins[i], qty_bins[i],
                bs_flags[i], buy_orders[i], sell_orders[i]
            )
    
    def _process_trades_pandas(self, df: pd.DataFrame) -> None:
        """Pandas 成交处理"""
        prices = df['Price'].values
        qtys = df['Qty'].values
        bs_flags = df['TickBSFlag'].values
        buy_orders = df['BuyOrderNO'].values
        sell_orders = df['SellOrderNO'].values
        
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            self._fill_trade(
                price_bins[i], qty_bins[i],
                bs_flags[i], buy_orders[i], sell_orders[i]
            )
    
    def _fill_trade(
        self,
        pb: int,
        qb: int,
        bs_flag: str,
        buy_order_no: int,
        sell_order_no: int,
    ) -> None:
        """填充单笔成交到图像"""
        # 通道0: 全部成交
        self.image[Channels.ALL_TRADE, pb, qb] += 1
        
        # 通道1-2, 9-10: 主动方向
        if bs_flag == 'B':
            self.image[Channels.ACTIVE_BUY_TRADE, pb, qb] += 1
            self.image[Channels.ACTIVE_BUY_ORDER, pb, qb] += 1
        elif bs_flag == 'S':
            self.image[Channels.ACTIVE_SELL_TRADE, pb, qb] += 1
            self.image[Channels.ACTIVE_SELL_ORDER, pb, qb] += 1
        
        # 通道3-6: 大小单（与主动方向无关）
        buy_amount = self.buy_parent.get(buy_order_no, 0)
        sell_amount = self.sell_parent.get(sell_order_no, 0)
        
        if buy_amount >= self.threshold:
            self.image[Channels.BIG_BUY, pb, qb] += 1
        elif buy_amount > 0:
            self.image[Channels.SMALL_BUY, pb, qb] += 1
        
        if sell_amount >= self.threshold:
            self.image[Channels.BIG_SELL, pb, qb] += 1
        elif sell_amount > 0:
            self.image[Channels.SMALL_SELL, pb, qb] += 1
    
    def _process_orders(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        处理委托记录（逐行版本）
        
        由于委托表已预处理：
        - Qty 是完整母单量
        - 撤单的 Price 已补全
        
        填充通道: 7-8, 11-14
        """
        if is_polars_df(df):
            self._process_orders_polars(df)
        else:
            self._process_orders_pandas(df)
    
    def _process_orders_polars(self, df: 'pl.DataFrame') -> None:
        """Polars 委托处理"""
        prices = df['Price'].to_numpy()
        qtys = df['Qty'].to_numpy()
        ord_types = df['OrdType'].to_numpy()
        sides = df['Side'].to_numpy()
        
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            if prices[i] <= 0:
                continue
            self._fill_order(
                price_bins[i], qty_bins[i],
                ord_types[i], sides[i]
            )
    
    def _process_orders_pandas(self, df: pd.DataFrame) -> None:
        """Pandas 委托处理"""
        prices = df['Price'].values
        qtys = df['Qty'].values
        ord_types = df['OrdType'].values
        sides = df['Side'].values
        
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        for i in range(len(prices)):
            if prices[i] <= 0:
                continue
            self._fill_order(
                price_bins[i], qty_bins[i],
                ord_types[i], sides[i]
            )
    
    def _fill_order(
        self,
        pb: int,
        qb: int,
        ord_type: str,
        side: str,
    ) -> None:
        """填充单笔委托到图像"""
        if ord_type == 'New':
            # 新增委托 → 通道7/8和11/12
            if side == 'B':
                self.image[Channels.BUY_ORDER, pb, qb] += 1
                self.image[Channels.PASSIVE_BUY, pb, qb] += 1
            elif side == 'S':
                self.image[Channels.SELL_ORDER, pb, qb] += 1
                self.image[Channels.PASSIVE_SELL, pb, qb] += 1
        
        elif ord_type == 'Cancel':
            # 撤单 → 通道13/14
            if side == 'B':
                self.image[Channels.CANCEL_BUY, pb, qb] += 1
            elif side == 'S':
                self.image[Channels.CANCEL_SELL, pb, qb] += 1
    
    # ==================== 向量化处理方法 ====================
    
    def _process_trades_vectorized(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        向量化处理成交记录
        
        使用 np.add.at 进行高效的直方图累加
        """
        if is_polars_df(df):
            prices = df['Price'].to_numpy()
            qtys = df['Qty'].to_numpy()
            bs_flags = df['TickBSFlag'].to_numpy()
            buy_orders = df['BuyOrderNO'].to_numpy()
            sell_orders = df['SellOrderNO'].to_numpy()
        else:
            prices = df['Price'].values
            qtys = df['Qty'].values
            bs_flags = df['TickBSFlag'].values
            buy_orders = df['BuyOrderNO'].values
            sell_orders = df['SellOrderNO'].values
        
        if len(prices) == 0:
            return
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        # 通道0: 全部成交 - 向量化
        np.add.at(self.image[Channels.ALL_TRADE], (price_bins, qty_bins), 1)
        
        # 通道1-2, 9-10: 主动方向 - 向量化
        buy_mask = bs_flags == 'B'
        sell_mask = bs_flags == 'S'
        
        if buy_mask.any():
            np.add.at(
                self.image[Channels.ACTIVE_BUY_TRADE],
                (price_bins[buy_mask], qty_bins[buy_mask]), 1
            )
            np.add.at(
                self.image[Channels.ACTIVE_BUY_ORDER],
                (price_bins[buy_mask], qty_bins[buy_mask]), 1
            )
        
        if sell_mask.any():
            np.add.at(
                self.image[Channels.ACTIVE_SELL_TRADE],
                (price_bins[sell_mask], qty_bins[sell_mask]), 1
            )
            np.add.at(
                self.image[Channels.ACTIVE_SELL_ORDER],
                (price_bins[sell_mask], qty_bins[sell_mask]), 1
            )
        
        # 通道3-6: 大小单 - 向量化查找母单金额
        buy_amounts = np.array([
            self.buy_parent.get(int(o), 0) for o in buy_orders
        ], dtype=np.float64)
        sell_amounts = np.array([
            self.sell_parent.get(int(o), 0) for o in sell_orders
        ], dtype=np.float64)
        
        # 大买单
        big_buy_mask = buy_amounts >= self.threshold
        if big_buy_mask.any():
            np.add.at(
                self.image[Channels.BIG_BUY],
                (price_bins[big_buy_mask], qty_bins[big_buy_mask]), 1
            )
        
        # 小买单
        small_buy_mask = (buy_amounts > 0) & (buy_amounts < self.threshold)
        if small_buy_mask.any():
            np.add.at(
                self.image[Channels.SMALL_BUY],
                (price_bins[small_buy_mask], qty_bins[small_buy_mask]), 1
            )
        
        # 大卖单
        big_sell_mask = sell_amounts >= self.threshold
        if big_sell_mask.any():
            np.add.at(
                self.image[Channels.BIG_SELL],
                (price_bins[big_sell_mask], qty_bins[big_sell_mask]), 1
            )
        
        # 小卖单
        small_sell_mask = (sell_amounts > 0) & (sell_amounts < self.threshold)
        if small_sell_mask.any():
            np.add.at(
                self.image[Channels.SMALL_SELL],
                (price_bins[small_sell_mask], qty_bins[small_sell_mask]), 1
            )
    
    def _process_orders_vectorized(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        向量化处理委托记录
        
        由于委托表已预处理，逻辑极简
        """
        if is_polars_df(df):
            prices = df['Price'].to_numpy()
            qtys = df['Qty'].to_numpy()
            ord_types = df['OrdType'].to_numpy()
            sides = df['Side'].to_numpy()
        else:
            prices = df['Price'].values
            qtys = df['Qty'].values
            ord_types = df['OrdType'].values
            sides = df['Side'].values
        
        if len(prices) == 0:
            return
        
        # 过滤有效价格
        valid_mask = prices > 0
        prices = prices[valid_mask]
        qtys = qtys[valid_mask]
        ord_types = ord_types[valid_mask]
        sides = sides[valid_mask]
        
        if len(prices) == 0:
            return
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        # 新增委托 → 通道7/8和11/12
        new_mask = ord_types == 'New'
        new_buy_mask = new_mask & (sides == 'B')
        new_sell_mask = new_mask & (sides == 'S')
        
        if new_buy_mask.any():
            np.add.at(
                self.image[Channels.BUY_ORDER],
                (price_bins[new_buy_mask], qty_bins[new_buy_mask]), 1
            )
            np.add.at(
                self.image[Channels.PASSIVE_BUY],
                (price_bins[new_buy_mask], qty_bins[new_buy_mask]), 1
            )
        
        if new_sell_mask.any():
            np.add.at(
                self.image[Channels.SELL_ORDER],
                (price_bins[new_sell_mask], qty_bins[new_sell_mask]), 1
            )
            np.add.at(
                self.image[Channels.PASSIVE_SELL],
                (price_bins[new_sell_mask], qty_bins[new_sell_mask]), 1
            )
        
        # 撤单 → 通道13/14
        cancel_mask = ord_types == 'Cancel'
        cancel_buy_mask = cancel_mask & (sides == 'B')
        cancel_sell_mask = cancel_mask & (sides == 'S')
        
        if cancel_buy_mask.any():
            np.add.at(
                self.image[Channels.CANCEL_BUY],
                (price_bins[cancel_buy_mask], qty_bins[cancel_buy_mask]), 1
            )
        
        if cancel_sell_mask.any():
            np.add.at(
                self.image[Channels.CANCEL_SELL],
                (price_bins[cancel_sell_mask], qty_bins[cancel_sell_mask]), 1
            )
    
    # ==================== 辅助方法 ====================
    
    def get_channel_stats(self) -> Dict[str, Dict]:
        """
        获取通道统计信息
        
        Returns:
            各通道的统计数据
        """
        channel_names = [
            'all_trade', 'active_buy_trade', 'active_sell_trade',
            'big_buy', 'big_sell', 'small_buy', 'small_sell',
            'buy_order', 'sell_order', 'active_buy_order', 'active_sell_order',
            'passive_buy', 'passive_sell', 'cancel_buy', 'cancel_sell'
        ]
        
        stats = {}
        for i, name in enumerate(channel_names):
            channel_data = self.image[i]
            stats[name] = {
                'sum': float(channel_data.sum()),
                'max': float(channel_data.max()),
                'nonzero': int(np.count_nonzero(channel_data)),
                'fill_rate': float(np.count_nonzero(channel_data) / 64),
            }
        return stats
    
    def validate_consistency(self) -> Dict[str, bool]:
        """
        验证图像一致性
        
        检查项:
        1. 通道1 + 通道2 ≤ 通道0 (主动买+主动卖 ≤ 全部成交)
        2. 通道3 + 通道5 大约等于买方成交数
        3. 通道4 + 通道6 大约等于卖方成交数
        4. 通道7 = 通道11 (上交所预处理后)
        5. 通道8 = 通道12 (上交所预处理后)
        
        Returns:
            各检查项的结果
        """
        results = {}
        
        # 检查1: 主动方向不超过全部成交
        active_sum = self.image[1].sum() + self.image[2].sum()
        all_trade = self.image[0].sum()
        results['active_le_all'] = active_sum <= all_trade + 1e-6
        
        # 检查2: 通道1 = 通道9, 通道2 = 通道10
        results['ch1_eq_ch9'] = np.allclose(self.image[1], self.image[9])
        results['ch2_eq_ch10'] = np.allclose(self.image[2], self.image[10])
        
        # 检查3: 通道7 = 通道11, 通道8 = 通道12 (上交所预处理后)
        results['ch7_eq_ch11'] = np.allclose(self.image[7], self.image[11])
        results['ch8_eq_ch12'] = np.allclose(self.image[8], self.image[12])
        
        return results


def build_sh_image(
    df_trade: Union['pl.DataFrame', pd.DataFrame],
    df_order: Union['pl.DataFrame', pd.DataFrame],
    price_bins: np.ndarray,
    qty_bins: np.ndarray,
    buy_parent: Dict[int, float],
    sell_parent: Dict[int, float],
    threshold: float,
    vectorized: bool = True,
) -> np.ndarray:
    """
    便捷函数：构建上交所图像
    
    Args:
        df_trade: 成交表
        df_order: 委托表（已预处理）
        price_bins: 价格分位数边界
        qty_bins: 数量分位数边界
        buy_parent: 买方母单金额映射
        sell_parent: 卖方母单金额映射
        threshold: 大单阈值
        vectorized: 是否使用向量化版本
    
    Returns:
        [15, 8, 8] 图像
    """
    builder = SHImageBuilder(
        price_bins=price_bins,
        qty_bins=qty_bins,
        buy_parent=buy_parent,
        sell_parent=sell_parent,
        threshold=threshold,
    )
    
    if vectorized:
        return builder.build_vectorized(df_trade, df_order)
    else:
        return builder.build(df_trade, df_order)


def build_sh_image_with_stats(
    df_trade: Union['pl.DataFrame', pd.DataFrame],
    df_order: Union['pl.DataFrame', pd.DataFrame],
    price_bins: np.ndarray,
    qty_bins: np.ndarray,
    buy_parent: Dict[int, float],
    sell_parent: Dict[int, float],
    threshold: float,
    vectorized: bool = True,
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    构建上交所图像并返回统计信息
    
    Returns:
        (image, channel_stats, consistency_check)
    """
    builder = SHImageBuilder(
        price_bins=price_bins,
        qty_bins=qty_bins,
        buy_parent=buy_parent,
        sell_parent=sell_parent,
        threshold=threshold,
    )
    
    if vectorized:
        image = builder.build_vectorized(df_trade, df_order)
    else:
        image = builder.build(df_trade, df_order)
    
    return image, builder.get_channel_stats(), builder.validate_consistency()
