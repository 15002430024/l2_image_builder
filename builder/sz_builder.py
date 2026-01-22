"""
深交所图像构建器

Prompt 3.2 实现：
- 主动方向判定：BidApplSeqNum > OfferApplSeqNum → 主动买入
- 通道9/10从成交表填充（"指鹿为马"，与上交所对齐）
- 通道11/12需要判断"纯挂单"（委托未作为主动方成交）
- 撤单在成交表中，ExecType='52'

15通道定义（深交所）：
| 索引 | 名称 | 数据源 | 说明 |
|------|------|--------|------|
| 0 | 全部成交 | 成交表 | ExecType='70' |
| 1 | 主动买入 | 成交表 | BidSeq > OfferSeq |
| 2 | 主动卖出 | 成交表 | OfferSeq > BidSeq |
| 3 | 大买单 | 成交表 | 买方母单≥阈值 |
| 4 | 大卖单 | 成交表 | 卖方母单≥阈值 |
| 5 | 小买单 | 成交表 | 买方母单<阈值 |
| 6 | 小卖单 | 成交表 | 卖方母单<阈值 |
| 7 | 买单 | 委托表 | Side='49' |
| 8 | 卖单 | 委托表 | Side='50' |
| 9 | 主动买入(委托) | 成交表 | BidSeq > OfferSeq |
| 10 | 主动卖出(委托) | 成交表 | OfferSeq > BidSeq |
| 11 | 非主动买入 | 委托表 | Side='49' & 未作为主动方 |
| 12 | 非主动卖出 | 委托表 | Side='50' & 未作为主动方 |
| 13 | 撤买 | 成交表 | ExecType='52' & BidSeq>0 |
| 14 | 撤卖 | 成交表 | ExecType='52' & OfferSeq>0 |
"""

import numpy as np
from typing import Dict, Optional, Set, Tuple, Union
import pandas as pd

from ..config import Channels
from ..data_loader.polars_utils import POLARS_AVAILABLE, is_polars_df

if POLARS_AVAILABLE:
    import polars as pl


class SZImageBuilder:
    """
    深交所图像构建器
    
    Features:
        - 支持 Polars 和 Pandas DataFrame
        - 向量化实现，高性能
        - 主动方序列号集合构建（用于通道11/12判断）
    
    Args:
        price_bins: 价格分位数边界 (7个值)
        qty_bins: 数量分位数边界 (7个值)
        buy_parent: 买方母单金额映射 {BidApplSeqNum -> amount}
        sell_parent: 卖方母单金额映射 {OfferApplSeqNum -> amount}
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
            df_trade: 成交表（包含成交和撤单）
            df_order: 委托表
        
        Returns:
            [15, 8, 8] 图像
        """
        self.reset()
        
        # 1. 构建主动方序列号集合（用于判断通道11/12）
        active_seqs = self._build_active_seqs(df_trade)
        
        # 2. 处理成交记录 → 填充通道0-6, 9-10
        self._process_trades(df_trade)
        
        # 3. 处理撤单记录 → 填充通道13-14
        self._process_cancels(df_trade)
        
        # 4. 处理委托记录 → 填充通道7-8, 11-12
        self._process_orders(df_order, active_seqs)
        
        return self.image.copy()
    
    def build_vectorized(
        self,
        df_trade: Union['pl.DataFrame', pd.DataFrame],
        df_order: Union['pl.DataFrame', pd.DataFrame],
    ) -> np.ndarray:
        """
        向量化构建图像（推荐，更高性能）
        
        Args:
            df_trade: 成交表（包含成交和撤单）
            df_order: 委托表
        
        Returns:
            [15, 8, 8] 图像
        """
        self.reset()
        
        # 1. 构建主动方序列号集合
        active_seqs = self._build_active_seqs_vectorized(df_trade)
        
        # 2. 向量化处理成交记录
        self._process_trades_vectorized(df_trade)
        
        # 3. 向量化处理撤单记录
        self._process_cancels_vectorized(df_trade)
        
        # 4. 向量化处理委托记录
        self._process_orders_vectorized(df_order, active_seqs)
        
        return self.image.copy()
    
    # ==================== 主动方序列号集合构建 ====================
    
    def _build_active_seqs(
        self, df: Union['pl.DataFrame', pd.DataFrame]
    ) -> Dict[str, Set[int]]:
        """
        构建主动方序列号集合（逐行版本）
        
        用于判断通道11/12：委托是否作为主动方成交
        """
        active_seqs = {'buy': set(), 'sell': set()}
        
        if is_polars_df(df):
            df_exec = df.filter(pl.col('ExecType') == '70')
            iterator = df_exec.iter_rows(named=True)
        else:
            df_exec = df[df['ExecType'] == '70']
            iterator = (row for _, row in df_exec.iterrows())
        
        for row in iterator:
            bid_seq = row['BidApplSeqNum']
            offer_seq = row['OfferApplSeqNum']
            
            if bid_seq > offer_seq:
                active_seqs['buy'].add(int(bid_seq))
            elif offer_seq > bid_seq:
                active_seqs['sell'].add(int(offer_seq))
        
        return active_seqs
    
    def _build_active_seqs_vectorized(
        self, df: Union['pl.DataFrame', pd.DataFrame]
    ) -> Dict[str, Set[int]]:
        """
        构建主动方序列号集合（向量化版本）
        """
        if is_polars_df(df):
            df_exec = df.filter(pl.col('ExecType') == '70')
            bid_seqs = df_exec['BidApplSeqNum'].to_numpy()
            offer_seqs = df_exec['OfferApplSeqNum'].to_numpy()
        else:
            df_exec = df[df['ExecType'] == '70']
            bid_seqs = df_exec['BidApplSeqNum'].values
            offer_seqs = df_exec['OfferApplSeqNum'].values
        
        if len(bid_seqs) == 0:
            return {'buy': set(), 'sell': set()}
        
        # 向量化判断主动方
        buy_mask = bid_seqs > offer_seqs
        sell_mask = offer_seqs > bid_seqs
        
        active_buy = set(bid_seqs[buy_mask].astype(int))
        active_sell = set(offer_seqs[sell_mask].astype(int))
        
        return {'buy': active_buy, 'sell': active_sell}
    
    # ==================== 逐行处理方法 ====================
    
    def _process_trades(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        处理成交记录（逐行版本）
        
        填充通道: 0, 1-2, 3-6, 9-10
        """
        if is_polars_df(df):
            df_exec = df.filter(pl.col('ExecType') == '70')
            iterator = df_exec.iter_rows(named=True)
        else:
            df_exec = df[df['ExecType'] == '70']
            iterator = (row for _, row in df_exec.iterrows())
        
        for row in iterator:
            price = row['LastPx']
            qty = row['LastQty']
            
            if price <= 0:
                continue
            
            pb = np.clip(np.digitize(price, self.price_bins), 0, 7)
            qb = np.clip(np.digitize(qty, self.qty_bins), 0, 7)
            
            bid_seq = row['BidApplSeqNum']
            offer_seq = row['OfferApplSeqNum']
            
            # 通道0: 全部成交
            self.image[Channels.ALL_TRADE, pb, qb] += 1
            
            # 通道1-2, 9-10: 主动方向
            if bid_seq > offer_seq:
                self.image[Channels.ACTIVE_BUY_TRADE, pb, qb] += 1
                self.image[Channels.ACTIVE_BUY_ORDER, pb, qb] += 1
            elif offer_seq > bid_seq:
                self.image[Channels.ACTIVE_SELL_TRADE, pb, qb] += 1
                self.image[Channels.ACTIVE_SELL_ORDER, pb, qb] += 1
            
            # 通道3-6: 大小单
            buy_amount = self.buy_parent.get(int(bid_seq), 0)
            sell_amount = self.sell_parent.get(int(offer_seq), 0)
            
            if buy_amount >= self.threshold:
                self.image[Channels.BIG_BUY, pb, qb] += 1
            elif buy_amount > 0:
                self.image[Channels.SMALL_BUY, pb, qb] += 1
            
            if sell_amount >= self.threshold:
                self.image[Channels.BIG_SELL, pb, qb] += 1
            elif sell_amount > 0:
                self.image[Channels.SMALL_SELL, pb, qb] += 1
    
    def _process_cancels(self, df: Union['pl.DataFrame', pd.DataFrame]) -> None:
        """
        处理撤单记录（逐行版本）
        
        填充通道: 13-14
        
        注意：撤单的 LastPx 原始值为 0，需要预先关联委托表补全
        """
        if is_polars_df(df):
            df_cancel = df.filter(pl.col('ExecType') == '52')
            iterator = df_cancel.iter_rows(named=True)
        else:
            df_cancel = df[df['ExecType'] == '52']
            iterator = (row for _, row in df_cancel.iterrows())
        
        for row in iterator:
            price = row['LastPx']
            qty = row['LastQty']
            
            if price <= 0:
                continue
            
            pb = np.clip(np.digitize(price, self.price_bins), 0, 7)
            qb = np.clip(np.digitize(qty, self.qty_bins), 0, 7)
            
            bid_seq = row['BidApplSeqNum']
            offer_seq = row['OfferApplSeqNum']
            
            # 通道13-14: 撤单
            if bid_seq > 0 and offer_seq == 0:
                self.image[Channels.CANCEL_BUY, pb, qb] += 1
            elif offer_seq > 0 and bid_seq == 0:
                self.image[Channels.CANCEL_SELL, pb, qb] += 1
    
    def _process_orders(
        self,
        df: Union['pl.DataFrame', pd.DataFrame],
        active_seqs: Dict[str, Set[int]],
    ) -> None:
        """
        处理委托记录（逐行版本）
        
        填充通道: 7-8, 11-12
        """
        if is_polars_df(df):
            iterator = df.iter_rows(named=True)
        else:
            iterator = (row for _, row in df.iterrows())
        
        for row in iterator:
            price = row['Price']
            qty = row['OrderQty']
            
            if price <= 0:
                continue
            
            pb = np.clip(np.digitize(price, self.price_bins), 0, 7)
            qb = np.clip(np.digitize(qty, self.qty_bins), 0, 7)
            
            appl_seq = int(row['ApplSeqNum'])
            side = row['Side']
            
            if side == '49':  # 买入
                self.image[Channels.BUY_ORDER, pb, qb] += 1
                # 非主动买入：未作为主动方成交
                if appl_seq not in active_seqs['buy']:
                    self.image[Channels.PASSIVE_BUY, pb, qb] += 1
            
            elif side == '50':  # 卖出
                self.image[Channels.SELL_ORDER, pb, qb] += 1
                # 非主动卖出
                if appl_seq not in active_seqs['sell']:
                    self.image[Channels.PASSIVE_SELL, pb, qb] += 1
    
    # ==================== 向量化处理方法 ====================
    
    def _process_trades_vectorized(
        self, df: Union['pl.DataFrame', pd.DataFrame]
    ) -> None:
        """
        向量化处理成交记录
        
        填充通道: 0, 1-2, 3-6, 9-10
        """
        if is_polars_df(df):
            df_exec = df.filter(pl.col('ExecType') == '70')
            prices = df_exec['LastPx'].to_numpy()
            qtys = df_exec['LastQty'].to_numpy()
            bid_seqs = df_exec['BidApplSeqNum'].to_numpy()
            offer_seqs = df_exec['OfferApplSeqNum'].to_numpy()
        else:
            df_exec = df[df['ExecType'] == '70']
            prices = df_exec['LastPx'].values
            qtys = df_exec['LastQty'].values
            bid_seqs = df_exec['BidApplSeqNum'].values
            offer_seqs = df_exec['OfferApplSeqNum'].values
        
        if len(prices) == 0:
            return
        
        # 过滤有效价格
        valid_mask = prices > 0
        prices = prices[valid_mask]
        qtys = qtys[valid_mask]
        bid_seqs = bid_seqs[valid_mask]
        offer_seqs = offer_seqs[valid_mask]
        
        if len(prices) == 0:
            return
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        # 通道0: 全部成交
        np.add.at(self.image[Channels.ALL_TRADE], (price_bins, qty_bins), 1)
        
        # 通道1-2, 9-10: 主动方向
        buy_mask = bid_seqs > offer_seqs
        sell_mask = offer_seqs > bid_seqs
        
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
        
        # 通道3-6: 大小单
        buy_amounts = np.array([
            self.buy_parent.get(int(s), 0) for s in bid_seqs
        ], dtype=np.float64)
        sell_amounts = np.array([
            self.sell_parent.get(int(s), 0) for s in offer_seqs
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
    
    def _process_cancels_vectorized(
        self, df: Union['pl.DataFrame', pd.DataFrame]
    ) -> None:
        """
        向量化处理撤单记录
        
        填充通道: 13-14
        """
        if is_polars_df(df):
            df_cancel = df.filter(pl.col('ExecType') == '52')
            prices = df_cancel['LastPx'].to_numpy()
            qtys = df_cancel['LastQty'].to_numpy()
            bid_seqs = df_cancel['BidApplSeqNum'].to_numpy()
            offer_seqs = df_cancel['OfferApplSeqNum'].to_numpy()
        else:
            df_cancel = df[df['ExecType'] == '52']
            prices = df_cancel['LastPx'].values
            qtys = df_cancel['LastQty'].values
            bid_seqs = df_cancel['BidApplSeqNum'].values
            offer_seqs = df_cancel['OfferApplSeqNum'].values
        
        if len(prices) == 0:
            return
        
        # 过滤有效价格
        valid_mask = prices > 0
        prices = prices[valid_mask]
        qtys = qtys[valid_mask]
        bid_seqs = bid_seqs[valid_mask]
        offer_seqs = offer_seqs[valid_mask]
        
        if len(prices) == 0:
            return
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        # 通道13: 撤买 (BidSeq > 0 且 OfferSeq == 0)
        cancel_buy_mask = (bid_seqs > 0) & (offer_seqs == 0)
        if cancel_buy_mask.any():
            np.add.at(
                self.image[Channels.CANCEL_BUY],
                (price_bins[cancel_buy_mask], qty_bins[cancel_buy_mask]), 1
            )
        
        # 通道14: 撤卖 (OfferSeq > 0 且 BidSeq == 0)
        cancel_sell_mask = (offer_seqs > 0) & (bid_seqs == 0)
        if cancel_sell_mask.any():
            np.add.at(
                self.image[Channels.CANCEL_SELL],
                (price_bins[cancel_sell_mask], qty_bins[cancel_sell_mask]), 1
            )
    
    def _process_orders_vectorized(
        self,
        df: Union['pl.DataFrame', pd.DataFrame],
        active_seqs: Dict[str, Set[int]],
    ) -> None:
        """
        向量化处理委托记录
        
        填充通道: 7-8, 11-12
        """
        if is_polars_df(df):
            prices = df['Price'].to_numpy()
            qtys = df['OrderQty'].to_numpy()
            appl_seqs = df['ApplSeqNum'].to_numpy()
            sides = df['Side'].to_numpy()
        else:
            prices = df['Price'].values
            qtys = df['OrderQty'].values
            appl_seqs = df['ApplSeqNum'].values
            sides = df['Side'].values
        
        if len(prices) == 0:
            return
        
        # 过滤有效价格
        valid_mask = prices > 0
        prices = prices[valid_mask]
        qtys = qtys[valid_mask]
        appl_seqs = appl_seqs[valid_mask]
        sides = sides[valid_mask]
        
        if len(prices) == 0:
            return
        
        # 计算 bin 索引
        price_bins = np.clip(np.digitize(prices, self.price_bins), 0, 7)
        qty_bins = np.clip(np.digitize(qtys, self.qty_bins), 0, 7)
        
        # 买入委托 (Side='49')
        buy_mask = sides == '49'
        if buy_mask.any():
            np.add.at(
                self.image[Channels.BUY_ORDER],
                (price_bins[buy_mask], qty_bins[buy_mask]), 1
            )
            
            # 非主动买入：未作为主动方成交
            buy_seqs = appl_seqs[buy_mask].astype(int)
            passive_buy_mask = np.array([
                seq not in active_seqs['buy'] for seq in buy_seqs
            ])
            if passive_buy_mask.any():
                np.add.at(
                    self.image[Channels.PASSIVE_BUY],
                    (price_bins[buy_mask][passive_buy_mask],
                     qty_bins[buy_mask][passive_buy_mask]), 1
                )
        
        # 卖出委托 (Side='50')
        sell_mask = sides == '50'
        if sell_mask.any():
            np.add.at(
                self.image[Channels.SELL_ORDER],
                (price_bins[sell_mask], qty_bins[sell_mask]), 1
            )
            
            # 非主动卖出
            sell_seqs = appl_seqs[sell_mask].astype(int)
            passive_sell_mask = np.array([
                seq not in active_seqs['sell'] for seq in sell_seqs
            ])
            if passive_sell_mask.any():
                np.add.at(
                    self.image[Channels.PASSIVE_SELL],
                    (price_bins[sell_mask][passive_sell_mask],
                     qty_bins[sell_mask][passive_sell_mask]), 1
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
        2. 通道1 = 通道9, 通道2 = 通道10 (深交所都从成交表填充)
        3. 通道7 ≥ 通道11, 通道8 ≥ 通道12 (全部委托 ≥ 纯挂单)
        
        Returns:
            各检查项的结果
        """
        results = {}
        
        # 检查1: 主动方向不超过全部成交
        active_sum = self.image[1].sum() + self.image[2].sum()
        all_trade = self.image[0].sum()
        results['active_le_all'] = active_sum <= all_trade + 1e-6
        
        # 检查2: 通道1 = 通道9, 通道2 = 通道10 (深交所)
        results['ch1_eq_ch9'] = np.allclose(self.image[1], self.image[9])
        results['ch2_eq_ch10'] = np.allclose(self.image[2], self.image[10])
        
        # 检查3: 全部委托 ≥ 纯挂单
        results['ch7_ge_ch11'] = self.image[7].sum() >= self.image[11].sum() - 1e-6
        results['ch8_ge_ch12'] = self.image[8].sum() >= self.image[12].sum() - 1e-6
        
        return results


def build_sz_image(
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
    便捷函数：构建深交所图像
    
    Args:
        df_trade: 成交表（包含成交和撤单）
        df_order: 委托表
        price_bins: 价格分位数边界
        qty_bins: 数量分位数边界
        buy_parent: 买方母单金额映射
        sell_parent: 卖方母单金额映射
        threshold: 大单阈值
        vectorized: 是否使用向量化版本
    
    Returns:
        [15, 8, 8] 图像
    """
    builder = SZImageBuilder(
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


def build_sz_image_with_stats(
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
    构建深交所图像并返回统计信息
    
    Returns:
        (image, channel_stats, consistency_check)
    """
    builder = SZImageBuilder(
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


def build_active_seqs_from_trade(
    df_trade: Union['pl.DataFrame', pd.DataFrame],
) -> Dict[str, Set[int]]:
    """
    从成交表构建主动方序列号集合
    
    用于判断委托是否作为主动方成交
    
    Args:
        df_trade: 成交表
    
    Returns:
        {'buy': set of BidApplSeqNum, 'sell': set of OfferApplSeqNum}
    """
    if is_polars_df(df_trade):
        df_exec = df_trade.filter(pl.col('ExecType') == '70')
        bid_seqs = df_exec['BidApplSeqNum'].to_numpy()
        offer_seqs = df_exec['OfferApplSeqNum'].to_numpy()
    else:
        df_exec = df_trade[df_trade['ExecType'] == '70']
        bid_seqs = df_exec['BidApplSeqNum'].values
        offer_seqs = df_exec['OfferApplSeqNum'].values
    
    if len(bid_seqs) == 0:
        return {'buy': set(), 'sell': set()}
    
    buy_mask = bid_seqs > offer_seqs
    sell_mask = offer_seqs > bid_seqs
    
    return {
        'buy': set(bid_seqs[buy_mask].astype(int)),
        'sell': set(offer_seqs[sell_mask].astype(int)),
    }
