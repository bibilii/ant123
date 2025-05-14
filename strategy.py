#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import pandas as pd
import numpy as np
from typing import Optional
from getdata_daily_from_localdata import StockDataReader


class BaseTradingStrategy:
    """基础交易策略类"""

    def __init__(self, symbol: str):
        """
        初始化基础交易策略
        参数:
            symbol (str): 股票代码
        """
        self.symbol = symbol
        self.trades = []


class MovingAverageStrategy(BaseTradingStrategy):
    """均线交叉策略"""
    
    def __init__(self, symbol: str, ma_short: int = 5, ma_long: int = 21):
        super().__init__(symbol)
        self.ma_short = ma_short
        self.ma_long = ma_long
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成均线交叉信号"""
        df['MA_short'] = df['收盘'].rolling(window=self.ma_short).mean()
        df['MA_long'] = df['收盘'].rolling(window=self.ma_long).mean()
        df['信号'] = 0
        df.loc[df['MA_short'] > df['MA_long'], '信号'] = 1
        df.loc[df['MA_short'] < df['MA_long'], '信号'] = -1
        return df


class AnnualMAStrategy(BaseTradingStrategy):
    """年均线突破策略"""
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成年均线突破信号"""
        df['MA_250'] = df['收盘'].rolling(window=250).mean()
        df['信号'] = 0
        df.loc[df['最高'] > df['MA_250'], '信号'] = 1
        df.loc[df['最低'] < df['MA_250'], '信号'] = -1
        return df


class ProportionalMAStrategy(BaseTradingStrategy):
    """比例交易策略"""
    
    def __init__(self, symbol: str, buy_ratio: float = 0.3, sell_ratio: float = 0.2):
        super().__init__(symbol)
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成比例交易信号"""
        df['MA_250'] = df['收盘'].rolling(window=250).mean()
        df['信号'] = 0
        df.loc[df['最高'] > df['MA_250'], '信号'] = 1
        df.loc[df['最低'] < df['MA_250'], '信号'] = -1
        return df


class TurtleTradingStrategy(BaseTradingStrategy):
    """海龟交易策略"""
    
    def __init__(self, symbol: str, entry_window: int = 20, exit_window: int = 10):
        super().__init__(symbol)
        self.entry_window = entry_window
        self.exit_window = exit_window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成海龟交易信号"""
        # 计算唐奇安通道
        df['最高价'] = df['最高'].rolling(window=self.entry_window).max()
        df['最低价'] = df['最低'].rolling(window=self.entry_window).min()
        df['退出最高价'] = df['最高'].rolling(window=self.exit_window).max()
        df['退出最低价'] = df['最低'].rolling(window=self.exit_window).min()
        
        # 生成信号
        df['信号'] = 0
        # 突破上轨买入
        df.loc[df['收盘'] > df['最高价'].shift(1), '信号'] = 1
        # 突破下轨卖出
        df.loc[df['收盘'] < df['最低价'].shift(1), '信号'] = -1
        # 退出信号
        df.loc[df['收盘'] < df['退出最低价'].shift(1), '信号'] = -1
        df.loc[df['收盘'] > df['退出最高价'].shift(1), '信号'] = 1
        
        return df


class RSIStrategy(BaseTradingStrategy):
    """RSI策略"""
    
    def __init__(self, symbol: str, period: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__(symbol)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成RSI交易信号"""
        # 计算价格变化
        delta = df['收盘'].diff()
        
        # 计算上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        # 计算RS和RSI
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 生成信号
        df['信号'] = 0
        df.loc[df['RSI'] < self.oversold, '信号'] = 1  # 超卖买入
        df.loc[df['RSI'] > self.overbought, '信号'] = -1  # 超买卖出
        
        return df


class MACDStrategy(BaseTradingStrategy):
    """MACD策略"""
    
    def __init__(self, symbol: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(symbol)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成MACD交易信号"""
        # 计算EMA
        df['EMA_fast'] = df['收盘'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_slow'] = df['收盘'].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算MACD线和信号线
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal'] = df['MACD'].ewm(span=self.signal_period, adjust=False).mean()
        
        # 生成信号
        df['信号'] = 0
        df.loc[df['MACD'] > df['Signal'], '信号'] = 1  # MACD线上穿信号线买入
        df.loc[df['MACD'] < df['Signal'], '信号'] = -1  # MACD线下穿信号线卖出
        
        return df


class BollingerBandsStrategy(BaseTradingStrategy):
    """布林带策略"""
    
    def __init__(self, symbol: str, period: int = 20, std_dev: float = 2.0):
        super().__init__(symbol)
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成布林带交易信号"""
        # 计算布林带
        df['MA'] = df['收盘'].rolling(window=self.period).mean()
        df['STD'] = df['收盘'].rolling(window=self.period).std()
        df['Upper'] = df['MA'] + (df['STD'] * self.std_dev)
        df['Lower'] = df['MA'] - (df['STD'] * self.std_dev)
        
        # 生成信号
        df['信号'] = 0
        df.loc[df['收盘'] < df['Lower'], '信号'] = 1  # 价格触及下轨买入
        df.loc[df['收盘'] > df['Upper'], '信号'] = -1  # 价格触及上轨卖出
        
        return df





