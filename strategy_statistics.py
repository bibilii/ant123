# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55
import pandas as pd
import os
from datetime import datetime
import pandas.errors
from typing import Optional
import threading

class StrategyStatistics:
    """策略统计类"""
    
    def __init__(self, output_file='strategy_statistics.csv'):
        """
        初始化策略统计类
        
        参数:
            output_file (str): 统计结果保存的文件路径
        """
        self.output_file = output_file
        self._init_statistics_file()
    
    def _init_statistics_file(self):
        """初始化统计文件，如果文件不存在则创建"""
        if not os.path.exists(self.output_file):
            columns = [
                '策略名称', '股票代码', '开始日期', '结束日期', '交易天数',
                '初始资产', '最终总资产', '最终股票市值', '持有现金',
                '持有股票数量', '信号次数', '买入次数', '卖出次数',
                '累计佣金', '累计印花税', '总收益率', '股票涨幅', '备注'
            ]
            pd.DataFrame(columns=columns).to_csv(self.output_file, index=False, encoding='utf-8-sig')
    
    def calculate_statistics(self, df, strategy_name, symbol, start_date, end_date, initial_cash):
        """
        计算策略统计指标
        
        参数:
            df (pd.DataFrame): 策略执行结果数据
            strategy_name (str): 策略名称
            symbol (str): 股票代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            initial_cash (float): 初始资金
            
        返回:
            dict: 统计结果字典
        """
        # 计算基本指标
        final_total = df['总金额'].iloc[-1]
        final_stock_value = df['股票市值'].iloc[-1]
        final_cash = df['余额'].iloc[-1]
        final_stock_quantity = df['持股数量'].iloc[-1]
        
        # 计算交易次数
        buy_count = len(df[df['业务名称'] == '买入'])
        sell_count = len(df[df['业务名称'] == '卖出'])
        signal_count = len(df[df['信号'] != 0])
        
        # 计算累计费用
        total_commission = df['佣金'].sum()
        total_stamp_duty = df['印花税'].sum()
        
        # 计算收益率
        total_return_rate = (final_total - initial_cash) / initial_cash * 100
        
        # 计算股票涨幅
        stock_return_rate = (df['收盘'].iloc[-1] - df['收盘'].iloc[0]) / df['收盘'].iloc[0] * 100
        
        # 计算交易天数
        trading_days = len(df)

        # 构建统计结果
        stats = {
            '策略名称': strategy_name,
            '股票代码': str(symbol),  # 确保股票代码为字符串
            '开始日期': start_date,
            '结束日期': end_date,
            '交易天数': trading_days,
            '初始资产': round(initial_cash, 2),
            '最终总资产': round(final_total, 2),
            '最终股票市值': round(final_stock_value, 2),
            '持有现金': round(final_cash, 2),
            '持有股票数量': final_stock_quantity,
            '信号次数': signal_count,
            '买入次数': buy_count,
            '卖出次数': sell_count,
            '累计佣金': round(total_commission, 2),
            '累计印花税': round(total_stamp_duty, 2),
            '总收益率': round(total_return_rate, 2),
            '股票涨幅': round(stock_return_rate, 2),
            '备注': ''
        }
        
        return stats
    
    def save_statistics(self, stats: dict, lock: Optional[threading.Lock] = None):
        """
        保存统计结果，支持多进程/多线程加锁，字段顺序对齐，防止错位。
        参数:
            stats (dict): 统计结果字典
            lock (Lock): 多进程/多线程锁
        """
        columns = [
            '策略名称', '股票代码', '开始日期', '结束日期', '交易天数',
            '初始资产', '最终总资产', '最终股票市值', '持有现金',
            '持有股票数量', '信号次数', '买入次数', '卖出次数',
            '累计佣金', '累计印花税', '总收益率', '股票涨幅', '备注'
        ]
        if lock is not None:
            lock.acquire()
        try:
            # 读取现有数据，遇到EmptyDataError则视为无数据
            if os.path.exists(self.output_file):
                try:
                    existing_df = pd.read_csv(self.output_file, encoding='utf-8-sig')
                except pd.errors.EmptyDataError:
                    existing_df = pd.DataFrame(columns=columns)
            else:
                existing_df = pd.DataFrame(columns=columns)

            # 新数据对齐字段顺序
            new_row = {col: stats.get(col, '') for col in columns}
            new_df = pd.DataFrame([new_row], columns=columns)

            # 设置数据类型
            money_columns = ['初始资产', '最终总资产', '最终股票市值', '持有现金', '累计佣金', '累计印花税']
            percent_columns = ['总收益率', '股票涨幅']
            for col in money_columns:
                if col in new_df:
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce').round(2)
            for col in percent_columns:
                if col in new_df:
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce').round(2)
            if '股票代码' in new_df:
                new_df['股票代码'] = new_df['股票代码'].astype(str)

            # 合并数据
            result_df = pd.concat([existing_df, new_df], ignore_index=True)

            # 股票代码：数字补零，英文不变
            def format_stock_code(code):
                code_str = str(code)
                return code_str.zfill(6) if code_str.isdigit() else code_str
            result_df['股票代码'] = result_df['股票代码'].apply(format_stock_code)

            # 保存结果，始终用columns顺序
            result_df.to_csv(self.output_file, index=False, encoding='utf-8-sig', columns=columns)
        finally:
            if lock is not None:
                lock.release()