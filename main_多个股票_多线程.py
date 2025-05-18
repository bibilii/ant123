#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import pandas as pd
from datetime import datetime
import os
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np
from getdata_daily_from_localdata import StockDataReader
from strategy import (
    MovingAverageStrategy, 
    AnnualMAStrategy, 
    ProportionalMAStrategy,
    TurtleTradingStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy
)
from strategy_executor import TradeExecutor
from strategy_plotter import StrategyPlotter
from strategy_statistics import StrategyStatistics

class GPUAccelerator:
    """GPU加速器类"""
    
    def __init__(self):
        """初始化GPU加速器"""
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            # 设置GPU内存分配器
            torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU内存
        else:
            print("警告: 未检测到GPU，将使用CPU模式")

class BatchStrategyRunner:
    """批量策略运行器类"""
    
    def __init__(self, 
                 data_folder: str,
                 start_date: str,
                 end_date: str,
                 initial_cash: float = 1000000.0,
                 buy_ratio: float = 0.2,
                 sell_ratio: float = 0.3,
                 max_workers: int = 4):
        """
        初始化批量策略运行器
        参数:
            data_folder (str): CSV文件所在文件夹路径
            start_date (str): 开始日期
            end_date (str): 结束日期
            initial_cash (float): 初始资金
            buy_ratio (float): 买入比例
            sell_ratio (float): 卖出比例
            max_workers (int): 最大线程数
        """
        self.data_folder = data_folder
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.max_workers = max_workers
        
        # 创建输出目录
        self.output_dir = '.strategy_results'
        self.batch_output_dir = os.path.join(self.output_dir, 'batch_gpu')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)
            
        # 初始化GPU加速器
        self.gpu_acc = GPUAccelerator()
        
        # 初始化各个组件
        self.data_reader = StockDataReader()
        self.plotter = StrategyPlotter()
        self.statistics = StrategyStatistics(output_file=os.path.join(self.output_dir, 'batch_gpu_strategy_statistics.csv'))
        
        # 创建线程锁
        self.stats_lock = threading.Lock()
        
    def get_stock_codes(self):
        """获取文件夹中所有CSV文件的股票代码"""
        stock_codes = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.csv'):
                stock_code = file[:-4]
                stock_codes.append(stock_code)
        return sorted(stock_codes)
    
    def run_strategy_for_stock(self, symbol: str, strategy_name: str):
        """
        对单个股票运行指定的策略
        参数:
            symbol (str): 股票代码
            strategy_name (str): 策略名称
        """
        try:
            print(f"\n开始运行策略: {strategy_name}")
            print(f"股票代码: {symbol}")
            print(f"回测区间: {self.start_date} 至 {self.end_date}")
            
            # 1. 获取数据
            df = self.data_reader.get_standard_data(symbol, self.start_date, self.end_date)
            if df is None or df.empty:
                print(f"错误: 无法获取股票 {symbol} 的数据")
                return None, None
                
            # 2. 创建策略实例（使用GPU加速器）
            if strategy_name == "均线交叉策略":
                strategy = MovingAverageStrategy(symbol, self.gpu_acc)
            elif strategy_name == "年均线突破策略":
                strategy = AnnualMAStrategy(symbol, self.gpu_acc)
            elif strategy_name == "比例交易策略":
                strategy = ProportionalMAStrategy(symbol, self.gpu_acc)
            elif strategy_name == "海龟交易策略":
                strategy = TurtleTradingStrategy(symbol, self.gpu_acc)
            elif strategy_name == "RSI策略":
                strategy = RSIStrategy(symbol, self.gpu_acc)
            elif strategy_name == "MACD策略":
                strategy = MACDStrategy(symbol, self.gpu_acc)
            elif strategy_name == "布林带策略":
                strategy = BollingerBandsStrategy(symbol, self.gpu_acc)
            else:
                print(f"错误: 未知的策略名称 {strategy_name}")
                return None, None
                
            # 3. 生成交易信号
            df = strategy.generate_signals(df)
            
            # 4. 执行交易
            executor = TradeExecutor(
                strategy_name=strategy_name,
                initial_cash=self.initial_cash,
                buy_ratio=self.buy_ratio,
                sell_ratio=self.sell_ratio
            )
            trade_records = executor.execute_trades(df)
            
            # 5. 保存交易记录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trade_records_file = os.path.join(self.batch_output_dir, f"{symbol}_{strategy_name}_{timestamp}_trades.csv")
            trade_records.to_csv(trade_records_file, index=False, encoding='utf-8-sig')
            
            # 6. 绘制策略结果图表
            plot_file = os.path.join(self.batch_output_dir, f"{symbol}_{strategy_name}_{timestamp}_plot.png")
            self.plotter.plot_strategy_result(trade_records, symbol, plot_file)
            
            # 7. 计算并保存统计结果
            stats = self.statistics.calculate_statistics(
                df=trade_records,
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_cash=self.initial_cash
            )
            
            # 使用线程锁保护统计结果的保存
            with self.stats_lock:
                self.statistics.save_statistics(stats)
            
            print(f"\n策略执行完成:")
            print(f"交易记录已保存到: {trade_records_file}")
            print(f"策略图表已保存到: {plot_file}")
            print(f"统计结果已保存到: {self.statistics.output_file}")
            
            return trade_records, stats
            
        except Exception as e:
            print(f"处理股票 {symbol} 的策略 {strategy_name} 时发生错误: {str(e)}")
            return None, None

def process_stock(runner, symbol, strategies):
    """处理单个股票的所有策略"""
    print(f"\n开始处理股票: {symbol}")
    results = {}
    
    for strategy_name in strategies:
        trade_records, stats = runner.run_strategy_for_stock(symbol, strategy_name)
        if trade_records is not None and stats is not None:
            results[strategy_name] = {
                'trade_records': trade_records,
                'statistics': stats
            }
    
    # 打印该股票的策略比较结果
    print(f"\n{symbol} 策略比较结果:")
    print("-" * 100)
    print(f"{'策略名称':<15} {'总收益率':>10} {'股票涨幅':>10} {'买入次数':>8} {'卖出次数':>8} {'最大回撤':>10}")
    print("-" * 100)
    
    for strategy_name in strategies:
        if strategy_name in results:
            stats = results[strategy_name]['statistics']
            print(f"{strategy_name:<15} {stats['总收益率']:>10.2f}% {stats['股票涨幅']:>10.2f}% "
                  f"{stats['买入次数']:>8} {stats['卖出次数']:>8} {stats.get('最大回撤', 0):>10.2f}%")
    
    return symbol, results

def main():
    """主函数"""
    # 设置回测参数
    data_folder = r"\\DH2600-866D\personal_folder\99-data\downtask\taskweek\01-日线数据-efinance爬虫\2025-05-09"  # CSV文件所在文件夹
    start_date = "2013-01-01"  # 开始日期
    end_date = "2024-01-01"    # 结束日期
    initial_cash = 1000000.0   # 初始资金
    buy_ratio = 0.2           # 买入比例
    sell_ratio = 0.3          # 卖出比例
    max_workers = 4           # 最大线程数
    
    # 创建批量策略运行器
    runner = BatchStrategyRunner(
        data_folder=data_folder,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        max_workers=max_workers
    )
    
    # 获取所有股票代码
    stock_codes = runner.get_stock_codes()
    print(f"找到 {len(stock_codes)} 个股票代码: {', '.join(stock_codes)}")
    
    # 运行所有策略
    strategies = [
        "均线交叉策略",
        "年均线突破策略",
        "比例交易策略",
        "海龟交易策略",
        "RSI策略",
        "MACD策略",
        "布林带策略"
    ]
    
    # 使用线程池处理多个股票
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_stock = {
            executor.submit(process_stock, runner, symbol, strategies): symbol 
            for symbol in stock_codes
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_stock):
            symbol = future_to_stock[future]
            try:
                symbol, results = future.result()
                print(f"\n完成股票 {symbol} 的处理")
            except Exception as e:
                print(f"\n处理股票 {symbol} 时发生错误: {str(e)}")

if __name__ == "__main__":
    main() 