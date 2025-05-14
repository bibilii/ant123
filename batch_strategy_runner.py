#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import pandas as pd
import os
from datetime import datetime
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

class BatchStrategyRunner:
    """批量策略运行器类"""
    
    def __init__(self, 
                 data_dir: str = 'data',
                 start_date: str = "2013-01-01",
                 end_date: str = "2024-01-01",
                 initial_cash: float = 1000000.0,
                 buy_ratio: float = 0.2,
                 sell_ratio: float = 0.3):
        """
        初始化批量策略运行器
        参数:
            data_dir (str): 数据目录
            start_date (str): 开始日期
            end_date (str): 结束日期
            initial_cash (float): 初始资金
            buy_ratio (float): 买入比例
            sell_ratio (float): 卖出比例
        """
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        
        # 创建输出目录
        self.output_dir = 'strategy_results'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 初始化各个组件
        self.data_reader = StockDataReader()
        self.plotter = StrategyPlotter()
        self.statistics = StrategyStatistics()
        
        # 定义所有策略
        self.strategies = {
            "均线交叉策略": MovingAverageStrategy,
            "年均线突破策略": AnnualMAStrategy,
            "比例交易策略": ProportionalMAStrategy,
            "海龟交易策略": TurtleTradingStrategy,
            "RSI策略": RSIStrategy,
            "MACD策略": MACDStrategy,
            "布林带策略": BollingerBandsStrategy
        }
    
    def get_stock_list(self):
        """获取数据目录下的所有股票代码"""
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return [f.split('.')[0] for f in stock_files]
    
    def run_single_stock(self, symbol: str):
        """运行单个股票的所有策略"""
        print(f"\n开始处理股票: {symbol}")
        print(f"回测区间: {self.start_date} 至 {self.end_date}")
        
        # 获取股票数据
        df = self.data_reader.get_standard_data(symbol, self.start_date, self.end_date)
        if df is None or df.empty:
            print(f"错误: 无法获取股票 {symbol} 的数据")
            return None
        
        results = {}
        for strategy_name, strategy_class in self.strategies.items():
            print(f"\n运行策略: {strategy_name}")
            
            # 创建策略实例
            strategy = strategy_class(symbol)
            
            # 生成交易信号
            df_with_signals = strategy.generate_signals(df.copy())
            
            # 执行交易
            executor = TradeExecutor(
                strategy_name=strategy_name,
                initial_cash=self.initial_cash,
                buy_ratio=self.buy_ratio,
                sell_ratio=self.sell_ratio
            )
            trade_records = executor.execute_trades(df_with_signals)
            
            # 保存交易记录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trade_records_file = f"{self.output_dir}/{symbol}_{strategy_name}_{timestamp}_trades.csv"
            trade_records.to_csv(trade_records_file, index=False, encoding='utf-8-sig')
            
            # 绘制策略结果图表
            plot_file = f"{self.output_dir}/{symbol}_{strategy_name}_{timestamp}_plot.png"
            self.plotter.plot_strategy_result(trade_records, symbol, plot_file)
            
            # 计算统计结果
            stats = self.statistics.calculate_statistics(
                df=trade_records,
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_cash=self.initial_cash
            )
            self.statistics.save_statistics(stats)
            
            results[strategy_name] = {
                'trade_records': trade_records,
                'statistics': stats
            }
            
            print(f"策略执行完成:")
            print(f"交易记录已保存到: {trade_records_file}")
            print(f"策略图表已保存到: {plot_file}")
        
        return results
    
    def run_all_stocks(self):
        """运行所有股票的策略"""
        stock_list = self.get_stock_list()
        all_results = {}
        
        for symbol in stock_list:
            results = self.run_single_stock(symbol)
            if results:
                all_results[symbol] = results
        
        # 打印汇总结果
        self.print_summary(all_results)
    
    def print_summary(self, all_results):
        """打印汇总结果"""
        print("\n策略回测汇总结果:")
        print("=" * 120)
        print(f"{'股票代码':<10} {'策略名称':<15} {'总收益率':>10} {'股票涨幅':>10} "
              f"{'买入次数':>8} {'卖出次数':>8} {'最大回撤':>10}")
        print("-" * 120)
        
        for symbol, results in all_results.items():
            for strategy_name, result in results.items():
                stats = result['statistics']
                print(f"{symbol:<10} {strategy_name:<15} {stats['总收益率']:>10.2f}% "
                      f"{stats['股票涨幅']:>10.2f}% {stats['买入次数']:>8} "
                      f"{stats['卖出次数']:>8} {stats.get('最大回撤', 0):>10.2f}%")
        
        print("=" * 120)

def main():
    """主函数"""
    # 设置回测参数
    data_dir = r'\\DX4600-A923\personal_folder\99-data\downtask\taskweek\01-日线数据-efinance爬虫'  # 数据目录
    start_date = "2013-01-01"  # 开始日期
    end_date = "2024-01-01"    # 结束日期
    initial_cash = 1000000.0   # 初始资金
    buy_ratio = 0.2           # 买入比例
    sell_ratio = 0.3          # 卖出比例
    
    # 创建批量策略运行器
    runner = BatchStrategyRunner(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio
    )
    
    # 运行所有股票的策略
    runner.run_all_stocks()

if __name__ == "__main__":
    main() 