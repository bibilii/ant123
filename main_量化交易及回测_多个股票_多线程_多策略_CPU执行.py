#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import os
import sys
import traceback
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import pandas as pd
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import multiprocessing
from strategy import (
    MovingAverageStrategy, 
    AnnualMAStrategy, 
    ProportionalMAStrategy,
    TurtleTradingStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy
)
from strategy_statistics import StrategyStatistics
from strategy_plotter import StrategyPlotter
from strategy_executor import TradeExecutor
from getdata_daily_from_localdata import StockDataReader
from stock_info_processor import StockInfoProcessor

# 配置日志
def setup_logging():
    """配置日志系统"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'strategy_run_{datetime.now().strftime("%Y%m%d")}.log')
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 配置文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 配置控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logging()

# 配置参数
START_DATE = "2014-01-01"  # 需要更长的历史数据来计算年均线
END_DATE = "2025-05-16"
INITIAL_CASH = 100000.0  # 初始资金10万

# 数据源配置，如果为空则使用StockDataReader中的默认值
DATA_SOURCE_TYPE = ''  # 'efinance' 或 'tushare'
INPUT_FOLDER = ''  # 数据文件夹路径

# 输出配置
OUTPUT_FOLDER = '.strategy_results/多线程-画图-多策略-多线程CPU-AMD5600X'

# 选定的股票代码
SELECTED_STOCK_CODES = [
    '002239',  # 奥特佳
    '002160',  # 常铝股份
    '600053',  # 九鼎投资
    '002135',  # 东南网架
    '000520',  # 凤凰航运

    '000001',  # 平安银行
    '002594',  # 比亚迪
    '002574',  # 明牌珠宝
    '601727',  # 上海电气
    '601877',  # 正泰电气
    '600900',  # 长江电力
    '600481',  # 双良节能
    '300750',  # 宁德时代
    '603843',  # 正平股份
    '600191',  # 华资实业
    '601166',  # 兴业银行
]

# 定义所有策略
STRATEGIES = {
    "均线交叉策略": MovingAverageStrategy,
    "年均线突破策略": AnnualMAStrategy,
    "比例交易策略": ProportionalMAStrategy,
    "海龟交易策略": TurtleTradingStrategy,
    "RSI策略": RSIStrategy,
    "MACD策略": MACDStrategy,
    "布林带策略": BollingerBandsStrategy
}

plotter = StrategyPlotter()

def get_csv_files_by_type(reader: StockDataReader):
    """根据数据类型获取CSV文件列表"""
    try:
        import glob
        data_dir = reader.csv_file_path
        if reader.data_type == 'efinance':
            pattern = os.path.join(data_dir, '[0-9][0-9][0-9][0-9][0-9][0-9].csv')
            return glob.glob(pattern)[:10000]
        elif reader.data_type == 'tushare':
            pattern_sz = os.path.join(data_dir, '*.SZ.csv')
            pattern_sh = os.path.join(data_dir, '*.SH.csv')
            pattern_bj = os.path.join(data_dir, '*.BJ.csv')
            return glob.glob(pattern_sz)[:10] + glob.glob(pattern_sh)[:10]+glob.glob(pattern_bj)[:10]
        else:
            raise ValueError("不支持的数据类型")
    except Exception as e:
        logger.error(f"获取CSV文件列表时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def save_error_statistics(symbol: str, output_folder: str, start_date: str, end_date: str, error_msg: str, lock=None):
    """保存错误统计信息，支持多进程加锁"""
    try:
        stats = StrategyStatistics(output_file=os.path.join(output_folder, 'strategy_statistics.csv'))
        for strategy_name in STRATEGIES.keys():
            statistics = {
                '策略名称': strategy_name,
                '股票代码': symbol,
                '开始日期': start_date,
                '结束日期': end_date,
                '交易天数': 0,
                '初始资产': INITIAL_CASH,
                '最终总资产': '',
                '最终股票市值': '',
                '持有现金': '',
                '持有股票数量': '',
                '信号次数': '',
                '买入次数': '',
                '卖出次数': '',
                '累计佣金': '',
                '累计印花税': '',
                '总收益率': '',
                '股票涨幅': '',
                '备注': f'执行失败: {error_msg}'
            }
            stats.save_statistics(statistics, lock=lock)
    except Exception as e:
        logger.error(f"保存错误统计信息时出错: {str(e)}")
        logger.error(traceback.format_exc())

def process_single_stock(args):
    """
    处理单个股票的策略执行和结果保存

    参数:
        args (tuple): 包含股票代码、输出文件夹、开始日期和结束日期的元组
    """
    symbol, output_folder, start_date, end_date, lock = args
    
    try:
        # 创建数据读取器
        reader = StockDataReader(
            data_type=DATA_SOURCE_TYPE if DATA_SOURCE_TYPE else None,
            parent_dir=INPUT_FOLDER if INPUT_FOLDER else None
        )
        
        # 获取股票名称
        stock_processor = StockInfoProcessor()
        try:
            stock_info = stock_processor.get_stock_info(symbol)
            stock_name = stock_info['name']
            safe_stock_name = stock_processor.sanitize_filename(stock_name)  # 处理文件名
        except:
            stock_name = "未知"
            safe_stock_name = "unknown"
        
        # 获取数据
        df = reader.get_standard_data(symbol, start_date, end_date)
        
        # 添加股票代码和名称列
        df['股票代码'] = symbol
        df['股票名称'] = stock_name
        
        # 重命名列以匹配策略需求
        df = df.rename(columns={
            'date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'amount': '成交额'
        })
        
        # 检查数据是否为空
        if df.empty:
            logger.warning(f"警告: 股票 {symbol} 在指定时间范围内没有数据")
            save_error_statistics(symbol, output_folder, start_date, end_date, "数据为空", lock=lock)
            return False
            
        # 确保日期列是datetime类型并按时间正序排列
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期', ascending=True)
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 检查必要的列是否存在
        required_columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"错误: 股票 {symbol} 数据缺少必要的列: {missing_columns}")
            save_error_statistics(symbol, output_folder, start_date, end_date, f"缺少必要的列: {missing_columns}", lock=lock)
            return False
            
        # 检查数据类型
        numeric_columns = ['开盘', '收盘', '最高', '最低', '成交量', '成交额']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 为每个策略创建子文件夹
        for strategy_name, strategy_class in STRATEGIES.items():
            try:
                strategy_folder = os.path.join(output_folder, strategy_name)
                if not os.path.exists(strategy_folder):
                    os.makedirs(strategy_folder)
                
                # 创建策略实例
                strategy = strategy_class(symbol)
                
                # 生成交易信号
                df_with_signals = strategy.generate_signals(df.copy())
                
                # 执行交易
                executor = TradeExecutor(
                    strategy_name=strategy_name,
                    initial_cash=INITIAL_CASH,
                    buy_ratio=0.2,
                    sell_ratio=0.3
                )
                trade_records = executor.execute_trades(df_with_signals)
                
                # 保存结果
                output_file = os.path.join(strategy_folder, f"{symbol}_{safe_stock_name}_result.csv")
                trade_records.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                # 生成图表
                output_plot = os.path.join(strategy_folder, f"{symbol}_{safe_stock_name}_plot.png")
                plotter.plot_strategy_result(trade_records, symbol, output_plot)
                
                # 计算并保存统计结果
                stats = StrategyStatistics(output_file=os.path.join(output_folder, 'strategy_statistics.csv'))
                statistics = stats.calculate_statistics(
                    df=trade_records,
                    strategy_name=strategy_name,
                    symbol=f"{symbol}_{safe_stock_name}",  # 在统计结果中使用安全的文件名
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=INITIAL_CASH
                )
                stats.save_statistics(statistics, lock=lock)
                
                logger.info(f"成功处理股票 {symbol}_{stock_name} 的 {strategy_name} 策略")
                
            except Exception as e:
                logger.error(f"处理股票 {symbol}_{stock_name} 的 {strategy_name} 策略时出错: {str(e)}")
                logger.error(traceback.format_exc())
                save_error_statistics(f"{symbol}_{safe_stock_name}", output_folder, start_date, end_date, f"{strategy_name} 策略执行失败: {str(e)}", lock=lock)
                continue
        
        return True
    except Exception as e:
        logger.error(f"处理股票 {symbol} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        save_error_statistics(symbol, output_folder, start_date, end_date, str(e), lock=lock)
        return False

def main():
    """主函数"""
    try:
        global SELECTED_STOCK_CODES

        # 创建数据读取器，如果DATA_SOURCE_TYPE为空则使用默认值
        reader = StockDataReader(
            data_type=DATA_SOURCE_TYPE if DATA_SOURCE_TYPE else None,
            parent_dir=INPUT_FOLDER if INPUT_FOLDER else None
        )
        
        # 创建输出文件夹
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            
        # 为每个策略创建子文件夹
        for strategy_name in STRATEGIES.keys():
            strategy_folder = os.path.join(OUTPUT_FOLDER, strategy_name)
            if not os.path.exists(strategy_folder):
                os.makedirs(strategy_folder)

        # 如果SELECTED_STOCK_CODES为空，则读取默认文件夹中所有CSV文件
        if len(SELECTED_STOCK_CODES) == 0:
            csv_files = get_csv_files_by_type(reader)
            if reader.data_type == 'efinance':
                SELECTED_STOCK_CODES = [os.path.basename(f).split('.')[0] for f in csv_files]
            elif reader.data_type == 'tushare':
                SELECTED_STOCK_CODES = [os.path.basename(f).split('.')[0] + '.' + os.path.basename(f).split('.')[1] for f in csv_files]
        
        if not SELECTED_STOCK_CODES:
            logger.error("\n没有找到任何股票数据")
            return
        
        # 准备参数列表
        with Manager() as manager:
            lock = manager.Lock()
            args_list = [(symbol, OUTPUT_FOLDER, START_DATE, END_DATE, lock) for symbol in SELECTED_STOCK_CODES]
            num_processes = min(cpu_count(), len(SELECTED_STOCK_CODES))
            logger.info(f"\n使用 {num_processes} 个进程进行处理")
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_single_stock, args_list),
                    total=len(SELECTED_STOCK_CODES),
                    desc="\n处理股票"
                ))
            success_count = sum(1 for r in results if r)
            logger.info(f"\n处理完成！成功处理 {success_count}/{len(SELECTED_STOCK_CODES)} 只股票")
        
    except Exception as e:
        logger.error(f"主程序执行出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 