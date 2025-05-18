# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55
import pandas as pd
from get_latest_date_subfolder import get_latest_date_subfolder
from typing import Literal
import os
import glob


class StockDataReader:
    """
    通用股票数据读取类，支持多种数据格式和数据源
    """
    DEFAULT_PATHS = {
        'efinance': r"\\DX4600-A923\personal_folder\99-data\downtask\taskweek\01-日线数据-efinance爬虫",
        # 'efinance': r"\\DH2600-866D\personal_folder\99-data\downtask\taskweek\01-日线数据-efinance爬虫",    # 回家的地址
        'tushare': r"D:\trading\dog\getdata_tushare_output\0102_股票数据_行情数据_历史日线\qfq",
        'akshare': r"..\data\akshare"
    }

    def __init__(self, data_type: Literal['efinance', 'tushare', 'akshare'] = 'efinance', parent_dir: str = None):
        if not data_type:
            data_type = 'efinance'
        self.data_type = data_type
        self.set_data_path(parent_dir)

    def set_data_path(self, parent_dir: str = None):
        if parent_dir is None:
            parent_dir = self.DEFAULT_PATHS[self.data_type]
        self.parent_dir = parent_dir
        self.latest = get_latest_date_subfolder(parent_dir)
        self.csv_file_path = os.path.join(self.parent_dir, self.latest)


    def get_raw_data(self, stock_code: str) -> pd.DataFrame:
        """
        获取原始数据（不同数据类型下的原始csv数据）
        """
        if self.data_type == 'tushare':
            # tushare: 000001.SZ.csv 或 600000.SH.csv
            if stock_code.endswith('.SZ') or stock_code.endswith('.SH') or stock_code.endswith('BJ'):
                file_name = f"{stock_code}.csv"

            else:
                # print('stock_code')
                file_name = f"{stock_code}.SZ.csv"
        elif self.data_type == 'akshare':
            file_name = f"{stock_code}.csv"
        else:
            # efinance: 000001.csv
            file_name = f"{stock_code}.csv"
        file_path = os.path.join(self.csv_file_path, file_name)
        # print(file_path)
        return pd.read_csv(file_path)

    def get_standard_data(self, stock_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取标准化后的数据，统一输出：symbol, date, open, close, high, low, volume, amount
        """
        df = self.get_raw_data(stock_code)  # 这是获取csv所有的数据，下面是需要标准化并筛选日期
        if self.data_type == 'efinance':
            df = df.rename(columns={
                '日期':       '日期',   #'date',
                '开盘':       '开盘',   #'open',
                '收盘':       '收盘',   #'close',
                '最高':       '最高',   #'high',
                '最低':       '最低',   #'low',
                '成交量':      '成交量',    #volume',
                '成交额':      '成交额',    #amount'
            })
        elif self.data_type == 'tushare':
            df = df.rename(columns={
                'trade_date':    '日期',   #'date',
                'open':          '开盘',   #'open',
                'close':         '收盘',   #'close',
                'high':          '最高',   #'high',
                'low':           '最低',   #'low',
                'vol':           '成交量',    #volume',
                'amount':        '成交额',    #amount'
            })
        elif self.data_type == 'akshare':
            df = df.rename(columns={
                'date':     '日期',   #'date',
                'open':     '开盘',   #'open',
                'close':    '收盘',   #'close',
                'high':     '最高',   #'high',
                'low':      '最低',   #'low',
                'volume':   '成交量',    #volume',
                'amount':   '成交额',    #amount'
            })
        else:
            raise ValueError('不支持的数据类型')
            
        # 统一日期格式
        if self.data_type == 'tushare':
            # tushare的日期格式是YYYYMMDD，需要特殊处理
            df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
        else:
            # 其他数据源的日期格式是标准格式
            df['日期'] = pd.to_datetime(df['日期'])
            
        if start_date:
            df = df[df['日期'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['日期'] <= pd.to_datetime(end_date)]
        df['股票代码'] = stock_code
        # 只保留标准列
        standard_columns = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额']
        df = df[standard_columns].copy()
        df = df.sort_values('日期', ascending=False).reset_index(drop=True)  #   ascending=False第一行是最新的日期，ascending=True最后一行是最新的日期
        return df
