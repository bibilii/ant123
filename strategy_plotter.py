 # !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

class StrategyPlotter:
    def __init__(self):
        # 设置matplotlib中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.check_chinese_font()

    def check_chinese_font(self):
        """检查中文字体是否可用"""
        fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
        if 'SimHei' not in fonts:
            print("警告: SimHei字体未找到，尝试使用其他中文字体...")
            chinese_fonts = ['Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
            for font in chinese_fonts:
                if font in fonts:
                    plt.rcParams['font.sans-serif'] = [font]
                    print(f"使用字体: {font}")
                    return True
            print("警告: 未找到合适的中文字体，图表中文可能无法正确显示")
            return False
        return True

    def plot_strategy_result(self, df, symbol, output_file):
        """
        绘制策略结果图表
        参数:
            df (pd.DataFrame): 策略结果数据
            symbol (str): 股票代码
            output_file (str): 输出文件路径
        """
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        # 绘制收盘价和年均线
        ax1.plot(df['日期'], df['收盘'], label='收盘价', color='black', linewidth=1)
        if 'MA_250' in df.columns:
            ax1.plot(df['日期'], df['MA_250'], label='250日均线', color='red', linewidth=1)
        # 标记买入和卖出点
        buy_points = df[df['业务名称'] == '买入']
        sell_points = df[df['业务名称'] == '卖出']
        if not buy_points.empty:
            ax1.scatter(buy_points['日期'], buy_points['收盘'], marker='^', color='red', s=100, label='买入信号')
        if not sell_points.empty:
            ax1.scatter(sell_points['日期'], sell_points['收盘'], marker='v', color='green', s=100, label='卖出信号')
        ax1.set_title(f'{symbol} 股票价格和年均线')
        ax1.legend()
        ax1.grid(True)
        # 计算股票市值
        df['股票市值'] = df['持股数量'] * df['收盘']
        # 绘制资产曲线
        ax2.plot(df['日期'], df['余额'], label='现金余额', color='green', linewidth=1)
        ax2.plot(df['日期'], df['股票市值'], label='股票市值', color='red', linewidth=1)
        ax2.plot(df['日期'], df['总金额'], label='总资产', color='blue', linewidth=2)
        # 在股票市值和总资产之间填充红色区域
        ax2.fill_between(df['日期'], df['股票市值'], df['总金额'], color='red', alpha=0.3, label='现金余额区域')
        ax2.set_title(f'{symbol} 资产分布')
        ax2.legend()
        ax2.grid(True)
        # 设置y轴格式为货币格式（不再用¥，避免字体警告）
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} 元'))
        # 设置x轴日期格式
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
