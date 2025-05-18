#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author   : huyifei   @Time : 2025/04/18 23:35:55

import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

class PerformanceTester:
    """性能测试类"""
    
    def __init__(self):
        """初始化性能测试器"""
        self.results = {
            '数据大小': [],
            'CPU时间': [],
            'GPU时间': [],
            '加速比': []
        }
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def test_matrix_multiplication(self, size):
        """测试矩阵乘法性能"""
        # 创建随机矩阵
        matrix_a = torch.randn(size, size)
        matrix_b = torch.randn(size, size)
        
        # CPU测试
        start_time = time.time()
        cpu_result = torch.matmul(matrix_a, matrix_b)
        cpu_time = time.time() - start_time
        
        # GPU测试
        if torch.cuda.is_available():
            matrix_a_gpu = matrix_a.cuda()
            matrix_b_gpu = matrix_b.cuda()
            
            # 预热
            torch.matmul(matrix_a_gpu, matrix_b_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            gpu_result = torch.matmul(matrix_a_gpu, matrix_b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # 验证结果
            is_close = torch.allclose(cpu_result, gpu_result.cpu(), rtol=1e-4, atol=1e-4)
            if not is_close:
                print(f"警告: 矩阵大小 {size}x{size} 的结果验证失败")
            
            return cpu_time, gpu_time
        else:
            return cpu_time, None
    
    def test_moving_average(self, size, window=20):
        """测试移动平均计算性能"""
        # 创建随机数据
        data = torch.randn(size)
        
        # CPU测试
        start_time = time.time()
        cpu_ma = torch.zeros_like(data)
        for i in range(len(data) - window + 1):
            cpu_ma[i + window - 1] = data[i:i+window].mean()
        cpu_time = time.time() - start_time
        
        # GPU测试
        if torch.cuda.is_available():
            data_gpu = data.cuda()
            
            # 预热
            gpu_ma = torch.zeros_like(data_gpu)
            for i in range(len(data_gpu) - window + 1):
                gpu_ma[i + window - 1] = data_gpu[i:i+window].mean()
            torch.cuda.synchronize()
            
            start_time = time.time()
            gpu_ma = torch.zeros_like(data_gpu)
            for i in range(len(data_gpu) - window + 1):
                gpu_ma[i + window - 1] = data_gpu[i:i+window].mean()
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # 验证结果
            is_close = torch.allclose(cpu_ma, gpu_ma.cpu(), rtol=1e-4, atol=1e-4)
            if not is_close:
                print(f"警告: 数据大小 {size} 的结果验证失败")
            
            return cpu_time, gpu_time
        else:
            return cpu_time, None
    
    def run_tests(self, sizes):
        """运行性能测试"""
        print("开始性能测试...")
        
        for size in sizes:
            print(f"\n测试数据大小: {size}")
            
            # 测试矩阵乘法
            cpu_time, gpu_time = self.test_matrix_multiplication(size)
            
            if gpu_time is not None:
                self.results['数据大小'].append(size)
                self.results['CPU时间'].append(cpu_time)
                self.results['GPU时间'].append(gpu_time)
                self.results['加速比'].append(cpu_time / gpu_time)
                
                print(f"CPU时间: {cpu_time:.4f}秒")
                print(f"GPU时间: {gpu_time:.4f}秒")
                print(f"加速比: {cpu_time/gpu_time:.2f}x")
            else:
                print("GPU不可用，跳过测试")
    
    def plot_results(self):
        """绘制性能对比图"""
        if not self.results['数据大小']:
            print("没有测试数据可供绘图")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制时间对比图
        x = self.results['数据大小']
        ax1.plot(x, self.results['CPU时间'], 'b-o', label='CPU时间')
        ax1.plot(x, self.results['GPU时间'], 'r-o', label='GPU时间')
        ax1.set_xlabel('数据大小')
        ax1.set_ylabel('计算时间 (秒)')
        ax1.set_title('CPU vs GPU 计算时间对比')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制加速比图
        ax2.plot(x, self.results['加速比'], 'g-o')
        ax2.set_xlabel('数据大小')
        ax2.set_ylabel('加速比 (CPU时间/GPU时间)')
        ax2.set_title('GPU加速比')
        ax2.grid(True)
        
        # 添加水平参考线
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"gpu_cpu_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n性能对比图已保存到: {filename}")
        
        # 显示图表
        plt.show()
    
    def save_results(self):
        """保存测试结果"""
        if not self.results['数据大小']:
            print("没有测试数据可供保存")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存到CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"gpu_cpu_comparison_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n测试结果已保存到: {filename}")

def main():
    """主函数"""
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，将只进行CPU测试")
    
    # 创建性能测试器
    tester = PerformanceTester()
    
    # 设置测试数据大小
    sizes = [1000, 2000, 4000, 8000, 16000]
    
    # 运行测试
    tester.run_tests(sizes)
    
    # 绘制结果
    tester.plot_results()
    
    # 保存结果
    tester.save_results()

if __name__ == "__main__":
    main() 