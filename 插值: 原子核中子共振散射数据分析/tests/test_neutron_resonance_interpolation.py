import numpy as np
import os
import sys
import unittest

# 调整路径以在GitHub CI环境中正确导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从学生实现中导入函数
try:
    from NeutronResonanceInterpolation_student import (
        lagrange_interpolation,
        cubic_spline_interpolation,
        find_peak
    )
except ImportError:
    # 备用导入路径，适应不同的项目结构
    from src.NeutronResonanceInterpolation_student import (
        lagrange_interpolation,
        cubic_spline_interpolation,
        find_peak
    )

class TestNeutronResonanceInterpolation(unittest.TestCase):
    def setUp(self):
        # 使用示例实验数据
        self.energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])  # MeV
        self.cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])  # mb
        self.test_points = np.array([10, 40, 90, 120, 180])  # 测试点
    
    def test_lagrange_interpolation(self):
        """测试拉格朗日插值函数"""
        # 测试单个点
        result = lagrange_interpolation(50, self.energy, self.cross_section)
        self.assertAlmostEqual(result, 45.0, places=1, 
                              msg="拉格朗日插值在已知点上应返回精确值")
        
        # 测试数组输入
        results = lagrange_interpolation(self.test_points, self.energy, self.cross_section)
        self.assertEqual(len(results), len(self.test_points), 
                         "插值结果长度应与输入点数量一致")
        self.assertTrue(isinstance(results, np.ndarray), 
                        "插值结果应为numpy数组")
        
        # 测试边界点
        boundary_result = lagrange_interpolation(200, self.energy, self.cross_section)
        self.assertAlmostEqual(boundary_result, 4.7, places=1, 
                              msg="拉格朗日插值在边界点上应返回精确值")
        
        # 测试超出边界的点
        out_of_bounds = lagrange_interpolation(250, self.energy, self.cross_section)
        self.assertTrue(np.isfinite(out_of_bounds), 
                       "拉格朗日插值应能处理边界外的点")
    
    def test_cubic_spline_interpolation(self):
        """测试三次样条插值函数"""
        # 测试单个点
        result = cubic_spline_interpolation(50, self.energy, self.cross_section)
        self.assertAlmostEqual(result, 45.0, places=1, 
                              msg="三次样条插值在已知点上应返回精确值")
        
        # 测试数组输入
        results = cubic_spline_interpolation(self.test_points, self.energy, self.cross_section)
        self.assertEqual(len(results), len(self.test_points), 
                         "插值结果长度应与输入点数量一致")
        self.assertTrue(isinstance(results, np.ndarray), 
                        "插值结果应为numpy数组")
        
        # 测试边界点
        boundary_result = cubic_spline_interpolation(200, self.energy, self.cross_section)
        self.assertAlmostEqual(boundary_result, 4.7, places=1, 
                              msg="三次样条插值在边界点上应返回精确值")
        
        # 测试超出边界的点（样条函数应能外推）
        out_of_bounds = cubic_spline_interpolation(250, self.energy, self.cross_section)
        self.assertTrue(np.isfinite(out_of_bounds), 
                       "三次样条插值应能处理边界外的点")
    
    def test_find_peak(self):
        """测试寻找峰值函数"""
        # 生成测试数据（使用拉格朗日插值）
        x = np.linspace(0, 200, 500)
        y = lagrange_interpolation(x, self.energy, self.cross_section)
        
        # 测试峰值查找
        peak_x, fwhm = find_peak(x, y)
        self.assertTrue(70 < peak_x < 90, 
                       f"峰值位置应在70-90 MeV之间，但得到{peak_x}")
        # 放宽FWHM范围检查，因为不同插值方法可能得到不同宽度
        self.assertTrue(20 < fwhm < 100, 
                       f"FWHM应在20-100 MeV之间，但得到{fwhm}")
        
        # 使用三次样条插值的测试数据
        y_spline = cubic_spline_interpolation(x, self.energy, self.cross_section)
        peak_x_spline, fwhm_spline = find_peak(x, y_spline)
        self.assertTrue(70 < peak_x_spline < 90, 
                       f"三次样条插值的峰值位置应在70-90 MeV之间，但得到{peak_x_spline}")
        
        # 测试无效输入
        with self.assertRaises(ValueError, msg="空输入应引发ValueError"):
            find_peak(np.array([]), np.array([]))
        
        # 测试单一点输入
        with self.assertRaises(ValueError, msg="单点输入应引发ValueError"):
            find_peak(np.array([1]), np.array([10]))

if __name__ == '__main__':
    unittest.main()    
