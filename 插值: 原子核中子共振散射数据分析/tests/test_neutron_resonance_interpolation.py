import numpy as np
import os
import sys
import unittest

# 修复路径导入问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 从学生实现中导入函数
try:
    from NeutronResonanceInterpolation_student import (
        lagrange_interpolation,
        cubic_spline_interpolation,
        find_peak
    )
except ImportError:
    # 备用导入路径
    from ..src.NeutronResonanceInterpolation_student import (
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
        self.assertAlmostEqual(result, 45.0, places=1)
        
        # 测试数组输入
        results = lagrange_interpolation(self.test_points, self.energy, self.cross_section)
        self.assertEqual(len(results), len(self.test_points))
        self.assertTrue(isinstance(results, np.ndarray))
        
        # 测试边界点
        boundary_result = lagrange_interpolation(200, self.energy, self.cross_section)
        self.assertAlmostEqual(boundary_result, 4.7, places=1)
    
    def test_cubic_spline_interpolation(self):
        """测试三次样条插值函数"""
        # 测试单个点
        result = cubic_spline_interpolation(50, self.energy, self.cross_section)
        self.assertAlmostEqual(result, 45.0, places=1)
        
        # 测试数组输入
        results = cubic_spline_interpolation(self.test_points, self.energy, self.cross_section)
        self.assertEqual(len(results), len(self.test_points))
        self.assertTrue(isinstance(results, np.ndarray))
        
        # 测试边界点
        boundary_result = cubic_spline_interpolation(200, self.energy, self.cross_section)
        self.assertAlmostEqual(boundary_result, 4.7, places=1)
    
    def test_find_peak(self):
        """测试寻找峰值函数"""
        # 生成测试数据
        x = np.linspace(0, 200, 500)
        y = lagrange_interpolation(x, self.energy, self.cross_section)
        
        # 测试峰值查找
        peak_x, fwhm = find_peak(x, y)
        self.assertTrue(70 < peak_x < 90, f"峰值位置应为70-90 MeV，但得到{peak_x}")
        # 放宽FWHM范围检查
        self.assertTrue(20 < fwhm < 100, f"FWHM应为20-100 MeV，但得到{fwhm}")
        
        # 测试无效输入
        with self.assertRaises(ValueError):
            find_peak(np.array([]), np.array([]))

if __name__ == '__main__':
    unittest.main()    
