import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def lagrange_interpolation(x, x_data, y_data):
    """
    实现拉格朗日多项式插值算法
    :param x: 要插值的点或点的数组
    :param x_data: 已知数据点的x坐标数组
    :param y_data: 已知数据点的y坐标数组
    :return: 插值结果，若x为标量返回标量，若为数组返回数组
    """
    x = np.atleast_1d(x)
    n = len(x_data)
    result = np.zeros_like(x)
    for i in range(n):
        L = np.ones_like(x)
        for j in range(n):
            if i != j:
                L *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += y_data[i] * L
    if np.isscalar(x):
        return result[0]
    return result


def cubic_spline_interpolation(x, x_data, y_data):
    """
    实现三次样条插值算法
    :param x: 要插值的点或点的数组
    :param x_data: 已知数据点的x坐标数组
    :param y_data: 已知数据点的y坐标数组
    :return: 插值结果，若x为标量返回标量，若为数组返回数组
    """
    cs = CubicSpline(x_data, y_data)
    return cs(x)


def find_peak(x, y):
    """
    根据插值结果确定共振峰位置(峰值能量)和计算共振峰的半高全宽(FWHM)
    :param x: x坐标数组
    :param y: y坐标数组
    :return: 峰值位置，半高全宽
    """
    peak_index = np.argmax(y)
    peak_x = x[peak_index]
    peak_y = y[peak_index]
    half_max = peak_y / 2
    left_indices = np.where(y[:peak_index] <= half_max)[0]
    right_indices = np.where(y[peak_index:] <= half_max)[0]
    if len(left_indices) > 0:
        left_x = x[left_indices[-1]]
    else:
        left_x = x[0]
    if len(right_indices) > 0:
        right_x = x[right_indices[0] + peak_index]
    else:
        right_x = x[-1]
    fwhm = right_x - left_x
    return peak_x, fwhm


def plot_interpolation_comparison(x_data, y_data):
    """
    绘制拉格朗日插值曲线、三次样条插值曲线与原始数据点的对比图
    :param x_data: 已知数据点的x坐标数组
    :param y_data: 已知数据点的y坐标数组
    """
    x = np.linspace(min(x_data), max(x_data), 1000)
    y_lagrange = lagrange_interpolation(x, x_data, y_data)
    y_spline = cubic_spline_interpolation(x, x_data, y_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Original Data', marker='o', c='b')
    plt.plot(x, y_lagrange, label='Lagrange Interpolation', c='r')
    plt.plot(x, y_spline, label='Cubic Spline Interpolation', c='g')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Cross Section (mb)')
    plt.title('Interpolation Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('interpolation_comparison.png')
    plt.close()
