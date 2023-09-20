# coding=utf-8
# 最小二乘法线性回归
import numpy as np
import copy
from sklearn.datasets import load_boston  # 导入波士顿房价数据集
class LinerRegression:
    M_x = []
    M_y = []
    M_theta = []  # 定义三个参数向量
     # 定义函数
    def regression(self, data, target):
        self.M_x = np.mat(data)
        fenliang = np.ones((len(data), 1))  # 每个向量对应添加一个分量1，用来对应系数
        self.M_x = np.hstack((self.M_x, fenliang))
        self.M_y = np.mat(target)
        M_x_T = self.M_x.T  # 计算X矩阵的转置矩阵
        self.M_theta = (M_x_T * self.M_x).I * M_x_T * self.M_y.T
 # 通过最小二乘法计算出参数向量
        self.trained = True
    def predict(self, vec):
        if not self.trained:
            print("You haven't finished the regression!")
            return
        M_vec = np.mat(vec)
        fenliang = np.ones((len(vec), 1))
        M_vec = np.hstack((M_vec, fenliang))
        estimate = np.matmul(M_vec, self.M_theta)
        return estimate
if __name__ == '__main__':
    # 从sklearn的数据集中获取相关向量数据集data和房价数据集target
    data, target = load_boston(return_X_y=True)
    lr = LinerRegression()
    lr.regression(data, target)
    test = data[::51]
    M_test = np.mat(test)
    real = target[::51]
    estimate = np.array(lr.predict(M_test))
    for i in range(len(test)):
        print("实际值:", real[i], " 估计值:", estimate[i, 0])