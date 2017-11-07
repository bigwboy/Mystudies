# -*- coding: utf8 -*-
# time:2017/10/30 下午1:44
# VERSION:1.0
# __OUTHOR__:guangguang

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


# 感知机
class Perceprtron(object):
    def __init__(self, eta=0.01, n_iter=10):
        """
        感知机分类：
            参数
                eta:float   学习率 ( 介于0.0 与1.0之间)
                n_iter:int  迭代次数
            属性
                w_: 1d-array    权重
                errors_: list   误差值列表
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
         拟合训练数据
                参数
                ------------
                X: {array-like}, shape=[n_samples, n_features]
                    训练样本【n_samples：特征  ，n_features数量】
                y: array-like, shape=[n_smaples]
                    训练目标值shape=[n_smaples]
                Returns
                ----------
                self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errrors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                #梯度迭代
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errrors_.append(errors)
        return self

    #输入并加权和
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    #激活函数
    #heaviside阶越函数
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#分类显示
def plot_decision_region(X, y, classifier, resolution=0.02):
    makers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contour(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=makers[idx], label=cl)




#自适应神经元
class AdalineGD(object):
    """自适应线性神经元分类器：
            参数
                eta:float        学习率 ( 介于0.0 与1.0之间)
                n_iter:int       迭代次数
            属性
                w_: 1d-array     权重
                errors_: list    误差值列表

    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        拟合训练数据
            参数
            X: {array-like}, shape=[n_samples, n_features]     训练样本【n_samples：特征  ，n_features数量】
            y: array-like, shape=[n_smaples]                   训练目标值shape=[n_smaples]

        """
        #初始化权值
        self.w_ = np.zeros(1 + X.shape[1])

        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            #损失函数
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """ 计算净输入"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ 计算线性激活"""
        return self.net_input(X)

    def predict(self, X):
        """ 激励函数"""
        return np.where(self.activation(X) >= 0.0, 1, -1)






# debug
if __name__ == "__main__":
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    #print  df.tail()
    #获取数据
    y = df.iloc[0:100, 4].values
    #转换
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    # 数据显示
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='b', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceprtron(eta=0.1, n_iter=12)
    ppn.fit(X, y)

    #迭代收敛显示
    plt.plot(range(1, len(ppn.errrors_) + 1), ppn.errrors_, marker='o')
    plt.xlabel('Epoches')
    plt.ylabel('Number of misclassifications')
    plt.show()

    #最终分类显示
    plot_decision_region(X, y, classifier=ppn)
    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.legend(loc='upper left')
    plt.show()
