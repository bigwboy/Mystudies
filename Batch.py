# -*- coding: utf8 -*-
#time:2017/11/8 下午5:03
#VERSION:1.0
#__OUTHOR__:guangguang

# -*- coding: utf8 -*-
#time:2017/11/7 下午3:42
#VERSION:1.0
#__OUTHOR__:guangguang

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
import urllib.request

from matplotlib.colors import ListedColormap



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
            #output = self.net_input(X)
            output = self.sigmoid(X)

            #f(y)=y**n
            #f'(y)=n*y**(n-1)
            errors = (y - output)

            #损失函数权重偏导
            SIG=X.T.dot(errors)
            funz=SIG*(1-SIG)
            #权重w迭代
            self.w_[1:] += self.eta * funz

            #参数b迭代
            self.w_[0] += self.eta * errors.sum()

            #损失函数
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        print ('fun=%s*x1+%s*x2+%s'%(self.w_[1],self.w_[2],self.w_[0]))
        print ("ada-----")
        return self

    def net_input(self, X):
        """ 计算净输入"""
        #Z = WX + b
        Z= np.dot(X, self.w_[1:]) + self.w_[0]
        return Z

    def activation(self, X):
        """ 计算线性激活"""
        return self.net_input(X)

    #def predict(self, X):
        """ 激励函数"""
        #Yhat=f(Z)  if f(z)>=0, yhat=1; else yhat=-1
    #    return np.where(self.activation(X) >= 0.0, 1, -1)

    #sigmoid激励
    def sigmoid(self,X):
        return 1.0 / (1 + np.exp(-self.activation(X)))
pass

#分类显示
def plot_decision_region(X, y, classifier, resolution=0.02):
    makers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.sigmoid(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contour(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=makers[idx], label=cl)



#debug
if __name__ == "__main__":
    #df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    context = ssl._create_unverified_context()
    file = urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                                  context=context)
    df = pd.read_csv(file, header=None)
    # print  df.tail()
    # 获取数据
    y = df.iloc[0:100, 4].values
    # 转换
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    X_std=np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    # 数据显示
    #plt.scatter(X_std[:50, 0], X_std[:50, 1], color='red', marker='o', label='setosa')
    #plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color='b', marker='x', label='versicolor')
    #plt.xlabel('petal length')
    #plt.ylabel('sepal length')
    #plt.legend(loc='upper left')
    #plt.show()

    fig,ax = plt.subplots(nrows=1, ncols=2,figsize=(8,4))
    ada1 = AdalineGD(n_iter=50, eta=0.01,).fit(X_std,y)
    ax[0].plot(range(1,len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adalie - Learning rata 0.01')

    ada2 = AdalineGD(n_iter=50, eta=0.0001, ).fit(X_std, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adalie - Learning rata 0.0001')
    plt.show()

# 最终分类显示
    plot_decision_region(X_std, y, classifier=ada1)
    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.legend(loc='upper left')
    plt.show()



