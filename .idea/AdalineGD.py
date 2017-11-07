# -*- coding: utf8 -*-
#time:2017/11/7 下午3:42
#VERSION:1.0
#__OUTHOR__:guangguang

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




#debug
if __name__ == "__main__":
    pass