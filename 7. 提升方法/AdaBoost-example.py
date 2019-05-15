import numpy as np
from copy import copy


class SimpleClassifier:
    '''弱分类模型'''
    def __init__(self):
        '''初始化'''
        self.sp_v = None  # 划分的值
        self.sp_i = 0  # 按(X[sp_i] + X[sp_i+1]) / 2 进行划分
        self.e = float('inf')  # 误差率e

        self.left = True  # Ture, x<sp_v时候，y=1

    def calc_loss(self,  y, y_hat, d):
        '''对两个方向计算误差率e'''
        idx = np.argwhere(y != y_hat).flatten()
        e1 = np.sum(d[idx])
        y_hat *= -1
        idx = np.argwhere(y != y_hat).flatten()
        e2 = np.sum(d[idx])

        return e1, e2

    def predict(self, X, sp_v):
        '''预测'''
        y_hat = []
        for x in X:
            if x < sp_v:
                y_hat.append(1)
            else:
                y_hat.append(-1)
        y_hat = np.array(y_hat)
        if not self.left:
            y_hat *= -1

        return y_hat

    def find_split_idx(self, X, y, d):
        '''查找划分点'''
        e = float('inf')
        left = True
        for i in range(0, len(y) - 1):
            sp_v = (X[i] + X[i + 1]) / 2
            y_hat = self.predict(X, sp_v)
            el, er = self.calc_loss(y, y_hat, d)
            if el < er and el < e:
                e = el
                left = True
                self.sp_v = sp_v
                self.sp_i = i
            elif el > er and er < e:
                e = er
                left = False
                self.sp_v = sp_v
                self.sp_i = i

        self.left = left
        self.e = e

    def train(self, X, y, d):
        '''训练一次若分类模型'''
        assert len(X) == len(y), "error"
        self.find_split_idx(X, y, d)


class AdaBoost:
    '''AdaBoost模型'''
    def __init__(self, model):
        '''初始化'''
        self.model = model  # 基础模型
        self.a = []  # 存放系数a的值
        self.D = None  # 权值D，个数等于样本数量
        self.model_list = []  # 存放每个弱分类器

    def sign(self, y_hat):
        '''指示函数'''
        y_hat[y_hat < 0] = -1
        y_hat[y_hat > 0] = 1
        return y_hat

    def predict(self, X):
        '''预测函数sign(f(x))'''
        if len(self.model_list) == 0:
            return self.model.predict(X, self.model.sp_v)
        else:
            y_hat = np.zeros(len(X))
            for i, m in enumerate(self.model_list):
                y_hat += m.predict(X, m.sp_v) * self.a[i]
            return self.sign(y_hat)

    def calc_a(self, e):
        '''计算a'''
        self.a[-1] = np.log((1 - e) / e) / 2

    def update_d(self, X, y):
        '''更新权重w'''
        t = self.D * np.exp(-self.a[-1] * y * self.model.predict(X, self.model.sp_v))
        z = np.sum(t)
        self.D = t / z

    def fit(self, X, y):
        '''单次训练'''
        if self.D is None:
            self.D = np.full_like(y, 1 / len(y), dtype=np.float)

        self.a.append(1)

        self.model.train(X, y, self.D)
        self.sp_v = (X[self.model.sp_i] + X[self.model.sp_i + 1]) / 2
        self.calc_a(self.model.e)
        self.update_d(X, y)
        self.model_list.append(copy(self.model))


if __name__ == "__main__":
    X = np.arange(10)
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

    clf = SimpleClassifier()
    ada_boost = AdaBoost(clf)

    ada_boost.fit(X, y)
    print(ada_boost.D)
    print(ada_boost.a)
    print(ada_boost.sp_v)

    ada_boost.fit(X, y)
    print(ada_boost.D)
    print(ada_boost.a)
    print(ada_boost.sp_v)

    ada_boost.fit(X, y)
    print(ada_boost.D)
    print(ada_boost.a)
    print(ada_boost.sp_v)

    y_hat = ada_boost.predict(X)

    print(y_hat)




