import numpy as np
from collections import Counter


def sigmoid(t):
    return 1. / (1. + np.exp(-t))


class MultiLogisticRegression:
    def __init__(self):
        self.classifiers = {}
        self.n_classes = None

    def fit(self, X, y, lr=1e-3, n_iter=1e3, epsilon=1e-8):
        '''训练函数'''
        y_counter = Counter(y)
        self.n_classes = len(y_counter)
        for k in y_counter:
            y_k = y.copy()
            # 做一个二分类
            y_k[y_k != k] = -1
            y_k[y_k != -1] = 1
            y_k[y_k == -1] = 0
            c_k = LogisticRegression().fit(X, y_k, lr, n_iter, epsilon)
            self.classifiers[k] = c_k

    def predict(self, X):
        '''预测，取概率最大的类别作为预测结果'''
        y_prob = np.zeros((len(X), self.n_classes))
        for y_k, classifier in self.classifiers.items():
            y_k_prob = classifier.predict_prob(X)  # 是第k类的概率
            y_prob[:, y_k] = y_k_prob

        y_pred = np.argmax(y_prob, axis=1)
        return y_pred


class LogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit(self, X, y, lr=1e-3, n_iter=1e3, epsilon=1e-8):
        '''
        训练函数
        :param X: 输入样本特征
        :param y: 输入样本标签
        :param lr: 学习lv
        :param n_iter: 最大迭代次数
        :param epsilon: 相邻两次迭代的loss小于这个数的时候停止训练
        :return: self
        '''
        assert len(X) == len(y), "the size of X_train must be equal to the size of y_train"

        def loss(y, y_hat):

            try:
                l = - np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat)) / len(y)
                return l
            except:
                return float('inf')

        def d_loss(X_b, y, y_hat):
            return X_b.T.dot(y_hat - y) / len(y)

        def gradient_descent(X_b, y, initial_w, lr, n_iter, epsilon):
            w = initial_w
            i = 0
            y_prob = sigmoid(X_b.dot(w))

            while i < n_iter:
                gradient = d_loss(X_b, y, y_prob)
                w = w - lr * gradient
                y_prob_new = sigmoid(X_b.dot(w))
                if abs(loss(y, y_prob) - loss(y, y_prob_new)) < epsilon:
                    break

                y_prob = y_prob_new
                i += 1

            return w

        X_b = np.hstack([X, np.ones((len(X), 1))])
        init_w = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y, init_w, lr, n_iter, epsilon)

        self.coef_ = self._theta[:-1]
        self.intercept_ = self._theta[-1]

        return self

    def predict_prob(self, X):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([X, np.ones((len(X), 1))])
        y_prob = sigmoid(X_b.dot(self._theta))
        return y_prob

    def predict(self, X):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        y_prob = self.predict_prob(X)
        return np.array(y_prob >= 0.5, dtype=np.int)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    print(y)
    # X = X[y < 2, :2]
    # y = y[y < 2]
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    # log_reg = LogisticRegression()
    # log_reg.fit(X_train, y_train, n_iter=1e4)
    # y_pred = log_reg.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # print(accuracy_score(y_pred, y_test))
    # print(log_reg.predict_prob(X_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    m_log_reg = MultiLogisticRegression()
    m_log_reg.fit(X_train, y_train, n_iter=1e5, lr=1e-2)
    y_pred = m_log_reg.predict(X_test)
    print(accuracy_score(y_pred, y_test))
