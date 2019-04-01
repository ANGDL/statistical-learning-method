import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def split(X, y, dim, value):
    '''
    :param X:
    :param y:
    :param dim: 切分维度
    :param value: 切分变量
    :return:
    '''
    index_l = (X[:, dim] <= value)
    index_r = (X[:, dim] > value)

    return X[index_l], X[index_r], y[index_l], y[index_r]


class Node:
    def __init__(self):
        self.left = None  # 子树
        self.right = None  # 叶节点，最终的预测类别
        self.split_dim = None  # 切分维度
        self.split_value = None  # 切分的值
        self.n_leaf = 0  # 以该节点为子树的叶节点个数


class CartBase:

    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2):
        self.root = None
        self.depth = 0
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_features = 0

    def loss_func(self, y, y_hat):
        print(self.loss_func.__name__)
        return -1

    def pred_func(self, y):
        print(self.pred_func.__name__)
        return y[0]

    def __build_tree(self, X, y):
        assert len(X) == len(y), 'shape error'
        # 达到最大深度
        if self.max_depth is not None and self.depth >= self.max_depth:
            return None

        # 样本数达到min_samples_leaf时，不在换份，该节点作为叶节点
        do_split = len(X) // 2 > self.min_samples_leaf

        node = Node()

        best_loss = float('inf')
        best_split_dim = -1
        best_split_value = -1

        if do_split:
            for dim in range(X.shape[1]):
                sorted_index = np.argsort(X[:, dim])
                for i in range(1, len(X)):
                    if X[sorted_index[i], dim] != X[sorted_index[i-1], dim]:
                        v = (X[sorted_index[i], dim] + X[sorted_index[i-1], dim]) / 2
                        X_l, X_r, y_l, y_r = split(X, y, dim, v)

                        if len(y_l) <= self.min_samples_split or len(y_r) <= self.min_samples_split:
                            continue

                        loss_l = self.loss_func(y_l, self.pred_func(y_l))
                        loss_r = self.loss_func(y_r, self.pred_func(y_r))
                        loss_sum = loss_l + loss_r
                        if loss_sum < best_loss:
                            best_loss = loss_sum
                            best_split_dim = dim
                            best_split_value = v
            # 找到合适的切分点
            if best_split_dim != -1:
                X_l, X_r, y_l, y_r = split(X, y, best_split_dim, best_split_value)
                node.split_dim = best_split_dim
                node.split_value = best_split_value

                if len(X_l) < self.min_samples_leaf or len(X_r) < self.min_samples_leaf:
                    print(len(X_l), len(X_r))

                if len(X_l) >= self.min_samples_leaf:
                    node.left = self.__build_tree(X_l, y_l)

                if len(X_r) >= self.min_samples_leaf:
                    node.right = self.__build_tree(X_r, y_r)

                if node.left is not None or node.right is not None:
                    self.depth += 1
            # 未找到合适的切分点
            else:
                node.n_leaf += 1  # 标记一次叶节点
                node.split_value = self.pred_func(y)
        else:
            node.n_leaf += 1  # 标记一次叶节点
            node.split_value = self.pred_func(y)

        return node

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self.__build_tree(X, y)
        return self

    def __predict(self, x, node):
        split_dim = node.split_dim
        split_value = node.split_value

        if node.left is None and node.right is None:
            return node.split_value

        if node.left is not None and x[split_dim] <= split_value:
            return self.__predict(x, node.left)
        if node.right is not None and x[split_dim] > split_value:
            return self.__predict(x, node.right)

        if node.left is None and x[split_dim] <= split_value \
                or node.right is None and x[split_dim] > split_value:
            raise Exception('build tree error')

    def predict(self, X):
        assert X.shape[1] == self.n_features, 'input features error'
        ret = []
        for x in X:
            ret.append(self.__predict(x, self.root))

        return np.array(ret)


class CartRegression(CartBase):
    def loss_func(self, y, y_hat):
        return np.sum((y - y_hat) ** 2)

    def pred_func(self, y):
        return np.mean(y)


class CartClassifier(CartBase):
    def loss_func(self, y, y_hat=None):
        y_counter = Counter(y)
        ret = 0.0
        for v in y_counter.values():
            ret += (v / len(y)) ** 2

        return 1 - ret

    def pred_func(self, y):
        y_counter = Counter(y)
        return y_counter.most_common(1)[0][0]


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    dt_reg = CartRegression(min_samples_leaf=3)
    dt_reg.fit(X_train, y_train)
    y_pred = dt_reg.predict(X_test)
    print(mean_squared_error(y_test, y_pred))

    dt_reg_2 = DecisionTreeRegressor()
    dt_reg_2.fit(X_train, y_train)
    y_pred_2 = dt_reg_2.predict(X_test)
    print(mean_squared_error(y_test, y_pred_2))

    X, y = datasets.make_moons(noise=0.25, random_state=666)
    dt_clf = CartClassifier()
    dt_clf.fit(X, y)

    plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.show()
