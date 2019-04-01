import numpy as np
from collections import Counter


def cal_empirical_entropy(y):
    '''计算经验熵'''
    ret = 0.0
    y_counter = Counter(y)
    # v : |C_k|
    # len(y): |D|
    for v in y_counter.values():
        ret += (v / len(y)) * np.log2(v / len(y))

    return -ret


def cal_conditional_entropy(X, y, dim, cal_empirical_entropy_=None):
    '''计算条件熵'''
    ret = 0.0
    x_counter = Counter(X[:, dim])
    # x_i: X的第dim个维度的第i个特征值
    # x_c: x_i的统计结果
    if cal_empirical_entropy_ is not None:
        for x_i, x_c in zip(x_counter.keys(), x_counter.values()):
            y_k = y[X[:, dim] == x_i]
            ret += (x_c / len(y)) * cal_empirical_entropy_(y_k)
    else:
        for x_c in x_counter.values():
            ret -= (x_c / len(y)) * np.log2(x_c / len(y))

    return ret


def cal_information_gan(X, y, dim):
    '''计算信息增益'''
    return cal_empirical_entropy(y) - cal_conditional_entropy(X, y, dim, cal_empirical_entropy)


def cal_information_gan_ratio(X, y, dim):
    gan = cal_information_gan(X, y, dim)
    h = cal_conditional_entropy(X, y, dim)

    # 处理 h == 0
    if abs(h) < 1e-9:
        return 0
    return gan / h


class Node:
    def __init__(self):
        self.sub = {}  # 子树
        self.leaf = None  # 叶节点，最终的预测类别
        self.condition = lambda x_i: x_i  # 子树的方向
        self.sample_classes = []  # 该节点包含的样本类别
        self.split_dim = None  # 划分维度


class DecisionTree:
    def __init__(self):
        self.root = None
        self.n_leaf = 0
        self.n_features = 0

    def loss_function(self, samples_n, entropy, alpha, diff_leaf):
        return np.sum(np.array(samples_n) * np.array(entropy)) + alpha * (self.n_leaf - diff_leaf)

    def __build_tree(self, X, y, epsilon):
        assert len(X) == len(y), "X and y len error"

        node = Node()  # 创建树的根节点
        node.sample_classes = y  # 对样本类别做一个保存
        y_counter = Counter(y)  # 对 label做一个统计

        # 1. 判断所有实例是否属于同一类， 设置node为点节点数
        # 2. 特征集是否是空集， 设置node为点节点数
        if len(y_counter) == 1 or len(X) == 0:
            node.leaf = y_counter.most_common(1)[0][0]
            self.n_leaf += 1
        else:
            # 3. 计算各个维度的信息增益比，找到最好的划分维度
            best_ratio = float('-inf')
            best_dim = None
            for dim in range(X.shape[-1]):

                gan_ratio = cal_information_gan_ratio(X, y, dim)
                if gan_ratio > best_ratio:
                    best_ratio = gan_ratio
                    best_dim = dim

            node.split_dim = best_dim

            # 4. 如果best_ratio 小于ε， 设置T为点节点数
            if best_ratio < epsilon:
                node.leaf = y_counter.most_common(1)[0][0]
                self.n_leaf += 1
            else:
                x_counter = Counter(X[:, best_dim])

                # 5. 对best_dim下的每个特征值，将D划分为子空间，并构建子节点
                for x_i in x_counter:
                    sub_X = X[X[:, best_dim] == x_i]
                    sub_y = y[X[:, best_dim] == x_i]
                    index = node.condition(x_i)
                    node.sub[index] = self.__build_tree(sub_X, sub_y, epsilon)

        return node

    def fit(self, X, y, epsilon, alpha):
        self.n_features = X.shape[1]
        self.root = self.__build_tree(X, y, epsilon)
        self.__pruning_tree(self.root, alpha)
        return self

    def __pruning_tree(self, node, alpha):
        # 计算节点熵
        entropy = cal_empirical_entropy(node.sample_classes)
        # 该节点是叶节点
        if node.leaf is not None:
            return 1, entropy, node.leaf
        # 否则遍历子节点
        else:
            n_sub_samples = 0  # 子节点的个数
            n_sub_sample_classes = []  # 子节点的样本的类别
            sub_node_entropy = []  # 子节点的先验熵
            sub_leaf_class = []  # 子节点叶子的存放的类别
            for index in node.sub:
                sub_node = node.sub[index]
                i, e, l = self.__pruning_tree(sub_node, alpha)
                n_sub_samples += i
                n_sub_sample_classes.append(len(sub_node.sample_classes))
                sub_node_entropy.append(e)
                sub_leaf_class.append(l)

            # 找到一组叶节点
            if n_sub_samples == len(node.sub):
                # 比较损失函数
                b_loss = self.loss_function(n_sub_sample_classes, sub_node_entropy, alpha, 0)
                a_loss = self.loss_function(len(node.sample_classes), entropy, alpha, n_sub_samples)
                if a_loss < b_loss:
                    # 剪枝
                    c = Counter(sub_leaf_class)
                    node.sub.clear()  # 清理子树
                    node.leaf = c.most_common(1)[0][0]  # 设置叶节点
                    self.n_leaf -= (n_sub_samples - 1)  # 叶子的数量减少 子节点的个数 + 1 个

                    return 1, entropy, node.leaf

            return 0, entropy, node.leaf

    def __predict(self, x, node):
        '''递归寻找预测值'''
        split_dim = node.split_dim
        sub_node_idx = node.condition(x[split_dim])

        if len(node.sub) == 0:
            return node.leaf
        else:
            return self.__predict(x, node.sub[sub_node_idx])

    def predict(self, X):
        '''预测函数，外部调用'''
        assert X.shape[1] == self.n_features, "shape error"
        ret = []
        for x in X:
            ret.append(self.__predict(x, self.root))

        return np.array(ret)


if __name__ == '__main__':
    D = np.array([['青年', '否', '否', '一般', '否'],
                  ['青年', '否', '否', '好', '否'],
                  ['青年', '是', '否', '好', '是'],
                  ['青年', '是', '是', '一般', '是'],
                  ['青年', '否', '否', '一般', '否'],
                  ['中年', '否', '否', '一般', '否'],
                  ['中年', '否', '否', '好', '否'],
                  ['中年', '是', '是', '好', '是'],
                  ['中年', '否', '是', '非常好', '是'],
                  ['中年', '否', '是', '非常好', '是'],
                  ['老年', '否', '是', '非常好', '是'],
                  ['老年', '否', '是', '好', '是'],
                  ['老年', '是', '否', '好', '是'],
                  ['老年', '是', '否', '非常好', '是'],
                  ['老年', '否', '否', '一般', '否'],
                  ['青年', '否', '否', '一般', '是']])

    X = D[:, :-1]
    y = D[:, -1]

    t = DecisionTree()
    t.fit(X, y, 1e-3, 0.5)

    print(t.predict(X[:2]))

