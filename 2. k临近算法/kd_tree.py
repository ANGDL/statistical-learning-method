import numpy as np


def median_split(X, ax=0):
    '''
    按维度分割样本数据
    :param X:
    :param ax:
    :return: 中位数、左子树、右子树
    '''
    assert ax < X.ndim, "ax error"

    median_index = int(0.5 * (len(X) + 1)) - 1

    sorted_index = np.argsort(X[:, ax])
    sorted_dataset = X[sorted_index]

    m = sorted_dataset[median_index]
    l, r = sorted_dataset[:median_index], sorted_dataset[median_index+1:]
    if len(sorted_dataset) == 2:
        l, r = r, None
    return m, l, r


class KDNode:
    def __init__(self, x, split_ax, left=None, right=None):
        '''
        初始化分类器
        :param x: 一个样本点
        :param split_ax: 分割维度
        :param left: 左节点
        :param right: 右节点
        '''
        self.data = x
        self.split_ax = split_ax
        self.left = left
        self.right = right
        self.distance = None

    def __repr__(self):
        return tuple(self.data)

    def __str__(self):
        return str(self.data)

    def calc_distance(self, other):
        '''计算欧拉距离'''
        self.distance = np.sqrt(np.sum((self.data - other) ** 2))
        return self

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __ge__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance and np.all(self.data == self.data)


class KDTree:
    def __init__(self, X):
        '''初始化KD Tree'''
        self.root_node = self.__create_tree(X, 0)
        self.X = X

    def query(self, obj, k=3):
        obj = np.array(obj, dtype=np.float32)
        '''返回最近的k个点的距离和索引'''
        k_nearest = []
        self.__search(obj, k, self.root_node, k_nearest)

        k_distance = [n.distance for n in k_nearest]
        k_index = [np.argwhere(np.sum(self.X - n.data, axis=1) == 0)[0][0] for n in k_nearest]
        return k_distance, k_index

    def __create_tree(self, X_, ax_):
        '''
        创建KD Tree
        :param X_: 所有样本点
        :param ax_: 分割轴
        :return: 根节点
        '''
        if X_ is None:
            return None

        n = len(X_)
        if n == 1:
            return KDNode(X_[0], ax_)
        if n == 0:
            return None

        data, left, right = median_split(X_, ax_)
        node = KDNode(data, ax_)
        node.left = self.__create_tree(left, (ax_ + 1) % X_.ndim)
        node.right = self.__create_tree(right, (ax_ + 1) % X_.ndim)
        return node

    def __search(self,  obj, k, curr_node, k_nearest):
        def record_nearest(k_, curr_node_, k_nearest_):
            '''记录距离最近的点'''
            if len(k_nearest_) < k_:
                k_nearest_.append(curr_node_)
            elif max(k_nearest_) > curr_node:
                i = k_nearest_.index(max(k_nearest_))
                k_nearest_[i] = curr_node_

        # 左节点是空的，右节点一定也是空的
        if curr_node is None:
            return

        curr_node.calc_distance(obj)  # 计算当前节点和目标点的距离

        if curr_node.left is None:
            record_nearest(k, curr_node, k_nearest)
            return

        ax = curr_node.split_ax

        to_left = obj[ax] < curr_node.data[ax]
        if to_left:
            self.__search(obj, k, curr_node.left, k_nearest)
        else:
            self.__search(obj, k, curr_node.right, k_nearest)

        record_nearest(k, curr_node, k_nearest)

        ax_node = obj.copy()
        ax_node[ax] = curr_node.data[ax]
        ax_node = KDNode(ax_node, None).calc_distance(obj)

        if max(k_nearest) > ax_node or len(k_nearest) < k:
            if to_left:
                self.__search(obj, k, curr_node.right, k_nearest)
            else:
                self.__search(obj, k, curr_node.left, k_nearest)


if __name__ == '__main__':

    # X = np.random.randint(low=0, high=10, size=(9, 2))
    X_train = np.array([
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ])

    test = np.array([[-1., -5.]])

    X2_train = np.array([
        [6.27, 5.50],
        [1.24, -2.86],
        [17.05, -12.79],
        [-6.88, -5.40],
        [-2.96, -0.50],
        [7.75, -22.68],
        [10.80, -5.03],
        [-4.60, -10.55],
        [-4.96, 12.61],
        [1.75, 12.26],
        [15.31, -13.16],
        [7.83, 15.70],
        [14.63, -0.35]
    ])

    #
    tree = KDTree(X2_train)

    dist, index = tree.query(test[0], k=3)

    print(dist, index)
