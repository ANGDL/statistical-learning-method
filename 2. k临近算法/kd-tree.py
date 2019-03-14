import numpy as np
from sklearn.neighbors import NearestNeighbors
np.random.seed(000)


def median(X, ax=0):
    assert ax < X.ndim, "ax error"

    median_index = int(0.5 * (len(X) + 1)) - 1

    sorted_index = np.argsort(X[:, ax])
    sorted_X = X[sorted_index]

    m = sorted_X[median_index]
    l, r = sorted_X[:median_index], sorted_X[median_index+1:]
    if len(sorted_X) == 2:
        l, r = r, None
    return m, l, r


class Node:
    def __init__(self, x, split_ax, left=None, right=None):
        self.data = x
        self.split_ax = split_ax
        self.left = left
        self.right = right

    def __repr__(self):
        return tuple(self.data)

    def __str__(self):
        return str(self.data)


class DisNode:
    def __init__(self, x, o):
        self.data = x
        self.distance = self.__distance(o)

    def __distance(self, o):
        return np.sqrt(np.sum((self.data - o) ** 2))

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __ge__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance and self.data == self.data


class KdTree:
    def __init__(self):
        self.root = None
        self.ret = []

    def create_tree(self, X):
        self.root = self.__create_tree(X, 0)

    def __create_tree(self, X, ax):
        if X is None:
            return None

        n = len(X)
        if n == 1:
            return Node(X[0], ax)
        if n == 0:
            return None

        m, left, right = median(X, ax)
        node = Node(m, ax)
        node.left = self.__create_tree(left, (ax+1) % X.ndim)
        node.right = self.__create_tree(right, (ax+1) % X.ndim)
        return node

    def search(self, k, obj):
        self.__search(k, obj, self.root)

    def __insert_to_ret(self, node, k):
        if len(self.ret) < k:
            self.ret.append(node)
        elif max(self.ret) > node:
            i = self.ret.index(max(self.ret))
            self.ret[i] = node

    def __search(self,  k, obj, curr_node):
        # 左节点是空的，右节点一定也是空的
        if curr_node is None:
            return

        curr_dis_node = DisNode(curr_node.data, obj)

        if curr_node.left is None:
            self.__insert_to_ret(curr_dis_node, k)
            return

        ax = curr_node.split_ax

        to_left = obj[ax] < curr_node.data[ax]
        if to_left:
            self.__search(k, obj, curr_node.left)
        else:
            self.__search(k, obj, curr_node.right)

        self.__insert_to_ret(curr_dis_node, k)

        temp = obj.copy()
        temp[ax] = curr_node.data[ax]

        if max(self.ret) > DisNode(temp, obj) or len(self.ret) < k:
            if to_left:
                self.__search(k, obj, curr_node.right)
            else:
                self.__search(k, obj, curr_node.left)


if __name__ == '__main__':

    # X = np.random.randint(low=0, high=10, size=(9, 2))
    X = np.array([
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ])

    obj = np.array([[-1., -5.]])

    X2 = np.array([
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
    tree = KdTree()
    tree.create_tree(X2)
    tree.search(3, obj[0])

    print(tree.ret)


    print('')

