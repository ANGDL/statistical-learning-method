import numpy as np
np.random.seed(000)


def median(X, ax=0):
    assert ax < X.ndim, "ax error"

    median_index = int(0.5 * len(X))
    sorted_index = np.argsort(X[:, ax])
    sorted_X = X[sorted_index]
    print(sorted_X)

    return sorted_X[median_index], sorted_X[:median_index], sorted_X[median_index+1:]


class Node:
    def __init__(self, x, split_ax, left=None, right=None):
        self.x = x
        self.split_ax = split_ax
        self.left = left
        self.right = right

    def __repr__(self):
        return tuple(self.x)

    def __str__(self):
        return str(self.x)


class KdTree:
    def __init__(self):
        self.root = None

    def create_tree(self, X):
        self.root = self.__create_tree(X, 0)

    def __create_tree(self, X, ax):
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

    tree = KdTree()
    tree.create_tree(X)

    print('')

