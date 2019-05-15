import numpy as np
from collections import Counter

class Node:
    def __init__(self):
        self.sp_v = None
        self.sp_dim = None
        self.left = None
        self.right = None

class AdaBoostTree:
    def __init__(self, tol):
        self.tol = tol
        self.node_list = []

        self.a_list = []
        self.w = None
        
    def calc_loss(self, y_hat, y):
        idx = y_hat != y
        loss = np.sum(self.w[idx])  # 重点，loos是由权重w计算而来
        return loss

    def node_predict(self, X, y, sp_v, sp_dim):
        y_hat = np.empty_like(y)
        y_unique = np.unique(y)

        idx_l, idx_r = X[:, sp_dim] < sp_v, X[:, sp_dim] > sp_v

        sp_y_eq_l = False
        sp_y_eq_r = False

        cr = Counter(y[idx_r])

        if np.sum(idx_l) != 0:
            cl = Counter(y[idx_l])
            if cl.get(y_unique[0], -1) == cl.get(y_unique[1], -2):
                sp_y_eq_l = True

        if np.sum(idx_r) != 0:
            cr = Counter(y[idx_r])
            if cr.get(y_unique[0], -1) == cr.get(y_unique[1], -2):
                sp_y_eq_r = True

        if sp_y_eq_l and not sp_y_eq_r and np.sum(idx_r) != 0:
            y_hat[idx_r] = list(Counter(y[idx_r]).most_common(1)[0])[0]
            y_hat[idx_l] = -y_hat[idx_r][0]

        elif sp_y_eq_r and not sp_y_eq_l and np.sum(idx_l) != 0:
            y_hat[idx_l] = list(Counter(y[idx_l]).most_common(1)[0])[0]
            y_hat[idx_r] = -y_hat[idx_l][0]

        else:
            if np.sum(idx_r) != 0:
                y_hat[idx_r] = list(Counter(y[idx_r]).most_common(1)[0])[0]
            if np.sum(idx_l) != 0:
                y_hat[idx_l] = list(Counter(y[idx_l]).most_common(1)[0])[0]
        
        return y_hat

    def sign(self, y):
        y_hat = y[:]
        y_hat[y > 0] = 1
        y_hat[y < 0] = -1

        return y_hat

    def predict(self, X):
        y_hat = np.zeros(len(X))
        for a, node in zip(self.a_list, self.node_list):
            y_hat += a * self.node_predict(X, y, node.sp_v, node.sp_dim)

        y_hat = self.sign(y_hat)
        return y_hat

    def __build_tree(self, X, y):
        ''' 创建树 '''
        ndim = len(X[0])

        while True:
            loss_best = float('inf') 
            node = Node()
            # 寻找最佳的划分维度和值
            for sp_dim in range(ndim):
                sp_values = [(X[i, sp_dim] + X[i + 1, sp_dim]) / 2 for i in range(len(X)-1)]
                for sp_v in sp_values:
                    y_hat = self.node_predict(X, y, sp_v, sp_dim)
                    loss_t = self.calc_loss(y_hat, y)    
                    if loss_t < loss_best:
                        loss_best = loss_t
                        node.sp_v = sp_v
                        node.sp_dim = sp_dim

            # 计算该子树的预测值
            y_hat = self.node_predict(X, y, node.sp_v, node.sp_dim)
            node.left = y_hat[0]
            node.right = y_hat[-1]
            self.node_list.append(node)
            
            a = np.log((1 - loss_best) / loss_best) / 2
            self.a_list.append(a)

            z = self.w * np.exp(-a * y * y_hat)
            self.w = z / np.sum(z)

            print(self.w)

            y_hat = self.predict(X)
            if self.calc_loss(y_hat, y) <= self.tol:
                break
        
    def fit(self, X, y):
        assert len(X) == len(y), "error"
        assert X.ndim > 1
        self.w = np.ones(X.shape[0]) / len(X)
        self.__build_tree(X, y)
        return self

    
if __name__ == "__main__":
    X = np.arange(10).reshape(-1, 1)
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    ada_tree = AdaBoostTree(tol=1e-3)
    ada_tree.fit(X, y)
    y_hat = ada_tree.predict(X)
    print(y_hat)
