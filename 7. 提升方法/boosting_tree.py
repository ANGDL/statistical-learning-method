import numpy as np


class Node:
    '''结点类型'''
    def __init__(self):
        self.left = None  # 左预测值
        self.right = None  # 右预测值
        self.sp_v = None   # 划分值
        self.sp_dim = None  # 划分维度


class BoostingTree:
    '''提升回归树'''
    def __init__(self, e):
        ''' 初始化 '''
        self.tree_list = []  # 存放弱子树 
        self.e = e   # 误差停止条件

    def calc_loss(self, y_hat, y):
        ''' 计算loss '''
        return np.sum((y - y_hat) ** 2)

    def node_predict(self, X, y, sp_v, sp_dim):
        '''针对子树预测'''
        y_hat = np.zeros_like(y)
        idx_l, idx_r = X[:, sp_dim] < sp_v, X[:, sp_dim] > sp_v
        y_hat[idx_l] = np.mean(y[idx_l])
        y_hat[idx_r] = np.mean(y[idx_r])

        return y_hat

    def predict(self, X):
        ''' 整体预测 '''
        y_hat = np.zeros(len(X))
        for tree in self.tree_list:
            sp_dim = tree.sp_dim
            sp_v =tree.sp_v
            idx_l = X[:, sp_dim] < sp_v
            idx_r = X[:, sp_dim] > sp_v
            y_hat[idx_l] += tree.left
            y_hat[idx_r] += tree.right
        
        return y_hat

    def __build_tree(self, X, y):
        ''' 创建树 '''
        ndim = len(X[0])

        # 初始化残差为y
        r = y[:]
        while True:
            loss_best = float('inf')
            node = Node()

            # 寻找最佳的划分维度和值
            for sp_dim in range(ndim):
                sp_values = [(X[i, sp_dim] + X[i + 1, sp_dim]) / 2 for i in range(len(X)-1)]
                for sp_v in sp_values:
                    r_hat = self.node_predict(X, r, sp_v, sp_dim)
                    loss_t = self.calc_loss(r_hat, r)    
                    if loss_t < loss_best:
                        loss_best = loss_t
                        node.sp_v = sp_v
                        node.sp_dim = sp_dim

            idx_l = X[:, node.sp_dim] < node.sp_v
            idx_r = X[:, node.sp_dim] > node.sp_v

            X_l, X_r = X[idx_l], X[idx_r]
            r_l, r_r = r[idx_l], r[idx_r]

            # 计算该子树的预测值
            node.left = self.node_predict(X_l, r_l, node.sp_v, node.sp_dim)
            node.right = self.node_predict(X_r, r_r, node.sp_v, node.sp_dim)
            
            # 添加子树
            self.tree_list.append(node)
            
            # 整体做一次预测
            y_hat = self.predict(X)
            # 计算整体loss
            loss = self.calc_loss(y_hat, y)
            if loss <= self.e:
                return 

            # 更新残差
            r  = y - y_hat

    def fit(self, X, y):
        assert len(X) == len(y), "error"
        self.__build_tree(X, y)
        return self


if __name__ == "__main__":
    X = np.arange(1, 11).reshape(-1, 1)
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

    bt = BoostingTree(e=0.17)
    bt.fit(X, y)

    print(bt.predict(X))