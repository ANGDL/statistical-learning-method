import numpy as np


class Forward:
    def __init__(self):
        self.alpha = None
        self.ret = None

    def fit(self, A, B, pi, O):
        '''
        :param A: 转移概率矩阵
        :param B: 观测概率矩阵
        :param pi: 初始状态分布
        :param O: 观测序列
        :return:
        '''
        N = A.shape[0]  # 隐状态数量
        T = len(O)  # 序列长度
        assert N == A.shape[1], 'status transition matrix shape error'
        assert N == B.shape[0], \
            'shape_0 is not accordance between observation matrix and status transition matrix '
        assert N == len(pi), \
            'shape_0 is not accordance between observation matrix and  initial status probability sequence '

        self.alpha = np.zeros((T, N))

        # step1 计算初值
        for i in range(len(pi)):
            self.alpha[0, i] = pi[i] * B[i, O[0]]

        for t in range(1, N):
            for i in range(N):
                self.alpha[t, i] = np.dot(self.alpha[t-1, :], A[:, i]) * B[i, O[t]]  # B.shape[1]观测值数量

        self.ret = np.sum(self.alpha[-1, :])

        return self

    @property
    def prob(self):
        return self.ret


class Backward:
    def __init__(self):
        self.beta = None
        self.ret = None

    def fit(self, A, B, pi, O):
        '''
        :param A: 转移概率矩阵
        :param B: 观测概率矩阵
        :param pi: 初始状态分布
        :param O: 观测序列
        :return:
        '''
        N = A.shape[0]  # 隐状态数量
        T = len(O)  # 序列长度
        assert N == A.shape[1], 'status transition matrix shape error'
        assert N == B.shape[0], \
            'shape_0 is not accordance between observation matrix and status transition matrix '
        assert N == len(pi), \
            'shape_0 is not accordance between observation matrix and  initial status probability sequence '

        self.beta = np.zeros((T, N))
        self.beta[-1, :] = 1

        for t in range(T-2, -1, -1):
            for i in range(N):
                for j in range(N):
                    self.beta[t, i] += A[i, j] * B[j, O[t+1]] * self.beta[t+1, j]

        self.ret = np.sum(pi * B[:, O[0]] * self.beta[0, :])

    @property
    def prob(self):
        return self.ret


if __name__ == '__main__':

    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])

    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])

    pi = np.array([0.2, 0.4, 0.4])

    O = np.array([0, 1, 0])

    fw = Forward()
    fw.fit(A, B, pi, O)
    print(fw.prob)

    bw = Backward()
    bw.fit(A, B, pi, O)
    print(bw.prob)





