import numpy as np
np.random.seed(666)


class Forward:
    def __init__(self):
        self.alpha = None

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

        return self

    @property
    def prob(self):
        ret = np.sum(self.alpha[-1, :])
        return ret


class Backward:
    def __init__(self):
        self.beta = None

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

    @property
    def prob(self):
        ret = np.sum(pi * B[:, O[0]] * self.beta[0, :])
        return ret


# Baum-Welch 算法
class Baumwelch:
    def __init__(self, n_status=3):
        '''初始化h'''
        self.n_status = n_status

        self.A = np.random.rand(n_status, n_status)
        self.A /= np.sum(self.A, axis=1)

        self.B = None

        self.pi = np.random.rand(n_status)
        self.pi /= np.sum(self.pi)

        self.forward = Forward()
        self.backward = Backward()

    def calc_gamma(self, alpha, beta, t, i):
        '''计算γ'''
        return alpha[t, i] * beta[t, i] / np.dot(alpha[t], beta[t])

    def calc_xi(self, alpha, beta, t, i, j, O):
        '''计算xi'''
        t1 = alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]
        t2 = 0

        for i in range(self.n_status):
            for j in range(self.n_status):
                t2 += alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]
        ret = t1 / t2
        return ret

    def update_a(self, i, j, O, T):
        '''更新矩阵A的i,j位置'''
        s1, s2 = 0, 0
        alpha, beta = self.forward.alpha, self.backward.beta

        for t in range(T-1):
            s1 += self.calc_xi(alpha, beta, t, i, j, O)
            s2 += self.calc_gamma(alpha, beta, t, i)

        ret = s1 / s2
        self.A[i, j] = ret

    def update_b(self, j, k, O, V, T):
        '''更新矩阵B的 j,K位置'''
        s1, s2 = 0, 0
        alpha, beta = self.forward.alpha, self.backward.beta

        for t in range(T):
            if O[t] == V[k]:
                s1 += self.calc_gamma(alpha, beta, t, j)
            s2 += self.calc_gamma(alpha, beta, t, j)

        ret = s1 / s2
        self.B[j, k] = ret

    def fit(self, O, e=100):
        '''训练'''
        V = np.unique(O)
        self.B = np.random.rand(self.n_status, len(V))
        self.B /= np.sum(self.B, axis=1).reshape(self.n_status, 1)

        T = len(O)

        while e > 0:
            e -= 1

            self.forward.fit(self.A, self.B, self.pi, O)
            self.backward.fit(self.A, self.B, self.pi, O)

            for i in range(self.n_status):
                for j in range(self.n_status):
                    self.update_a(i, j, O, T)

                for k in range(self.B.shape[1]):
                    self.update_b(i, k, O, V, T)

        return self


class Viterbi:
    def __init__(self):
        self.delta = None
        self.psi = None

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

        self.delta = np.zeros((T, N))
        self.psi = np.zeros((T, N))

        self.delta[0] = pi * B[:, O[0]]

        for t in range(1, T):
            for i in range(N):
                self.delta[t, i] = np.max((self.delta[t-1, :] * A[:, i]) * B[i, O[t]])
                self.psi[t, i] = np.argmax(self.delta[t-1, :] * A[:, i])

        return self

    @property
    def best_path(self):
        I = [np.argmax(self.delta[-1])]
        T = self.delta.shape[0]
        for t in range(T-2, -1, -1):
            i = int(I[-1])
            I.append(int(self.psi[t+1, i]))

        I.reverse()
        return I


if __name__ == '__main__':

    print(np.std([1, 5, 8, 12, 12]))

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

    bw = Baumwelch()
    bw.fit(O, e=500)
    print(bw.A)
    print(bw.B)
    print(bw.pi)

    fw.fit(bw.A, bw.B, bw.pi, O)
    print(fw.prob)

    vtb = Viterbi()
    vtb.fit(A, B, pi, O)
    print(vtb.best_path)



