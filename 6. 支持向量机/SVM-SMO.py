import numpy as np


def kernel_linear(x1, x2):
    return np.dot(x1, x2)


class Loss:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, X, y, a, idx1, idx2):
        x1, x2 = X[idx1], X[idx2]
        y1, y2 = y[idx1], y[idx2]
        a1, a2 = a[idx1], a[idx2]

        idxi = np.arange(len(X))
        idxi = idxi[np.where((idxi != idx1) & (idxi != idx2))]
        xi, yi, ai = X[idxi], y[idxi], a[idxi]

        loss = 0.5 * self.kernel(x1, x1) * (a1**2) + \
               0.5 * self.kernel(x2, x2) * (a2 ** 2) + \
               y1 * y2 * self.kernel(x1, x2) * a1 * a2 - \
               (a1 + a2) + \
               y1 * a1 * np.sum(yi * ai * self.kernel(xi, x1)) + \
               y2 * a2 * np.sum(yi * ai * self.kernel(xi, x2))

        return loss


class CalcGx:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, X, y, a, xi, b):
        return np.sum(a * y * kernel_linear(X, xi)) + b


class CalcEi:
    def __call__(self, gi, yi):
        return gi - yi


class CalcA2Unc:
    def __call__(self, a2_old, y2, e1, e2, eta):
        return a2_old + y2 * (e1 - e2) / eta


class CalcA2:
    def __init__(self, c):
        self.c = c

    def __call__(self, a1_old, a2_old, a2_new_unc, y1, y2):
        if y1 == y2:
            l = max(0, a2_old - a1_old)
            h = min(self.c, a2_old - a1_old)
        else:
            l = max(0, a2_old + a1_old - self.c)
            h = min(self.c, a2_old + a1_old)

        if a2_new_unc > h:
            return h
        elif l <= a2_new_unc <= h:
            return a2_new_unc
        else:
            return l


class CalcA1:
    def __call__(self, a1_old, a2_old, a2_new, y1, y2, ):
        return a1_old + y1 * y2 * (a2_old - a2_new)


class CalcEta:
    def __call__(self, k11, k22, k12):
        return k11 + k22 - 2 * k12


class CalcB:
    def __call__(self, x1, x2, y1, y2, a1_old, a2_old, a1_new, a2_new, k1, k2, e, b_old):
        return -e - y1 * k1 * (a1_new - a1_old) - y2 * k2 * (a2_new - a2_old) + b_old


class SVM:
    def __init__(self, c=1, epsilon=0.02, kernel=kernel_linear):
        self.c = c
        self.epsilon = epsilon
        self.kernel = kernel
        self.CalcGx = CalcGx(self.kernel)
        self.CalcEi = CalcEi()
        self.CalcA2Unc = CalcA2Unc()
        self.CalcA2 = CalcA2(self.c)
        self.CalcA1 = CalcA1()
        self.CalcEta = CalcEta()

        self.X = None
        self.y = None
        self.b = 0
        self.a = None
        self.sv_idx = None
        self.mask_idx = None

    # 判断是否违反kkt条件
    def check_kkt(self, ai, yi, gi):
        if ai == 0:
            return yi * gi >= 1 - self.epsilon
        elif 0 < ai < self.c:
            return 1 - self.epsilon <= yi * gi <= 1 + self.epsilon
        elif ai == self.c:
            return yi * gi <= 1 + self.epsilon
        else:
            print('check_kkt error, ai={}'.format(ai))
            return None

    # 违反kkt条件的严重程度
    def score_violate_kkt(self, yi, gi):
        return np.abs(yi * gi - 1)

    def update_b(self, b1_new, b2_new, a1_new, a2_new):
        if 0 < a1_new < self.c:
            self.b = b1_new
        elif 0 < a2_new < self.c:
            self.b = b2_new
        else:
            self.b = (b1_new + b2_new) / 2

    def find_a1_idx(self):
        # 外层循环, 寻找a1
        a1_idx = 0
        a1_v_kkt_score = float("-inf")

        for i in range(len(self.X)):
            xi = self.X[i]
            yi = self.y[i]
            ai = self.a[i]
            gi = self.CalcGx(self.X, self.y, self.a, xi, self.b)

            if not self.check_kkt(ai, yi, gi):
                score = self.score_violate_kkt(yi,gi)
                if score > a1_v_kkt_score:
                    a1_idx = i
                    a1_v_kkt_score = score

        return a1_idx



if __name__ == '__main__':

    def test_loss():
        l = Loss(kernel_linear)
        size = 10
        X = np.random.rand(size, 2)
        y = np.zeros(size) + 1
        a = np.random.randn(size)
        idx1, idx2 = 2, 6
        r = l(X, y, a, idx1, idx2)
        print(kernel_linear(X, X[1]))
        print(r)

    test_loss()
