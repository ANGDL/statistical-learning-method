import numpy as np
np.random.seed(666)


class KernelLinear:
    def __call__(self, x1, x2):
        return np.dot(x1, x2)


class KernelPloy:
    def __init__(self, p):
        self.p = p

    def __call__(self, x1, x2):
        return (np.dot(x1, x2) + 1) ** self.p


class KernelGauss:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, x1, x2):
        return np.exp(- self.gamma * np.sum((x1 - x2) ** 2))


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


class CalcGxi:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, X, y, a, xi, b):
        return np.sum(a * y * self.kernel(X, xi)) + b


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
        if y1 != y2:
            l = max(0, a2_old - a1_old)
            h = min(self.c, self.c + a2_old - a1_old)
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
    def __call__(self, a1_old, a2_old, a2_new, y1, y2):
        return a1_old + y1 * y2 * (a2_old - a2_new)


class CalcEta:
    def __call__(self, k11, k22, k12):
        return k11 + k22 - 2 * k12


class CalcB:
    def __call__(self, x1, x2, y1, y2, a1_old, a2_old, a1_new, a2_new, k1, k2, e, b_old):
        return -e - y1 * k1 * (a1_new - a1_old) - y2 * k2 * (a2_new - a2_old) + b_old


class SVM:
    def __init__(self, c=1.0, epsilon=0.0, kernel='linear', max_iter=-1, p=2, gamma=0.01):
        self.c = c
        self.epsilon = epsilon

        if kernel == 'poly':
            self.kernel = KernelPloy(p)
        elif kernel == 'rbf':
            self.kernel = KernelGauss(gamma)
        else:
            self.kernel = KernelLinear()

        self.CalcGxi = CalcGxi(self.kernel)
        self.CalcEi = CalcEi()
        self.CalcA2Unc = CalcA2Unc()
        self.CalcA2 = CalcA2(self.c)
        self.CalcA1 = CalcA1()
        self.CalcEta = CalcEta()
        self.CalcB = CalcB()

        self.Loss = Loss(self.kernel)

        self.X = None
        self.y = None
        self.w = 0
        self.b = 0
        self.a = None
        self.e = None
        self.g = None
        self.sv_idx = []
        self.mask_idx2 = set()
        self.mask_idx1 = set()

        self.max_iter = max_iter

    # 判断是否违反kkt条件
    def check_kkt(self, ai, yi, gi):
        if ai == 0.0:
            return yi * gi >= 1 - self.epsilon
        elif 0 < ai < self.c:
            return 1 - self.epsilon <= yi * gi <= 1 + self.epsilon
        elif ai == self.c:
            return yi * gi <= 1 + self.epsilon
        else:
            raise Exception('ai={} out range of {}-{}'.format(ai, 0, self.c))

    # 违反kkt条件的严重程度
    def score_violate_kkt(self, ai, yi, gi):
        if ai == 0 and not (yi * gi >= 1 - self.epsilon):
            return np.abs(1 - self.epsilon - yi * gi)

        if 0 < ai < self.c and not (1 - self.epsilon <= yi * gi <= 1 + self.epsilon):
            return max(np.abs(1 - self.epsilon - yi * gi), np.abs(1 + self.epsilon - yi * gi))

        if ai == self.c and not(yi * gi <= 1 + self.epsilon):
            return np.abs(1 + self.epsilon - yi * gi)

        else:
            # print('ai={} does not violate kkt'.format(ai))
            return 0

    def update_b(self, b1_new, b2_new, a1_new, a2_new):
        if 0 < a1_new < self.c:
            self.b = b1_new
        elif 0 < a2_new < self.c:
            self.b = b2_new
        else:
            self.b = (b1_new + b2_new) / 2

    def find_a1_idx(self):
        '''外层循环, 寻找a1'''
        a1_idx = 0
        a1_v_kkt_score = float("-inf")

        for i in range(len(self.X)):
            yi = self.y[i]
            ai = self.a[i]
            gi = self.g[i]

            score = self.score_violate_kkt(ai, yi, gi)
            if score > a1_v_kkt_score and i not in self.mask_idx1:
                a1_idx = i
                a1_v_kkt_score = score

        self.mask_idx1.clear()
        return a1_idx

    def find_a2_idx(self, idx1):
        '''寻找e2'''
        e1 = self.e[idx1]
        sorted_e = np.argsort(self.e)

        if e1 > 0:
            for i in sorted_e:
                if self.e[i] not in self.mask_idx2 and i != idx1:
                    return i
        if e1 < 0:
            for i in sorted_e[::-1]:
                if self.e[i] not in self.mask_idx2 and i != idx1:
                    return i

        sorted_e = sorted_e[(sorted_e != idx1) & (sorted_e not in self.mask_idx2)]
        return sorted_e[-1]

    def find_a2_in_sv(self, idx1, idx2):
        self.mask_idx2.add(idx2)
        idx2_list = np.argwhere(self.a > 0)

        for idx in idx2_list:
            if idx[0] not in self.mask_idx2 and idx[0] != idx1 \
                    and self.check_kkt(self.a[idx], self.y[idx], self.g[idx]):
                return idx[0]

        # print("some thing error in a2!")
        return None

    def check_terminal(self):
        ret = True
        for ai, yi, gi in zip(self.a, self.y, self.g):
            if not 0 <= ai <= self.c or not self.check_kkt(ai, yi, gi):
                return False

        if np.sum(np.multiply(self.a, self.y)) != 0:
            return False

        return ret

    def get_w(self):
        ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
        '''
        alphas, dataset, labels = np.array(self.a), np.array(self.X), np.array(self.y)
        yx = labels.reshape(1, -1).T * np.array([1, 1]) * dataset
        w = np.dot(yx.T, alphas)

        return w

    def fit(self, X, y):
        assert len(X) == len(y), "X, y must have a same len"

        self.X = X
        self.y = y
        self.a = np.zeros(len(y), dtype=np.float)
        self.e = np.array(-self.y, dtype=np.float)
        self.g = np.zeros(len(y), dtype=np.float)

        it = 0
        loss = float('inf')
        a2_in_sv = False

        while True:
            # 满足停机条件，退出循环
            if self.check_terminal():
                break

            it += 1
            if self.max_iter != -1 and it > self.max_iter:
                break

            idx1 = self.find_a1_idx()  # 寻找a1的索引
            # 获取idx1对应的值
            x1 = self.X[idx1]
            y1 = self.y[idx1]
            a1_old = self.a[idx1]
            g1_old = self.g[idx1]
            e1_old = self.CalcEi(g1_old, y1)

            idx2 = self.find_a2_idx(idx1)  # 寻找a2的索引
            if a2_in_sv:  # 特殊条件，上次循环目标函数下降较小，支持向量点寻找
                idx2 = self.find_a2_in_sv(idx1, idx2)

            if idx2 is None:  # 如果找不到，放弃这个a1，继续寻找
                self.mask_idx1.add(idx1)
                a2_in_sv = False
                continue

            # 取出idx2对应的值
            x2 = self.X[idx2]
            y2 = self.y[idx2]

            a2_old = self.a[idx2]
            e2_old = self.e[idx2]

            # 计算核函数
            k11 = self.kernel(x1, x1)
            k22 = self.kernel(x2, x2)
            k12 = self.kernel(x1, x2)
            k21 = k12

            # 计算η
            eta = self.CalcEta(k11, k22, k12)
            if eta <= 0:
                self.mask_idx2.add(idx2)
                continue

            # 计算未裁剪的a2
            a2_new_unc = self.CalcA2Unc(a2_old, y2, e1_old, e2_old, eta)
            # 计算裁剪后的a2
            a2_new = self.CalcA2(a1_old, a2_old, a2_new_unc, y1, y2)
            # 使用新的a2更新a1
            a1_new = self.CalcA1(a1_old, a2_old, a2_new, y1, y2)

            # 如果新的a1不满足 0<=a1<=C，说明a2不合适
            if a1_new < 0.0 or a1_new > self.c:
                self.mask_idx2.add(idx2)
                continue

            # update
            self.a[idx1] = a1_new
            self.a[idx2] = a2_new

            # 计算新的b
            b1_new = self.CalcB(x1, x2, y1, y2, a1_old, a2_old, a1_new, a2_new, k11, k21, self.e[idx1], self.b)
            b2_new = self.CalcB(x1, x2, y1, y2, a1_old, a2_old, a1_new, a2_new, k12, k22, self.e[idx2], self.b)
            self.update_b(b1_new, b2_new, a1_new, a2_new)

            self.g[idx1] = self.CalcGxi(self.X, self.y, self.a, x1, self.b)
            self.g[idx2] = self.CalcGxi(self.X, self.y, self.a, x2, self.b)
            self.e[idx1] = self.CalcEi(self.g[idx1], self.y[idx1])
            self.e[idx2] = self.CalcEi(self.g[idx2], self.y[idx2])

            new_loss = self.Loss(self.X, self.y, self.a, idx1, idx2)  # 计算loss

            # 如果loss减低太小，则在支持向量点寻找a2
            if abs(new_loss - loss) < 1e-5:
                a2_in_sv = True
                self.mask_idx2.add(idx2)
            else:
                a2_in_sv = False
                self.mask_idx2.clear()

            loss = new_loss

        # 计算下w
        self.w = self.get_w()

        return self

    def predict(self, x):

        ret = []
        for xi in x:
            g = self.CalcGxi(self.X, self.y, self.a, xi, self.b)
            ret.append(g)

        ret = np.array(ret)
        ret[ret >= 0] = 1
        ret[ret < 0] = -1
        return ret


def plot_line(svm, X, y):
    b = svm.b
    x1, _ = max(X, key=lambda x: x[0])
    x2, _ = min(X, key=lambda x: x[0])
    a1, a2 = svm.get_w()
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])

    y1, y2 = (-b - 1 - a1 * x1) / a2, (-b - 1 - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2], 'b--')

    y1, y2 = (-b + 1 - a1 * x1) / a2, (-b + 1 - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2], 'b--')

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()


def plot_sk_svc_decision_boundary(model, axis, X, y):

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

    w = model.coef_[0]
    b = model.intercept_[0]

    # w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0] / w[1] * plot_x - b / w[1] + 1 / w[1]
    down_y = -w[0] / w[1] * plot_x - b / w[1] - 1 / w[1]

    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()


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
    import matplotlib.pylab as plt
    from sklearn.svm import LinearSVC

    # X = np.array([
    #     [3.393533211, 2.331273381],
    #     [3.110073483, 1.781539638],
    #     [1.343808831, 3.368360954],
    #     [3.582294042, 4.679179110],
    #     [2.280362439, 2.866990263],
    #     [7.423436942, 4.696522875],
    #     [5.745051997, 3.533989803],
    #     [9.172168622, 2.511101045],
    #     [7.792783481, 3.424088941],
    #     [7.939820817, 0.791637231]
    # ])
    # y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

    # my_svm = SVM(c=1, epsilon=0.09, max_iter=5000)
    # my_svm.fit(X, y)
    # plot_line(svm, X, y)
    #
    # clf = LinearSVC(C=0.5)
    # clf.fit(X, y)
    # plot_sk_svc_decision_boundary(clf, [0, 10, -5, 10], X, y)
    from sklearn import datasets
    X, y = datasets.make_circles(n_samples=50, random_state=888)
    y[y == 0] = -1
    # my_svm = SVM(c=10, epsilon=0.15, kernel='poly', max_iter=10000, p=7, sigma=1.3)
    my_svm = SVM(c=5, epsilon=0.01, kernel='rbf', max_iter=10000, gamma=0.11)
    my_svm.fit(X, y)
    plot_decision_boundary(my_svm, axis=[-3.5, 3.5, -2.5, 2.5])


    plt.scatter(X[y == -1, 0], X[y == -1, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.show()
