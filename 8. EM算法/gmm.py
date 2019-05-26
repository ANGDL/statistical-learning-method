import numpy as np
np.random.seed(666)


class GMM:
    def __init__(self, k=2, e=1e-3):
        self.k = k
        self.e = e
        self.y = None
        self.a = np.random.rand(k)
        self.a = self.a / np.sum(self.a)  # 满足条件a_k≥0， Σa_k=1
        self.mu = np.random.randn(self.k)
        self.sigma = np.random.randn(self.k)
        self.gamma = None

    def calc_q(self, gamma, gamma_hat):
        '''计算Q方程， 由于log(σ)中，σ可能是负数，所以这里还没明白'''
        q = 0
        for k in range(self.k):
            n_k = 0
            t = 0
            for r, r_hat, y_j in zip(gamma[:, k], gamma_hat[:, k], self.y):
                n_k += r
                t += r_hat * (np.log(1/np.sqrt(2*np.pi)) -
                              np.log(self.sigma[k]) - (y_j-self.mu[k])**2 / (2 * self.sigma[k]))

            q += (n_k * np.log(self.a[k]) + t)

        return q

    def calc_phi(self, y_j):
        '''计算高斯分布密度Φ, y_j是观察值'''
        return np.exp(-(y_j - self.mu)**2 / (2 * self.sigma**2)) / (np.sqrt(2 * np.pi * self.sigma**2))

    def __calc_gamma(self, y_j):
        '''计算γ'''
        phi_list = self.calc_phi(y_j)

        t = self.a * phi_list

        if np.sum(t) == 0:
            return 0.0

        return t / np.sum(t)

    def calc_gamma(self):
        '''计算γ'''
        gamma = np.zeros_like(self.gamma)
        for j in range(len(self.y)):
            gamma[j] = self.__calc_gamma(self.y[j])

        return gamma

    def update_mu(self):
        '''更新μ'''

        a, b = 0, 0
        for j, y_j in enumerate(self.y):
            a += self.gamma[j] * y_j
            b += self.gamma[j]

        ret = a / b
        self.mu = ret

    def update_sigma(self):
        '''更新σ, 便于理解写成循环'''
        for k in range(len(self.a)):
            a, b = 0, 0
            for j, y_j in enumerate(self.y):
                a += self.gamma[j][k] * (y_j - self.mu[k]) ** 2
                b += self.gamma[j][k]

            assert b != 0, "error in update_sigma"
            self.sigma[k] = np.sqrt(a / b)

    def update_alpha(self):
        '''更新α'''
        self.a = np.sum(self.gamma, axis=0) / len(self.y)

    def fit(self, y):
        self.y = y
        self.gamma = np.random.randn(len(y), len(self.a))
        self.gamma[:, 0][self.gamma[:, 0] < np.mean(self.gamma)] = 0
        self.gamma[:, 0][self.gamma[:, 0] >= np.mean(self.gamma)] = 1
        self.gamma[:, 1] = 1 - self.gamma[:, 0]

        # q_i = self.calc_q(self.gamma, self.gamma)

        while True:
            a = self.a[:]
            gamma_hat = self.calc_gamma()
            # q_i_1 = self.calc_q(self.gamma, gamma_hat)
            #
            # if abs(q_i - q_i_1) <= self.e:
            #     break

            # q_i = q_i_1
            self.gamma = gamma_hat
            self.update_mu()
            self.update_sigma()
            self.update_alpha()

            if np.sqrt(np.sum((a - self.a) ** 2)) < self.e:
                break

        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture as SKGMM

    mu = 0
    sigma = 2

    X1 = np.random.normal(0, 1, 100)
    X2 = np.random.normal(-10, 0.1, 100)
    X = np.append(X1, X2)
    np.random.shuffle(X)
    # plt.hist(X1)
    # plt.hist(X2)
    # plt.show()
    plt.hist(X)
    plt.show()

    gmm = GMM(k=2, e=1e-5)
    gmm.fit(X)

    print(gmm.mu)
    print(gmm.sigma)
    print("done")

    sk_gmm = SKGMM(n_components=2).fit(X.reshape(-1, 1))
    print(sk_gmm.means_)
    print(sk_gmm.covariances_)

