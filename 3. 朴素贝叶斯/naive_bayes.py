class MultinomialNB:
    '''
    适用于多项分布
    预测过程输入的特征值在训练集中必须见过
    '''
    def __init__(self, lambda_=1):
        self.y_counter = {}  # I(y_i = c_k)
        self.sj = {}  # S_j, 不同特征维度下的特征值统计
        self.n_simples = 0  # N
        self.joint_xy_counter = {}  # I(x_i_j = a_jl, yi = c_k)
        self.lambda_ = lambda_   # 贝叶斯估计的λ

    def __counter(self, counter, key):
        '''统计key个数，保存到counter里'''
        if key not in counter:
            counter[key] = 1
        else:
            counter[key] += 1

    def fit(self, X, y):
        '''训练函数， 求最大似然估计的过程'''
        assert len(X) == len(y), "fist of X dim must match with dim!"
        self.sj = {j: set() for j in range(len(X[0]))}

        for x_i, y_i in zip(X, y):
            self.__counter(self.y_counter, y_i)
            self.n_simples += 1
            # j 是x的特征维度
            for j, x_i_j in enumerate(x_i):
                self.sj[j].add(x_i_j)
                self.__counter(self.joint_xy_counter, (j, x_i_j, y_i))

        return self

    def predict(self, X):
        '''预测'''
        assert len(X[0]) == len(self.sj), 'X dim error'

        pred_list = []
        prob_values = [-1.0] * len(self.sj)

        sj = 0

        for i, x_i in enumerate(X, 1):
            prob = dict(zip(self.y_counter.keys(), prob_values))
            for y_k in self.y_counter:
                for j, x_i_j in enumerate(x_i):
                    if self.lambda_ != 0:
                        sj = len(self.sj[j])
                    try:
                        p_xij_yk = (self.joint_xy_counter[(j, x_i_j, y_k)] + self.lambda_) / \
                                   (self.y_counter[y_k] + sj * self.lambda_)
                        prob[y_k] *= p_xij_yk
                    except KeyError:
                        # print('Unknown feature: {} in {} dim of simple {}.'.format(x_i_j, j, i))
                        prob[y_k] *= -2

                p_yk = (self.y_counter[y_k] + self.lambda_) / (self.n_simples + len(self.y_counter) * self.lambda_)
                prob[y_k] *= -p_yk

            pred_list.append(max(prob, key=prob.get))

        return pred_list


if __name__ == '__main__':
    X = [
        [1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
        [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
        [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']
    ]

    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

    assert len(X) == len(y)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X, y)

    ret = nb_classifier.predict(X)

    print(ret)

    # import numpy as np
    #
    # X = np.random.randint(5, size=(60, 100))
    # y = np.array([1, 2, 3, 4, 5, 6]).repeat(10)
    # nb2 = MultinomialNB()
    # nb2.fit(X, y)
    # ret = nb2.predict(X[1:3])
    # print(ret)
