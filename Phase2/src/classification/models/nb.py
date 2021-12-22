from sklearn.base import BaseEstimator, ClassifierMixin
from math import sqrt, pi, log, e
from statistics import median


def normal_f(x, mean, std):
    return e ** (-1 / 2 * (((x - mean) / std) ** 2)) / (std * sqrt(2 * pi))


def get_mean_std(x, dim):
    mean = []
    std = []

    n = len(x)

    for i in range(dim):
        chopped_x = [x[j][i] for j in range(n)]
        mean.append(sum(chopped_x) / n)

        tmp = sum([(x[j][i] - mean[-1]) ** 2 for j in range(n)]) / n
        std.append(sqrt(tmp))

    return mean, std


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            kind  #: th.Literal['gaussian', 'bernoulli', 'bernoulli_3_parts', 'bernoulli_median'],
            # add required hyper-parameters (if any)
    ):
        self.kind = kind
        self.mean = None
        self.std = None
        self.freq_prob = None
        self.thmin = self.thmax = self.th = self.p = None

    def get_p(self, X, dim):
        n = len(X)
        if self.kind == 'bernoulli':
            cnt = [0] * dim
            for x in X:
                for i in range(dim):
                    if x[i] > (self.thmin[i] + self.thmax[i]) / 2:
                        cnt[i] += 1
            return [x / n for x in cnt]
        elif self.kind == 'bernoulli_3_parts':
            cnt1 = [0.01] * dim
            cnt0 = [0.01] * dim

            for x in X:
                for i in range(dim):
                    if x[i] < (2 * self.thmin[i] + self.thmax[i]) / 3:
                        cnt0[i] += 1
                    elif x[i] < (self.thmin[i] + 2 * self.thmax[i]) / 3:
                        cnt1[i] += 1

            return [[x / (n + 0.03) for x in cnt0], [x / (n + 0.03) for x in cnt1]]

        elif self.kind == 'bernoulli_median':
            cnt = [0] * dim
            for x in X:
                for i in range(dim):
                    if x[i] > self.th[i]:
                        cnt[i] += 1
            return [x / n for x in cnt]

    def fit(self, x, y, **fit_params):
        # y is in {0,1}

        # x=lists of lists
        # y=lists of {0,1}
        n = len(x)
        dim = len(x[0])
        nx = [[], []]

        for i in range(n):
            nx[y[i]].append(x[i])

        self.freq_prob = [len(nx[i]) / n for i in range(2)]

        if self.kind == 'gaussian':
            self.mean = [[], []]
            self.std = [[], []]
            for i in range(2):
                self.mean[i], self.std[i] = get_mean_std(nx[i], dim)
        elif self.kind == 'bernoulli_median':
            self.th = []
            for i in range(dim):
                l = [x[j][i] for j in range(n)]
                self.th.append(median(l))

            self.p = []
            for i in range(2):
                self.p.append(self.get_p(nx[i], dim))
        else:
            self.thmin = []
            self.thmax = []
            for i in range(dim):
                l = [x[j][i] for j in range(n)]
                self.thmin.append(min(l))
                self.thmax.append(max(l))

            self.p = []
            for i in range(2):
                self.p.append(self.get_p(nx[i], dim))

        return self

    def predict(self, x):
        answer = []
        for single_x in x:
            tmp = self.predict_single_x(single_x)
            if tmp[1] > tmp[0]:
                answer.append(1)
            else:
                answer.append(0)
        return answer

    def predict_single_x(self, x):
        answer = [log(self.freq_prob[i]) for i in range(2)]
        dim = len(x)
        if self.kind == 'gaussian':
            for i in range(2):
                for j in range(dim):
                    answer[i] += log(normal_f(x[j], self.mean[i][j], self.std[i][j]))
        elif self.kind == 'bernoulli':
            for i in range(2):
                for j in range(dim):
                    if x[j] > (self.thmin[j] + self.thmax[j]) / 2:
                        answer[i] += log(self.p[i][j])
                    else:
                        answer[i] += log(1 - self.p[i][j])
        elif self.kind == 'bernoulli_3_parts':
            for i in range(2):
                for j in range(dim):
                    if x[j] < (2 * self.thmin[j] + self.thmax[j]) / 3:
                        answer[i] += log(self.p[i][0][j])
                    elif x[j] < (self.thmin[j] + 2 * self.thmax[j]) / 3:
                        answer[i] += log(self.p[i][1][j])
                    else:
                        answer[i] += log(1 - self.p[i][0][j] - self.p[i][1][j])
        else:
            for i in range(2):
                for j in range(dim):
                    if x[j] > self.th[j]:
                        answer[i] += log(self.p[i][j])
                    else:
                        answer[i] += log(1 - self.p[i][j])
        return answer
