from sklearn.base import BaseEstimator, ClassifierMixin
from math import *
import random


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            k: int,
            max_train_size: int,
            dis_type  # in : ['euclidian','dot'
            # add required hyper-parameters (if any)
    ):
        self.y = self.x = None
        self.k = k
        self.dis_type = dis_type
        self.max_train_size = max_train_size
        self.dim = None

    def get_distance(self, x, y):
        ans = 0
        if self.dis_type == 'euclidian':
            for i in range(self.dim):
                ans += (x[i] - y[i]) ** 2
        if self.dis_type == 'dot':
            lx = 0
            ly = 0
            for i in range(self.dim):
                lx += x[i] ** 2
                ly += y[i] ** 2
                ans += (x[i] * y[i])
            ans = ans / sqrt(lx * ly)
        return ans

    def fit(self, x, y, **fit_params):
        self.dim = len(x[0])
        if self.max_train_size == -1:
            self.y = y
            print(sum(y) / len(y))
            self.x = x
        else:
            l = [(x[i], y[i]) for i in range(len(x))]
            random.Random(4).shuffle(l)
            l = l[:self.max_train_size]

            self.y = [x[1] for x in l]
            self.x = [x[0] for x in l]
        return self

    def predict(self, x):
        answer = []
        for single_x in x:
            answer.append(self.predict_single_x(single_x))

        return answer

    def predict_single_x(self, x):
        tmp = []
        for i in range(len(self.x)):
            dis = self.get_distance(x, self.x[i])
            if len(tmp) == self.k and dis > tmp[-1][0]:
                continue
            tmp.append([dis, self.y[i]])
            tmp = sorted(tmp)[:self.k]

        # print(tmp)
        tmp = [x[1] for x in tmp]

        cnt1 = sum(tmp)
        cnt0 = len(tmp) - cnt1
        if cnt0 >= cnt1:
            return 0
        return 1
