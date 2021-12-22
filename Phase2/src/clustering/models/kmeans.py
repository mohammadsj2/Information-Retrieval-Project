from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from random import randrange
from math import sqrt


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(
            self,
            cluster_count: int,
            max_iteration: int,
            # add required hyper-parameters (if any)
    ):
        self.k = cluster_count
        self.max_iteration = max_iteration
        self.centers = None
        self.dis_type = 'euclidean'
        self.dim = None

    def get_distance(self, x, y):
        ans = 0
        if self.dis_type == 'euclidean':
            for i in range(self.dim):
                ans += (x[i] - y[i]) ** 2
            ans = sqrt(ans)
        return ans

    def fit(self, x_list):
        self.dim = len(x_list[0])
        centers = []
        for i in range(self.k):
            centers.append(x_list[randrange(len(x_list))])

        for ii in range(self.max_iteration):
            clusters = [[] for i in range((len(centers)))]
            for i in range(len(x_list)):
                x = x_list[i]

                # random number is useful in case of equal centers
                cid = min([[self.get_distance(x, centers[j]), j] for j in range(len(centers))])[1]
                clusters[cid].append(x)

            centers = []

            for cluster in clusters:
                if len(cluster) == 0:
                    continue
                else:
                    l=len(cluster)
                    new_center=[0]*self.dim
                    for c in cluster:
                        for i in range(self.dim):
                            new_center[i] += c[i]
                    for i in range(self.dim):
                        new_center[i] /= l
                    centers.append(new_center)

        self.centers = centers
        return self

    def predict(self, x_list):
        answer = []
        for x in x_list:
            cid = min([[self.get_distance(x, self.centers[j]), j] for j in range(len(self.centers))])[1]
            answer.append(cid)
        return answer
