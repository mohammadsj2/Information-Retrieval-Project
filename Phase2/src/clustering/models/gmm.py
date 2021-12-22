from abc import ABCMeta
from sklearn.base import DensityMixin, BaseEstimator

# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.mixture import GaussianMixture


class GMM(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(
            self,
            cluster_count: int,
            max_iteration=100, covariance_type='full', init_params='kmeans',
            warm_start=False):
        self.gm = GaussianMixture(n_components=cluster_count,
                                  covariance_type=covariance_type,
                                  max_iter=max_iteration,
                                  init_params=init_params,
                                  warm_start=warm_start)
        pass

    def fit(self, x):
        self.gm.fit(x)
        return self

    def predict(self, x):
        return self.gm.predict(x)
