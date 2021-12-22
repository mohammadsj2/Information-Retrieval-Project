import typing as th
from sklearn.base import ClusterMixin, BaseEstimator
# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.cluster import AgglomerativeClustering


class Hierarchical(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            cluster_count: int, affinity='euclidean',
            compute_full_tree='auto', linkage='ward', distance_threshold=None
    ):
        self.hc = AgglomerativeClustering(n_clusters=cluster_count, affinity=affinity,
                                          compute_full_tree=compute_full_tree, linkage=linkage,
                                          distance_threshold=distance_threshold)

    def fit_predict(self, x, **kwargs):
        self.hc.fit(x)
        return self.hc.labels_
