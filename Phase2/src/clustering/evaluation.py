import typing as th
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score
import numpy as np


def purity(labels_true, labels_pred) -> float:
    tmp = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(tmp, axis=0)) / np.sum(tmp)


def adjusted_rand_index(labels_true, labels_pred) -> float:
    return adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)


evaluation_functions = dict(purity=purity, adjusted_rand_index=adjusted_rand_index)


def evaluate(labels_true, labels_pred) -> th.Dict[str, float]:
    """
    :param labels_true: ground truth
    :param labels_pred: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(labels_true, labels_pred) for name, func in evaluation_functions.items()}
