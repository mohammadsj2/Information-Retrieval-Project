import typing as th
from sklearn.metrics import *


def accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def f1(y_true, y_pred, alpha: float = 0.5, beta: float = 1.):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)

    return 2 * p * r / (p + r)


def precision(y_true, y_pred) -> float:
    return precision_score(y_true=y_true, y_pred=y_pred)


def recall(y_true, y_pred) -> float:
    return recall_score(y_true=y_true, y_pred=y_pred)


evaluation_functions = dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)


def evaluate(y_true, y_pred) -> th.Dict[str, float]:
    """
    :param y_true: ground truth
    :param y_pred: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(y_true, y_pred) for name, func in evaluation_functions.items()}
