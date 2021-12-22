import typing
from sklearn.base import BaseEstimator, ClassifierMixin

# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.svm import SVC


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            c=1.0, kernel='rbf', degree=3, max_iter=-1,
            decision_function_shape='ovr', shrinking=True, gamma='scale',
            coef0=0, probability=False, tol=1e-3, cache_size=200,
            break_ties=False):
        # initialize parameters
        self.svc = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                       shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size,
                       max_iter=max_iter, decision_function_shape=decision_function_shape,
                       break_ties=break_ties)
        pass

    def fit(self, x, y, **fit_params):
        self.svc.fit(x, y)
        return self

    def predict(self, x):
        return self.svc.predict(x)
