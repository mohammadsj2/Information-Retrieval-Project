import typing
from sklearn.base import BaseEstimator, ClassifierMixin

# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
            max_iter=200, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, n_iter_no_change=10, max_fun=15000
    ):
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                                 learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, warm_start=warm_start,
                                 momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2,
                                 n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        pass

    def fit(self, x, y, **fit_params):
        self.mlp.fit(x,y)
        return self

    def predict(self, x):
        return self.mlp.predict(x)
