import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class RemoveZeros(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = np.array([X[i, np.nonzero(X[i, :])] for i in range(X.shape[0])])
        print(X)
        print(X.shape)
        return X
