import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import sys


class IncrementalPCAInChunks(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self,
                 row_chunk=50,
                 n_components=None,
                 whiten=False,
                 copy=True,
                 batch_size=None):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.row_chunk = row_chunk

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        ipca = IncrementalPCA(self.n_components, self.whiten, self.copy,
                              self.batch_size)

        for i in range(0, n_samples // self.row_chunk):
            ipca.partial_fit(X[i * self.row_chunk:(i + 1) * self.row_chunk])
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        # TODO: FINISH
        return X


class Flatten(BaseEstimator, TransformerMixin):
    """Flatten"""

    def __init__(self, dim=2):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)  # Bad practice: hard-coded dimensions
        X = X.mean(axis=self.dim)
        return X.reshape(X.shape[0], -1)


class StandardScalerTranspose(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True, with_std=True):
        self.enabled = enabled
        self.with_std = with_std
        self.scaler = None

    def fit(self, X, y=None):
        if (self.enabled):
            self.scaler = StandardScaler(copy=False, with_std=self.with_std)
            self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        if (self.enabled):
            X = check_array(X)
            X = self.scaler.transform(X)
        return X
