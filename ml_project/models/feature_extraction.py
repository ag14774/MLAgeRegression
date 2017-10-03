import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted


class ShadeExtraction(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self, n_shades=10, n_rows=3, random_state=None):
        self.n_shades = n_shades
        self.n_rows = n_rows
        self.random_state = random_state
        self.boundaries = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        sample_indices = sample_without_replacement(
            n_samples, self.n_rows, random_state=random_state)
        res = np.zeros((self.n_rows, self.n_shades))
        for idx, i in enumerate(sample_indices):
            X_subset = np.reshape(X[i], (-1, 1))
            temp = MiniBatchKMeans(
                n_clusters=self.n_shades, batch_size=10000).fit(X_subset)
            print(temp)
            sys.stdout.flush()
            # res[idx, :] = np.ravel(temp)

        res = np.sort(np.round(np.mean(res, axis=0)))
        self.boundaries = np.zeros(self.n_shades + 1)
        for i in range(1, self.n_shades):
            self.boundaries[i] = (res[i - 1] + res[i]) / 2
        self.boundaries[self.n_shades] = 999999999

        self.boundaries = np.round(self.boundaries)
        print(self.boundaries)
        sys.stdout.flush()

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["boundaries"])
        X = check_array(X)
        X = [
            np.histogram(row, bins=self.boundaries, density=False)[0]
            for row in X
        ]
        X = np.array(X)

        return X
