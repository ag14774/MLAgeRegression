import numpy as np
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import k_means
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement


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
            X_subset = X[i, :]
            X_subset = np.reshape(X_subset, (n_features, 1))
            temp = k_means(X_subset, self.n_shades)[0]
            res[idx, :] = np.ravel(temp)

        res = np.sort(np.round(np.mean(res, axis=0)))
        self.boundaries = np.zeros(self.n_shades + 1)
        for i in range(1, self.n_shades):
            self.boundaries[i] = (res[i - 1] + res[i]) / 2
        self.boundaries[self.n_shades] = 999999999
        self.boundaried[0] = 1

        self.boundaries = np.round(self.boundaries)
        print(self.boundaries)
        sys.stdout.flush()

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["boundaries"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X = np.appy_along_axis(self.np_histogram, 1, X,
                               {"bins": self.boundaries,
                                "density": False})

        return X

    def np_histogram(arr,
                     bins=10,
                     range=None,
                     normed=False,
                     weights=None,
                     density=None):
        return np.histogram(arr, bins, range, normed, weights, density)[0]
