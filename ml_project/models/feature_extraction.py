import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted


def calcBinBoundaries(Xrow, args):
    Xrow = np.reshape(Xrow, (-1, 1))
    args["mbk"].fit(Xrow)
    print("Sample center", args["mbk"].cluster_centers_)
    sys.stdout.flush()
    return np.sort(args["mbk"].cluster_centers_)


class ShadeExtraction(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self, n_shades=10, n_rows=5, density=False):
        self.n_shades = n_shades
        self.n_rows = n_rows
        self.density = density
        self.boundaries = None


#NEXT STEPS: USE KMEANS INSTEAD OF GAUSSIAN
#OR USE GAUSSIAN AND THEN KMEANS
#ALSO TRY SPREAD MATRIX WITH LOWER EPS

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        mbk = MiniBatchKMeans(
            n_clusters=self.n_shades,
            batch_size=1000,
            n_init=5,
            reassignment_ratio=0.02)
        sample_rows = sample_without_replacement(n_features, self.n_rows)
        X_subset = X[sample_rows]
        X_subset = np.apply_along_axis(calcBinBoundaries, 1, X_subset,
                                       {"mbk": mbk})
        averaged_centers = np.mean(X_subset, axis=0)
        print("Averaged centers", averaged_centers)
        sys.stdout.flush()
        self.boundaries = np.zeros(self.n_shades + 1)
        for i in range(1, self.n_shades):
            self.boundaries[i] = (
                averaged_centers[i - 1] + averaged_centers[i]) / 2
        self.boundaries[self.n_shades] = 99999999999
        self.boundaries = np.round(self.boundaries)
        print("Final boundaries: ", self.boundaries)
        sys.stdout.flush()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["boundaries"])
        print("Shape before bin count: ", X.shape)
        sys.stdout.flush()
        X = check_array(X)
        X = np.apply_along_axis(
            np.histogram, 1, X, bins=self.boundaries, density=self.density)
        print("X after bincount: ")
        print(X)
        sys.stdout.flush()
        X = np.array(X)

        return X
