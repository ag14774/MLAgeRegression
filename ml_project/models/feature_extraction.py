import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.stats import binned_statistic


def calcBinBoundaries(Xrow, args):
    Xrow = np.reshape(Xrow, (-1, 1))
    args["mbk"].fit(Xrow)
    centers = np.sort(args["mbk"].cluster_centers_, axis=0)
    print("Sample center", centers)
    sys.stdout.flush()
    return centers


def doHistStartFromOne(a, bins=10, statistic='count'):
    a = binned_statistic(a, a, statistic, bins, (1, a.max())).statistic
    a = np.nan_to_num(a, False)
    return a


class ShadeExtraction(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self, n_shades=10, n_rows=5, statistic='count'):
        self.n_shades = n_shades
        self.n_rows = n_rows
        self.statistic = statistic
        self.boundaries = None

    def fit(self, X, y=None):
        if (self.n_rows > 0):
            X = check_array(X)
            n_samples, n_features = X.shape
            mbk = MiniBatchKMeans(
                n_clusters=self.n_shades,
                batch_size=1000,
                n_init=5,
                reassignment_ratio=0.02)
            sample_rows = sample_without_replacement(n_samples, self.n_rows)
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
            self.boundaries[0] = 1
            self.boundaries = np.sort(np.round(self.boundaries))
            self.boundaries = self.boundaries[np.nonzero(self.boundaries)]
            print("Final boundaries: ", self.boundaries)
            sys.stdout.flush()
        return self

    def transform(self, X, y=None):
        if (self.n_rows > 0):
            check_is_fitted(self, ["boundaries"])
            bins = self.boundaries
        else:
            bins = self.n_shades
        print("Shape before bin count: ", X.shape)
        sys.stdout.flush()
        X = check_array(X)
        X = np.apply_along_axis(
            doHistStartFromOne, 1, X, bins=bins, statistic=self.statistic)
        print("X after bincount: ")
        print(X)
        sys.stdout.flush()
        X = np.array(X)
        return X
