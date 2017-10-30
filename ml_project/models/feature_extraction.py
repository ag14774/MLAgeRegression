import sys
import math
import numbers
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      check_random_state)


def calcBinBoundaries(Xrow, args):
    Xrow = np.reshape(Xrow, (-1, 1))
    args["mbk"].fit(Xrow)
    centers = np.sort(args["mbk"].cluster_centers_, axis=0)
    print("Sample center", centers)
    sys.stdout.flush()
    return centers


def singleRowExtract(a,
                     bins=10,
                     cube_x=16,
                     cube_y=16,
                     cube_z=16,
                     max_range=4000):
    bin_num = 0
    if isinstance(bins, numbers.Number):
        bin_num = bins
    else:
        bin_num = len(bins) - 1
    if cube_x + cube_y + cube_z == 0:
        a, _ = np.histogram(a, bins, (1, max_range))
        a = np.nan_to_num(a, False)
    else:
        cubetotalsize = cube_x * cube_y * cube_z
        if cubetotalsize < bin_num:
            raise Exception(
                'cubetotalsize must be greater or equal to bin_num')
        hist_num = math.ceil(a.shape[0] / cubetotalsize)
        for i in range(0, hist_num):
            a[i * bin_num:(i + 1) * bin_num], _ = np.histogram(
                a[i * cubetotalsize:(i + 1) * cubetotalsize], bins,
                (1, max_range))
    # Total surface
    # a[a > threshold] = threshold
    # a[a < threshold] = 0
    # a = binary_erosion(a)
    # a = len(a[a == True])
    # result[bin_num * hist_num] = a
    return hist_num * bin_num


class ShadeExtraction(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self,
                 random_state=32,
                 n_shades=10,
                 n_rows=5,
                 max_range=None,
                 cube_x=16,
                 cube_y=16,
                 cube_z=16):
        self.random_state = random_state
        self.n_shades = n_shades
        self.n_rows = n_rows
        self.cube_x = cube_x
        self.cube_y = cube_y
        self.cube_z = cube_z
        self.max_range = max_range
        self.boundaries = None

    def fit(self, X, y=None):
        if (self.n_rows > 0):
            X = check_array(X)
            random_state = check_random_state(self.random_state)
            n_samples, n_features = X.shape
            mbk = MiniBatchKMeans(
                n_clusters=self.n_shades,
                batch_size=4000,
                n_init=5,
                random_state=random_state)
            sample_rows = sample_without_replacement(
                n_samples, self.n_rows, random_state=random_state)
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
            self.boundaries[self.n_shades] = self.max_range
            self.boundaries[0] = 1
            self.boundaries = np.sort(np.round(self.boundaries))
            self.boundaries = self.boundaries[np.nonzero(self.boundaries)]
            self.boundaries = np.unique(self.boundaries)
            self.boundaries = self.boundaries[self.boundaries <=
                                              self.max_range]
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
        if self.max_range is None:
            self.max_range = X.max()
        bin_num = 0
        if isinstance(bins, numbers.Number):
            bin_num = bins
        else:
            bin_num = len(bins) - 1
        cubetotalsize = self.cube_x * self.cube_y * self.cube_z
        hist_num = math.ceil(X.shape[1] / cubetotalsize)
        if cubetotalsize < bin_num:
            raise Exception(
                'cubetotalsize must be greater or equal to bin_num')
        for j in range(0, X.shape[0]):
            for i in range(0, hist_num):
                X[j, i * bin_num:(i + 1) * bin_num], _ = np.histogram(
                    X[j, i * cubetotalsize:(i + 1) * cubetotalsize], bins,
                    (1, self.max_range))
        X = X[:, 0:hist_num * bin_num]
        print("X after bincount: ")
        print(X)
        sys.stdout.flush()
        return X
