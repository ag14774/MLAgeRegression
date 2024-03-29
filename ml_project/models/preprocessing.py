import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.ndimage.filters import gaussian_filter


class RemoveZerosAndAlign(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self, orig_x, orig_y, orig_z, new_x, new_y, new_z):
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.orig_z = orig_z
        self.new_x = new_x
        self.new_y = new_y
        self.new_z = new_z

    def removeZeros(self, row, orig_x, orig_y, orig_z, axis=0):
        row = row.reshape((orig_x, orig_y, orig_z))
        row = np.swapaxes(row, 0, axis)
        firstNonZeroSide = -1
        lastNonZeroSide = -1
        for i, side in enumerate(row):
            if (side.max() != 0 and firstNonZeroSide == -1):
                firstNonZeroSide = i
            if (side.max() == 0 and firstNonZeroSide != -1
                    and lastNonZeroSide == -1):
                lastNonZeroSide = i - 1
                break
        oldcenter = round(
            (lastNonZeroSide - firstNonZeroSide + 1) / 2) + firstNonZeroSide
        newsize = [self.new_x, self.new_y, self.new_z]
        newcenter = round(newsize[axis] / 2)
        row = np.swapaxes(row, 0, axis)
        row = np.roll(row, newcenter - oldcenter, axis)
        row = np.swapaxes(row, 0, axis)
        row = row[0:newsize[axis]]
        row = np.swapaxes(row, 0, axis)
        row = row.reshape(-1)
        return row

    def removeZerosAllAxes(self, row):
        row = self.removeZeros(row, self.orig_x, self.orig_y, self.orig_z, 0)
        row = self.removeZeros(row, self.new_x, self.orig_y, self.orig_z, 1)
        row = self.removeZeros(row, self.new_x, self.new_y, self.orig_z, 2)
        return row

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = np.apply_along_axis(self.removeZerosAllAxes, 1, X)
        return X


class MinMaxBrightness(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.min_brightness = None
        self.max_brightness = None

    def fit(self, X, y=None):
        self.min_brightness = X.min()
        self.max_brightness = X.max()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["min_brightness", "max_brightness"])
        X = check_array(X)
        for i in range(0, len(X)):
            X[i] = minmax_scale(X[i], (self.min_brightness,
                                       self.max_brightness), 0, False)
        return X


class GaussianSmoothing(BaseEstimator, TransformerMixin):
    def __init__(self, orig_x, orig_y, orig_z, sigma=1.0):
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.orig_z = orig_z
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        for i in range(0, len(X)):
            row = np.reshape(X[i], (self.orig_x, self.orig_y, self.orig_z))
            row = gaussian_filter(row, self.sigma, mode='constant')
            X[i] = row.reshape(-1)
        return X


class RearrangeToCubicParts(BaseEstimator, TransformerMixin):
    """docstring"""

    def __init__(self,
                 cube_x=16,
                 cube_y=16,
                 cube_z=16,
                 orig_x=176,
                 orig_y=208,
                 orig_z=176):
        self.cube_x = cube_x
        self.cube_y = cube_y
        self.cube_z = cube_z
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.orig_z = orig_z

    def createTransformationMatrix(self, shape):
        index_map = np.arange(shape[0] * shape[1] * shape[2]).reshape(shape)
        matrix = np.zeros((shape[0] * shape[1] * shape[2]), dtype=np.int32)
        last = 0
        for i in range(0, shape[0], self.cube_x):
            for j in range(0, shape[1], self.cube_y):
                for k in range(0, shape[2], self.cube_z):
                    index = index_map[i:i + self.cube_x, j:j + self.cube_y, k:
                                      k + self.cube_z]
                    index = index.reshape(-1)
                    matrix[last:last + index.shape[0]] = index
                    last = last + index.shape[0]
        return matrix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        # X = X.reshape(-1, 176, 208, 176)
        # X = np.swapaxes(X, 1, 3)
        # X = X.reshape(X.shape[0], -1)
        indices = self.createTransformationMatrix((self.orig_x, self.orig_y,
                                                   self.orig_z))
        for i in range(0, X.shape[0]):
            X[i, :] = X[i, indices]
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
