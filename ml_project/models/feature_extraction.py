import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted


def runKmeans(Xrow, args):
    Xrow = np.reshape(Xrow, (-1, 1))
    Xrow = args["mbk"].fit_predict(Xrow)
    print(Xrow, Xrow.shape)
    sys.stdout.flush()
    return np.bincount(Xrow, minlength=args["minlength"])


class ShadeExtraction(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self, n_shades=10):
        self.n_shades = n_shades


#NEXT STEPS: USE KMEANS INSTEAD OF GAUSSIAN
#OR USE GAUSSIAN AND THEN KMEANS
#ALSO TRY SPREAD MATRIX WITH LOWER EPS

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        mbk = MiniBatchKMeans(
            n_clusters=self.n_shades,
            batch_size=1000,
            n_init=5,
            reassignment_ratio=0.02)
        X = np.apply_along_axis(runKmeans, 1, X,
                                {"mbk": mbk,
                                 "minlength": self.n_shades})
        print("X after bincount: ")
        print(X)
        sys.stdout.flush()
        X = np.array(X)

        return X
