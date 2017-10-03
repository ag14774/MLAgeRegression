import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted


def runKmeans(Xrow, mbk):
    Xrow = np.reshape(Xrow, (-1, 1))
    return mbk["mbk"].fit_predict(Xrow)


class ShadeExtraction(BaseEstimator, TransformerMixin):
    """Count how many pixels fall in each shade category"""

    def __init__(self, n_shades=10, n_rows=3, random_state=None):
        self.n_shades = n_shades
        self.n_rows = n_rows
        self.random_state = random_state
        self.boundaries = None


#NEXT STEPS: USE KMEANS INSTEAD OF GAUSSIAN
#OR USE GAUSSIAN AND THEN KMEANS
#ALSO TRY SPREAD MATRIX WITH LOWER EPS

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["boundaries"])
        X = check_array(X)
        mbk = MiniBatchKMeans(n_clusters=self.n_shades, batch_size=1000)
        X = np.apply_along_axis(runKmeans, 1, X, {"mbk": mbk})
        print(X.shape)
        X = np.apply_along_axis(np.bincount, 1, X,
                                {"minlength": self.n_shades})
        print(X)
        sys.stdout.flush()
        X = np.array(X)

        return X
