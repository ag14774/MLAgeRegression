from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (SelectKBest, f_regression,
                                       mutual_info_regression)
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


class SelectKBest2(BaseEstimator, TransformerMixin):
    """docstring"""
    def __init__(self, score_func="f_regression", k=10):
        self.score_func = str(score_func)
        self.k = k
        self.skb = None

    def fit(self, X, y=None):
        if self.score_func == "f_regression":
            self.skb = SelectKBest(f_regression, self.k)
        elif self.score_func == "mutual_info_regression":
            self.skb = SelectKBest(mutual_info_regression, self.k)
        self.skb.fit(X, y)
        return self

    def transform(self, X, y=None):
        X = self.skb.transform(X)
        return X
