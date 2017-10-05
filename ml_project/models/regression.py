import numpy as np
import pandas as pd
import sklearn as skl
from matplotlib import pyplot as plt
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""

    def __init__(self, save_path=None):
        super(KernelEstimator, self).__init__()
        self.save_path = save_path

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.y_mean = np.mean(y)
        y -= self.y_mean
        Xt = np.transpose(X)
        cov = np.dot(X, Xt)
        alpha, _, _, _ = np.linalg.lstsq(cov, y)
        self.coef_ = np.dot(Xt, alpha)

        if self.save_path is not None:
            plt.figure()
            plt.hist(
                self.coef_[np.where(self.coef_ != 0)], bins=50, normed=True)
            plt.savefig(self.save_path + "KernelEstimatorCoef.png")
            plt.close()

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "y_mean"])
        X = check_array(X)

        prediction = np.dot(X, self.coef_) + self.y_mean

        if self.save_path is not None:
            plt.figure()
            plt.plot(prediction, "o")
            plt.savefig(self.save_path + "KernelEstimatorPrediction.png")
            plt.close()

        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "KernelEstimatorScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "KernelEstimatorScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class AdaBoostRegressor(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""

    def __init__(self,
                 base_estimator='LinearRegression',
                 n_estimators=50,
                 learning_rate=1.0,
                 loss='linear',
                 random_state=None,
                 save_path=None):
        super(AdaBoostRegressor, self).__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.save_path = save_path
        self.abr = None

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        if (self.base_estimator == 'LinearRegression'):
            self.base_estimator = LinearRegression()
        elif (self.base_estimator == 'DecisionTreeRegressor'):
            self.base_estimator = DecisionTreeRegressor()
        else:
            raise Exception('Unsupported supported base_estimator')
        self.abr = ABR(self.base_estimator, self.n_estimators,
                       self.learning_rate, self.loss, self.random_state)
        self.abr.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        check_is_fitted(self, ["abr"])
        X = check_array(X)

        return self.abr.predict(X)

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "AdaBoostScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "AdaBoostScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path
