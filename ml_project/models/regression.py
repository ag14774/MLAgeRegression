import numpy as np
import pandas as pd
import sklearn as skl
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression)
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import BaggingRegressor
import sys


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


class MixedRegressor(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""

    def __init__(self, save_path=None):
        super(MixedRegressor, self).__init__()
        self.save_path = save_path
        self.regressor = None
        self.regressorlt40 = None
        self.regressorgt60 = None

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)

        self.regressor = BayesianRidge()
        self.regressorlt40 = BayesianRidge()
        self.regressorgt60 = BayesianRidge()

        self.regressor.fit(X, y)

        lt40 = y < 40
        gt60 = y > 60

        Xlt40 = X[lt40]
        Ylt40 = y[lt40]

        Xgt60 = X[gt60]
        Ygt60 = y[gt60]

        self.regressorlt40.fit(Xlt40, Ylt40)
        self.regressorgt60.fit(Xgt60, Ygt60)

        return self

    def predict(self, X):
        check_is_fitted(self, ["regressor", "regressorlt40", "regressorgt60"])
        X = check_array(X)
        predictions = self.regressor.predict(X)

        lt18 = predictions < 18
        gt88 = predictions > 88

        if(len(predictions[lt18]) > 0):
            predlt18 = self.regressorlt40.predict(X[lt18])
            predictions[lt18] = predlt18

        if(len(predictions[gt88]) > 0):
            predgt88 = self.regressorgt60.predict(X[gt88])
            predictions[gt88] = predgt88
        return predictions

    def score(self, X, y, sample_weight=None):
        scores = -(self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class Regressor(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""

    def __init__(self,
                 base_estimator='AdaBoostedLinearRegression',
                 n_estimators=50,
                 learning_rate=1.0,
                 loss='linear',
                 random_state=None,
                 save_path=None):
        super(Regressor, self).__init__()
        self.base_estimator = str(base_estimator)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.save_path = save_path
        self.regressor = None

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        if (self.base_estimator == 'BayesianRidge'):
            self.regressor = BayesianRidge()
        elif (self.base_estimator == 'LASSO'):
            self.regressor = Lasso()
        elif (self.base_estimator == 'ElasticNet'):
            self.regressor = ElasticNet()
        elif (self.base_estimator == 'MLPRegressor'):
            self.regressor = MLPRegressor()
        elif (self.base_estimator == 'KernelRidge'):
            self.regressor = KernelRidge(kernel='polynomial')
        elif (self.base_estimator == 'LinearRegression'):
            self.regressor = LinearRegression()
        elif (self.base_estimator == 'BaggingRegressorLinear'):
            base_estimator = LinearRegression()
            self.regressor = BaggingRegressor(base_estimator)
        elif (self.base_estimator == 'BaggingRegressorKernelRidge'):
            base_estimator = KernelRidge(kernel='polynomial')
            self.regressor = BaggingRegressor(base_estimator)
        elif (self.base_estimator == 'BaggingRegressorLasso'):
            base_estimator = Lasso()
            self.regressor = BaggingRegressor(base_estimator)
        else:
            raise Exception('Unsupported base_estimator: ' +
                            self.base_estimator)
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["regressor"])
        X = check_array(X)

        return self.regressor.predict(X)

    def score(self, X, y, sample_weight=None):
        scores = -(self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        print(score)
        sys.stdout.flush()

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path
