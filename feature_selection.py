from sklearn.feature_selection import (RFECV, VarianceThreshold,
                                       chi2, SelectKBest)
import pandas as pd
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from sklearn.svm import SVC
from IPython.display import display, HTML
from multiprocessing import Pool
from functools import partial
from help_functions import load_data, clean, makeXY, pd_dataframe, kfold
import numpy as np


#####################
# Feature selection methods.
#####################


def lasso(X_data, y_data, df, alpha=.06):
    mod = Lasso(alpha=alpha)
    mod.fit(X_data, y_data)
    return [b for a, b in zip(mod.coef_, df.columns) if a != 0]


def relief(_, _2, reli, df, param):
    # https://github.com/EpistasisLab/scikit-rebate
    return [df.columns[top] for top in reli.top_features_[:param]]


def variance_threshold(X_data, y_data, df, threshold=0.0):
    clf = VarianceThreshold(threshold)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a != False]


def select_k_best(X_data, y_data, df, score_func=chi2, k=20):
    clf = SelectKBest(score_func, k)
    mini = 0
    for x in range(0, len(X_data)):
        mini = min(min(X_data[x]), mini)
    if mini < 0:
        for x in range(0, len(X_data)):
            for y in range(0, len(X_data[x])):
                X_data[x][y] -= mini
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a != False]


def rfecv(X_data, y_data, df, estimator, step=1, cv=3):
    clf = RFECV(estimator, step, cv)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a != False]

#######################
# The actual feature selection code.
#######################


def smap(f):
    """Helping function for multiprocessing"""
    return f()


def feature_selection(X_train, y_train, df, processes=4, debug=False):
    """Executes the feature selection process with multiprocessing.
    Args:
      X_train (ndarray)
      y_train (ndarray)
      df (DataFrame)
      processes (Int): Number of processes used for multiprocessing
      debug (Bool)

    Returns:
      featurelists(List)
      """
    reli = ReliefF()
    reli.fit(X_train, y_train)
    rfecv_estimator = SVC(kernel="linear")
    if debug:
        functions = [
            partial(lasso, X_train, y_train, df, alpha=.05),
            partial(relief, X_train, y_train, reli, df, 40),
            partial(variance_threshold, X_train, y_train, df, threshold=1)]
    else:
        functions = []
        for alpha in [.05, 0.1]:  # Lasso
            functions.append(partial(lasso, X_train, y_train, df, alpha=alpha))
        for features in [40, 60, 80]:  # Relief
            functions.append(partial(relief, X_train, y_train, reli, df, features))
        for threshold in [1]:  # Variance Threshold
            functions.append(partial(variance_threshold, X_train, y_train, df, threshold=1))
        for k in [20]:  # Select K Best
            functions.append(partial(select_k_best, X_train, y_train, df, k=k))
        for stepsize in [1, 2, 3]:  # RFECV (Testing)
            functions.append(partial(rfecv, X_train, y_train, df, rfecv_estimator, step=stepsize))
    with Pool(processes) as pool:
        featurelists = pool.map(smap, functions)
    featurelists.append(df.columns)
    tmp = pd.DataFrame([[1 if f in featurelist else 0 for f in df.columns]
                        for featurelist in featurelists], columns=df.columns)
    display(HTML(tmp.loc[:, (tmp != 0).any(axis=0)].to_html()))
    return featurelists


if __name__ == "__main__":
    testsize = .3
    randseed = 42

    p, n = load_data(True)
    X, Y, df = pd_dataframe(p, n)
    folds = kfold(X, Y, n_splits=2, randseed=randseed)
    for X_train, X_test, y_train, y_test in folds:
        fl = feature_selection(X_train, y_train, df)
