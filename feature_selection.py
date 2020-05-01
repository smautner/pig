from sklearn.feature_selection import (RFECV, VarianceThreshold,
                                       chi2, SelectKBest)
import pandas as pd
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split  # tmp
from IPython.display import display, HTML
import pandas as pd
import copy
from multiprocessing import Pool
from functools import partial
import numpy as np

#########################
# Note: These functions are exactly the same as the ones in the Notebook.
#########################


def load_data():
    debug = True

    from sklearn.model_selection import train_test_split
    import loadfiles
    p, n = loadfiles.loaddata("data", numneg=3000 if not debug else 200, pos='1' if debug else 'both', seed=9)
    return p, n


def clean(di, oklist):
    for k in list(di.keys()):
        if k not in oklist:
            di.pop(k)
    return di


def makeXY(featurelist, p, n):
    asd = [clean(e, featurelist) for e in copy.deepcopy(p+n)]
    df = pd.DataFrame(asd)
    X = df.to_numpy()
    y = [1]*len(p)+[0]*len(n)
    return X, y, df


def pd_dataframe(p, n):
    allfeatures = list(p[1].keys())  # the filenames are the last one and we dont need that (for now)
    allfeatures.remove("name")
    X, Y, df = makeXY(allfeatures, p, n)
    return X, Y, df

###################
# Functions for train_test_split and KFold Cross Validation.
###################


def tts(X, y, test_size, randseed):
    """Train_test_split with the given data.
    Probably no longer neccessary but I didnt want to discard this yet."""
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize,
                                                        random_state=randseed)  # USE THE SAME SEED AS BELOW!
    return X_train, X_test, y_train, y_test


def kfold(X, y, n_splits=2, randseed=None, shuffle=True):
    """Applies KFold Cross Validation to the given data.
    Returns:
      folds (List): A list where each entry represents each fold with [X_train, X_test, y_train, y_test]
    """
    X = StandardScaler().fit_transform(X)
    folds = []
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    for train, test in kf.split(X):
        folds.append([X[train], X[test],
                      [y[i] for i in train], [y[i] for i in test]])
    return folds

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
    with Pool(processes) as p:
        featurelists = p.map(smap, functions)
    featurelists.append(df.columns)
    tmp = pd.DataFrame([[1 if f in featurelist else 0 for f in df.columns]
                        for featurelist in featurelists], columns=df.columns)
    display(HTML(tmp.loc[:, (tmp != 0).any(axis=0)].to_html()))
    return featurelists


if __name__ == "__main__":
    testsize = .3
    randseed = 42

    p, n = load_data()
    X, Y, df = pd_dataframe(p, n)
    folds = kfold(X, Y, n_splits=2, randseed=randseed)
    for X_train, X_test, y_train, y_test in folds:
        fl = feature_selection(X_train, y_train, df)
