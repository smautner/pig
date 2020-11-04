from sklearn.feature_selection import (RFECV, VarianceThreshold,
                                       chi2, SelectKBest, SelectFromModel)
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#####################
# Feature selection methods.
#####################

def svcl1(X_data, y_data, df, args):
    """
    Linear Support Vector Classification with L1 Regularization.
    """
    randseed, C = args
    clf = LinearSVC(penalty="l1", dual=False, random_state=randseed, C=C)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.coef_[0], df.columns) if a]

def svcl2(X_data, y_data, df, args):
    """
    Linear Support Vector Classification with L2 Regularization.
    """
    randseed, C = args
    clf = LinearSVC(penalty="l2", random_state=randseed, C=C)
    clf.fit(X_data, y_data)
    sel = SelectFromModel(clf, prefit=True)
    support = sel.get_support(True)
    return [b for a, b in zip(clf.coef_[0][support],  df.columns[support])]


def lasso(X_data, y_data, df, alpha):
    clf = Lasso(alpha=alpha)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.coef_, df.columns) if a]


def relief(X_data, y_data, df, param):
    clf = ReliefF()
    clf.fit(X_data, y_data)
    # https://github.com/EpistasisLab/scikit-rebate
    return [df.columns[top] for top in clf.top_features_[:param]]


def variance_threshold(X_data, y_data, df, threshold):
    clf = VarianceThreshold(threshold)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def select_k_best(X_data, y_data, df, k):
    score_func=chi2

    clf = SelectKBest(score_func, k=k)
    mini = 0
    for x in range(0, len(X_data)):
        mini = min(min(X_data[x]), mini)
    if mini < 0:
        for x in range(0, len(X_data)):
            for y in range(0, len(X_data[x])):
                X_data[x][y] -= mini
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def rfecv(X_data, y_data, df, step, cv):
    rfecv_estimator = SVC(kernel="linear")

    clf = RFECV(rfecv_estimator, step=step, min_features_to_select=20, cv=cv)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a]


def random_forest(X_data, y_data, df, args):
    randseed, max_features = args
    clf = RandomForestClassifier(max_features=max_features, random_state=randseed, n_jobs=1)
    clf.fit(X_data, y_data)
    sel = SelectFromModel(clf, max_features=max_features, prefit=True)
    support = sel.get_support(True)
    return [b for a, b in zip(clf.feature_importances_[support],  df.columns[support])]


def random(X_data, y_data, df, args):
    """
    Args:
      args: (num_features, seed)
    Make sure 1, 20001 40001 etc have same seed.
    """
    num_features, seed = args

    np.random.seed(seed)
    return np.random.choice(df.columns, num_features, replace=False).tolist()

#######################
# The actual feature selection code.
#######################


def feature_selection(foldxy, fstype, args, df):
    """Executes the feature selection using the given task.
    Args:
      task: A FS task made by maketasks() above
      foldxy: [X_train, X_test, y_train, y_test]
      df: The used dataframe

    Returns:
      featurelist(List)
      """
    X_train, X_test, y_train, y_test = foldxy
    if fstype == "Lasso":
        fl = lasso(X_train, y_train, df, args)
    elif fstype == "Relief":
        fl = relief(X_train, y_train, df, args)
    elif fstype == "VarThresh":
        fl = variance_threshold(X_train, y_train, df, args)
    elif fstype == "SelKBest":
        fl = select_k_best(X_train, y_train, df, args)
    elif fstype == "RFECV":
        fl = rfecv(X_train, y_train, df, args)
    elif fstype == "SVC1":
        fl = svcl1(X_train, y_train, df, args)
    elif fstype == "SVC2":
        fl = svcl2(X_train, y_train, df, args)
    elif fstype == "Forest":
        fl = random_forest(X_train, y_train, df, args)
    elif fstype == "Random":
        fl = random(X_train, y_train, df, args)
    else:
        raise ValueError(f"'{fstype}' is not a valid Feature selection method.")
    mask = [True if f in fl else False for f in df.columns]
    return fl, mask, f"{fstype}: {args}"
