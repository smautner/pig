from sklearn.feature_selection import (RFECV, VarianceThreshold,
                                       chi2, SelectKBest)
from sklearn.linear_model import Lasso
from skrebate import ReliefF
from sklearn.svm import SVC
import other.help_functions as h
import numpy as np
#####################
# Feature selection methods.
#####################


def lasso(X_data, y_data, df, alpha=.06):
    mod = Lasso(alpha=alpha)
    mod.fit(X_data, y_data)
    return [b for a, b in zip(mod.coef_, df.columns) if a != 0]


def relief(X_data, y_data, df, param):
    reli = ReliefF()
    reli.fit(X_data, y_data)
    # https://github.com/EpistasisLab/scikit-rebate
    return [df.columns[top] for top in reli.top_features_[:param]]


def variance_threshold(X_data, y_data, df, threshold=0.0):
    clf = VarianceThreshold(threshold)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a != False]


def select_k_best(X_data, y_data, df, k=20):
    score_func=chi2

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


def rfecv(X_data, y_data, df, step=1, cv=3):
    rfecv_estimator = SVC(kernel="linear")

    clf = RFECV(rfecv_estimator, step=step, cv=cv)
    clf.fit(X_data, y_data)
    return [b for a, b in zip(clf.get_support(), df.columns) if a != False]

#######################
# The actual feature selection code.
#######################


def maketasks(folds, df, debug=False):
    """Creates the feature selection tasks"""

    tasks = []
    foldnr = 0
    for X_train, X_test, y_train, y_test in folds:
        FOLDXY = (X_train, X_test, y_train, y_test)
        # Each task: (i, type, FOLDXY, df, (args)
        if debug:
            tasks.extend([(foldnr, "Lasso", FOLDXY, df, .05),
                          (foldnr, "Relief", FOLDXY, df, 40),
                          (foldnr, "VarThresh", FOLDXY, df, 1)])

        else:
            for alpha in [.05, 0.1]:  # Lasso
                tasks.append((foldnr, "Lasso", FOLDXY, df, alpha))
##            for features in [40, 60, 80]:  # Relief
##                tasks.append((foldnr, "Relief", FOLDXY, df, features))
            for threshold in [.99, 1, 1.01]:  # Variance Threshold
                tasks.append((foldnr, "VarThresh", FOLDXY, df, threshold))
            for k in [20]:  # Select K Best
                tasks.append((foldnr, "SelKBest", FOLDXY, df, k))
##            for stepsize in [1, 2, 3]:  # RFECV
##                tasks.append((foldnr, "RFECV", FOLDXY, df, stepsize))
        foldnr += 1

    np.array(tasks, dtype=object).dump("tmp/fs_tasks")
    return tasks


def feature_selection(taskid):
    """Executes the feature selection using the given task.
    Args:
      taskid: An ID for a made from maketasks()

    Returns:
      featurelist(List)
      """
    tasks = np.load("tmp/fs_tasks", allow_pickle=True)
    foldnr, fstype, FOLDXY, df, args = tasks[taskid]
    X_train, X_test, y_train, y_test = FOLDXY
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
    else:
        raise ValueError(f"'{fstype}' is not a valid Feature selection method.")
    return foldnr, fl, f"{fstype}: {args}", FOLDXY
