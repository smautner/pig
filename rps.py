import numpy as np
from numpy import reshape
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV as RSCV
from multiprocessing import Pool
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from functools import partial
import other.randomsearch as  rs
import other.help_functions as h


def score(xy, clf_param, n_jobs, debug):
    X_train, X_test, y_train, y_test = xy
    clf, param = clf_param
    searcher = RSCV(clf, 
                param, 
                n_iter=50 if not debug else 5, 
                scoring=None,
                n_jobs=n_jobs,
                iid=False,
                #fefit=True,
                cv=3,
                verbose=0,
                pre_dispatch="2*n_jobs",
                random_state=None,
                error_score=np.nan,
                return_train_score=False)
    searcher.fit(X_train, y_train)
    best_esti = searcher.best_estimator_
    #print(searcher.best_params_)
    return h.scorer(best_esti,X_test,np.array(y_test)), best_esti

def maketasks(featurelists, p, n, randseed, n_splits):
    """Creates tasks that can be used for random_param_search.
    Args:
      featurelists: A list of featurelists.
      p and n: The results from load_data().
      randseed: The seed used.
      n_splits: The number of splits made by each of the featurelists.
    """
    splits = []
    for FEATURELIST in featurelists:  # loop over all the selectors 
        # make some data 
        X,y,df = h.makeXY(FEATURELIST, p, n)
        X = StandardScaler().fit_transform(X)
        splits += h.kfold(X, y, n_splits, randseed, True)
    tasks = [(xy, cp) for xy in splits for cp in zip(rs.classifiers,rs.param_lists)]
    tasks = np.array(tasks)
    tasks.dump("tasks")
    return tasks

def random_param_search(task, n_jobs=4, debug=False):
    """
    Args:
      tasks: A list with tasks made by maketasks().
      processes: Number of parallel processes in the multiprocessing pool.
      debug: True if debug mode.
    """
    res = []
    best_esti = []
    res, best_esti = score(task[0], task[1], n_jobs, debug)
    return best_esti


if __name__ == "__main__":
    from feature_selection import feature_selection as fs
    debug = True
    randseed = 42

    p, n = h.load_data(debug)
    X, Y, df = h.pd_dataframe(p, n)
    folds = h.kfold(X, Y, n_splits=2, randseed=randseed)
    featurelists = []
    for X_train, X_test, y_train, y_test in folds:
        featurelists += fs(X_train, y_train, df, debug=debug)

    tasks = maketasks(featurelists, p, n, randseed, n_splits=2)
    tasks = np.load("tasks", allow_pickle=True)
    for task in tasks:
        random_param_search(task, debug=debug)
