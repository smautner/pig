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


def score(X, y, clf_param, n_jobs, debug):
    clf, param = clf_param
    searcher = RSCV(clf, 
                param, 
                n_iter=50 if not debug else 5, 
                scoring="f1",
                n_jobs=n_jobs,
                iid=False,
                #fefit=True,
                cv=3,
                verbose=0,
                pre_dispatch="2*n_jobs",
                random_state=None,
                error_score=np.nan,
                return_train_score=False)
    searcher.fit(X, y)
    best_esti = searcher.best_estimator_
    best_score = searcher.best_score_
    return best_score, best_esti

def maketasks(featurelists, func_names, p, n, randseed, n_splits):
    """Creates tasks that can be used for random_param_search.
    Args:
      featurelists: A list of featurelists each list is a fold of featurelists.
      func_names: A list of strings, describing how the featurelists were made.
      p and n: The results from load_data().
      randseed: The seed used.
    """
    tasks = []
    for i in range(0, len(featurelists)): # Each list is a fold
        for j in range(0, len(featurelists[i])): # Each fold contains featurelists.
            FEATURELIST = featurelists[i][j]
            FUNCNAME = func_names[i][j]
            X,y,df = h.makeXY(FEATURELIST, p, n)
            X = StandardScaler().fit_transform(X)  # < Do I need this?
            tasks.extend([(i, X, y, cp, FEATURELIST, FUNCNAME) for cp in zip(rs.classifiers,rs.param_lists)])
    tasks = np.array(tasks)
    print(len(tasks)) # = 18 !
    tasks.dump("tmp/tasks")
    return tasks

def random_param_search(task, n_jobs=4, debug=False):
    """
    Args:
      task: A task made by maketasks().
      n_jobs: Number of parallel jobs used by score().
      debug: True if debug mode.
    """
    res = []
    best_esti = []
    best_score, best_esti = score(task[1], task[2], task[3], n_jobs, debug)
    return task[0], best_score, best_esti, task[4], task[5]


if __name__ == "__main__":
    from feature_selection import feature_selection
    debug = True
    randseed = 42

    p, n = h.load_data(debug)
    X, Y, df = h.pd_dataframe(p, n)
    folds = h.kfold(X, Y, n_splits=2, randseed=randseed)
    featurelists = []
    func_names = []
  
    for X_train, X_test, y_train, y_test in folds:
        fs = feature_selection(X_train, y_train, df, debug=debug)
        featurelists.append(fs[0])
        func_names.append(fs[1])

    maketasks(featurelists, p, n, randseed, n_splits=2)
    tasks = np.load("tmp/tasks", allow_pickle=True)
    for task in tasks:
        random_param_search(task, debug=debug)
        
# vor every fold: 
# use trainXY to get featurelists =>
# dann alle featurelists x classifiers (x2000) ;; dump score, classifier+params, feature_list (und wie generiert); (1)
# report (1) and score(best_esti, testXY) 
