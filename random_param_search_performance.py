import numpy as np
from numpy import reshape
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV as RSCV
from multiprocessing import Pool
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from functools import partial
import other.randomsearch as  rs
from other.help_functions import load_data, makeXY, pd_dataframe, scorer, kfold


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
    return scorer(best_esti,X_test,np.array(y_test)), best_esti

def makesplits(featurelists, p, n, randseed, n_splits):
    splits = []
    for FEATURELIST in featurelists:  # loop over all the selectors 
        # make some data 
        X,y,df = makeXY(FEATURELIST, p, n)
        X = StandardScaler().fit_transform(X)
        splits += kfold(X, y, n_splits, randseed, True)
    return splits

def random_param_search(featurelists, p, n, randseed, n_splits=2, n_jobs=4, processes=4, debug=False):
    """
    Args:
      featurelists: A list of featurelists.
      p and n: The results from load_data().
      randseed: The seed used.
      n_splits: The number of splits made by each of the featurelists.
      n_jobs: Used for score().
      processes: Number of parallel processes in the multiprocessing pool.
      debug: True if debug mode.
    """
    res = []
    best_esti = []
    splits = makesplits(featurelists, p, n, randseed, n_splits)

    tasks = [(xy, cp) for xy in splits for cp in zip(rs.classifiers,rs.param_lists)]
    with Pool(processes) as pool:
        score2 = partial(score, n_jobs=n_jobs, debug=debug)
        res_best_esti = pool.starmap(score2, tasks)
    for x, y in res_best_esti:
        res.append(x)
        best_esti.append(y)
    res = reshape(res, (len(splits), -1))
    display(HTML(pd.DataFrame(res,columns=rs.clfnames).to_html()))
    return best_esti


if __name__ == "__main__":
    from feature_selection import feature_selection as fs
    debug = True
    randseed = 42

    p, n = load_data(debug)
    X, Y, df = pd_dataframe(p, n)
    folds = kfold(X, Y, n_splits=2, randseed=randseed)
    featurelists = []
    print("A")
    for X_train, X_test, y_train, y_test in folds:
        featurelists += fs(X_train, y_train, df, debug=debug)
    print("B")
    random_param_search(featurelists, p, n, randseed, n_splits=2, debug=debug)
