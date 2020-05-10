import draw
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV as RSCV
from multiprocessing import Pool
from IPython.display import display, HTML
import randomsearch as  rs
from help_functions import load_data, clean, makeXY, pd_dataframe, scorer, kfold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from functools import partial
from time import time


def score(xy, clf_param, debug):
    X_train, X_test, y_train, y_test = xy
    clf, param = clf_param
    searcher = RSCV(clf, 
                param, 
                n_iter=50 if not debug else 5, 
                scoring=None,
                n_jobs=4,
                iid=False,
                #fefit=True,
                cv=4,
                verbose=0,
                pre_dispatch="2*n_jobs",
                random_state=None,
                error_score=np.nan,
                return_train_score=False)
    searcher.fit(X_train, y_train)
    #print(searcher.best_params_)
    return scorer(searcher.best_estimator_,X_test,np.array(y_test))

def makesplits(featurelists, p, n, randseed, n_splits):
    splits = []
    for FEATURELIST in featurelists:  # loop over all the selectors 
        # make some data 
        X,y,df = makeXY(FEATURELIST, p, n)
        X = StandardScaler().fit_transform(X)
        splits += kfold(X, y, n_splits, randseed, True)
    return splits

def random_param_search(featurelists, p, n, randseed, n_splits=2, processes=4, debug=False):
    from numpy import reshape
    res = []
    splits = makesplits(featurelists, p, n, randseed, n_splits)

    tasks = [(xy, cp) for xy in splits for cp in zip(rs.classifiers,rs.param_lists)]
    with Pool(processes) as pool:
        score2 = partial(score, debug=debug)
        res = pool.starmap(score2, tasks)
    res = reshape(res, (len(splits), -1))
    display(HTML(pd.DataFrame(res,columns=rs.clfnames).to_html()))


if __name__ == "__main__":
    from feature_selection import feature_selection as fs
    from feature_selection import kfold
    from time import time
    debug=True
    testsize = .3
    randseed = 42
    starttime = time()

    p, n = load_data(True)
    X, Y, df = pd_dataframe(p, n)
    folds = kfold(X, Y, n_splits=2, randseed=randseed)
    featurelists = []
    for X_train, X_test, y_train, y_test in folds:
        featurelists.append(fs(X_train, y_train, df, debug=True))
    a = time()
    random_param_search(featurelists, p, n, randseed, n_splits=2, debug=debug)
    print(time()-a)
