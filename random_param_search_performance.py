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


def score(clf, param, Xy, debug):
    X_train, X_test, y_train, y_test = Xy
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


def random_param_search(featurelists, p, n, randseed, n_splits=2, processes=4, debug=False):
    t1 = time()
    res = []
    tasks = [(clf,param) for clf,param in zip(rs.classifiers,rs.param_lists)]
    for FEATURELIST in featurelists:  # loop over all the selectors 
        # make some data 
        X,y,df = makeXY(FEATURELIST, p, n)
        X = StandardScaler().fit_transform(X)
        splits = [kfold(X, y, n_splits, randseed, True)[0]]

        with Pool(processes) as pool:
            for Xy in splits:
                score2 = partial(score, Xy=Xy, debug=debug)
                res.append(pool.starmap(score2, tasks))
###### Demonstration #######
##        with Pool(processes) as pool:
##            score2 = partial(score, Xy=splits[0], debug=debug)
##            res.append(pool.starmap(score2, tasks))
##        score2 = partial(score, Xy=splits[0], debug=debug)
##        res.append(score2(clf, param) for clf, param in tasks)####
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
    random_param_search(featurelists[0], p, n, randseed, n_splits=2, debug=debug)
    print(time()-a)
