import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.preprocessing import StandardScaler
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
                cv=5,
                verbose=0,
                pre_dispatch="2*n_jobs",
                random_state=None,
                error_score=np.nan,
                return_train_score=False)
    searcher.fit(X, y)
    best_esti_score = searcher.best_score_
    best_esti = searcher.best_estimator_
    return best_esti_score, best_esti

def maketasks(featurelists, p, n, randseed):
    """Creates tasks that can be used for random_param_search.
    Args:
      featurelists: A dictionary of featurelists each entry is a fold of featurelists.
      p and n: The results from load_data().
      randseed: The seed used.
    """
    tasks = []
    for i in range(0, len(featurelists)): # Each list is a fold
        for flist in featurelists[i]: # Each fold contains featurelists.
            FEATURELIST = flist[0]
            FUNCNAME = flist[1]
            FOLDXY = flist[2] # Includes X and y train/test from the fold
            X,y,df = h.makeXY(FEATURELIST, p, n)
            X = StandardScaler().fit_transform(X)
            tasks.extend([(i, X, y, cp, FEATURELIST, FUNCNAME, FOLDXY) for cp in zip(rs.classifiers,rs.param_lists)])
    tasks = np.array(tasks) # task = (FoldNr., X, y, classifier/param, Featurelist, Function_name, FoldXY)
    tasks.dump("tmp/rps_tasks")
    return tasks

def random_param_search(task, n_jobs=4, debug=False):
    """
    Args:
      task: A task made by maketasks().
      n_jobs: Number of parallel jobs used by score().
      debug: True if debug mode.
    """
    X_train, X_test, y_train, y_test = task[6] # FOLDXY
    best_esti_score, best_esti = score(task[1], task[2], task[3], n_jobs, debug)
    clf = clone(best_esti)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    test_score = f1_score(y_test, y_pred) #
    acc_score = accuracy_score(y_test, y_pred)#
    scores = (best_esti_score, test_score, acc_score)
    return task[0], scores, best_esti, task[4], task[5]
