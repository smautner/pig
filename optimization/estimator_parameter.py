import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.preprocessing import StandardScaler
import optimization.randomsearch as  rs


def score(X, y, clf_param, n_jobs, debug, randseed):
    clf, param = clf_param
    searcher = RSCV(clf, 
                param, 
                n_iter=50 if not debug else 5, 
                scoring="f1",
                n_jobs=n_jobs,
                #iid=False,
                #fefit=True,
                cv=5,
                verbose=0,
                pre_dispatch="2*n_jobs",
                random_state=randseed,
                error_score=np.nan,
                return_train_score=False)
    searcher.fit(X, y)
    best_esti_score = searcher.best_score_
    best_esti = searcher.best_estimator_
    return best_esti_score, best_esti

def maketasks(featurelists, clfnames):
    """Creates tasks that can be used for random_param_search.
    Args:
      featurelists (dict): A dictionary of featurelists each entry is a fold of featurelists.
      clfnames (list): A list of classifiernames. A list of options can be found in "randomsearch.py"
    """
    clf =   [rs.classifiers[clfname][0] for clfname in clfnames]
    param = [rs.classifiers[clfname][1] for clfname in clfnames]

    tasks = []
    for i in range(0, len(featurelists)): # Each list is a fold
        for flist in featurelists[i]: # Each fold contains featurelists.
            FEATURELIST = flist[0]
            mask = flist[1]
            FUNCNAME = flist[2]
            FOLDXY = flist[3] # Includes X and y train/test from the fold
            X_train, X_test = FOLDXY[0], FOLDXY[1]
            X_train = StandardScaler().fit_transform(X_train)
            FOLDXY[0] = np.array(X_train)[:,mask]
            FOLDXY[1] = np.array(X_test)[:,mask]
            tasks.extend([[i, FOLDXY, list(cp), FEATURELIST, FUNCNAME] for cp in zip(clf, param)])
    tasks = np.array(tasks) # task = [FoldNr., FOLDXY, classifier/param, Featurelist, Function_name]
    tasks.dump("tmp/rps_tasks")
    return tasks

def random_param_search(task, n_jobs, debug, randseed):
    """
    Args:
      task (list): A task made by maketasks().
      n_jobs (int): Number of parallel jobs used by score().
      debug (bool): True if debug mode.
      randseed (int): Seed used.
    """
    X_train, X_test, y_train, y_test = task[1] # FOLDXY
    task[2][1]["random_state"] = [randseed]
    best_esti_score, best_esti = score(X_train, y_train, task[2], n_jobs, debug, randseed)
    clf = clone(best_esti)
    clf.fit(X_train, y_train)
    y_labels = (y_test, list(clf.predict_proba(X_test)[:,1]))
    y_test = np.array(y_test)
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred)
    test_score = f1_score(y_test, y_pred)
    tpr = sum(y_pred[y_test == 1]) / sum(y_test == 1)
    tnr = sum(y_pred[y_test == 0] == 0)/sum(y_test == 0)
    acc_score = (tpr, tnr)
    scores = (best_esti_score, test_score, acc_score)
    return task[0], scores, best_esti, task[3], task[4], y_labels
