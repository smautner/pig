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

def maketasks(featurelists, clfnames, randseed):
    """Creates tasks that can be used for random_param_search.
    Args:
      featurelists (dict): A dictionary of featurelists each entry is a fold of featurelists.
      clfnames (list): A list of classifiernames. A list of options can be found in "randomsearch.py"
      randseed (int): The Seed used
    """
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
            tasks.extend([[i, FOLDXY, clfname, FEATURELIST, FUNCNAME, randseed] for clfname in clfnames])
    tasks = np.array(tasks) # task = [FoldNr., FOLDXY, clfname, Featurelist, Functioname, Seed]
    tasks.dump("tmp/rps_tasks")
    return tasks


def execute_classifier_string(clfname):
    """
    Note: For this to work clfname NEEDS to include a part with 'global clf' and 'clf = ClassifierName()'
    """
    exec(clfname, globals())
    return clf


def random_param_search(task, n_jobs, debug):
    """
    Args:
      task (list): A task made by maketasks().
      n_jobs (int): Number of parallel jobs used by score().
      debug (bool): True if debug mode.
    """
    randseed = task[5]
    X_train, X_test, y_train, y_test = task[1] # FOLDXY
    clfname = task[2]
    if len(clfname) < 20:
        clf_param = rs.classifiers[clfname]
        clf_param[1]["random_state"] = [randseed]
        best_esti_score, best_esti = score(X_train, y_train, clf_param, n_jobs, debug, randseed)
        clf = clone(best_esti)
    else:
        clf = execute_classifier_string(clfname)
        best_esti_score = -1
        best_esti = clf
    clf.fit(X_train, y_train)
    y_labels = (y_test, list(clf.predict_proba(X_test)[:,1]))
    y_test = np.array(y_test)
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred)
    test_score = f1_score(y_test, y_pred)
    tpr = sum(y_pred[y_test == 1]) / sum(y_test == 1) # Sensitivity
    tnr = sum(y_pred[y_test == 0] == 0)/sum(y_test == 0) # Specificity
    precision = tpr / (tpr + (1-tnr))
    acc_score = (tpr, tnr, precision)
    scores = (best_esti_score, test_score, acc_score)
    return task[0], scores, best_esti, task[3], task[4], y_labels
