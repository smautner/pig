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
        # for ftlistnr in range(0, len(featurelists[i])):
        for flist in featurelists[i]: # Each fold contains featurelists.
            ftlist = flist[0]
            mask = flist[1]
            fname = flist[2]
            #tasks.extend([[i, ftlistnr, clfname, randseed] for clfname in clfnames])
            tasks.extend([[i, mask, clfname, ftlist, fname, randseed] for clfname in clfnames])
    tasks = np.array(tasks) # task = [FoldNr., mask, clfname, Featurelist, Functioname, Seed]
    return tasks


def execute_classifier_string(clfname):
    """
    Note: For this to work clfname NEEDS to include a part with 'clf = ClassifierName()'
    """
    exec(clfname, globals())
    return clf


def random_param_search(task, foldxy, n_jobs, debug):
    """
    Args:
      task (list): A task made by maketasks().
      foldxy (list): [X_train, X_test, y_train, y_test]
      n_jobs (int): Number of parallel jobs used by score().
      debug (bool): True if debug mode.
    """
    foldnr = task[0]
    mask = task[1]
    X_train, X_test, y_train, y_test = foldxy
    X_train = StandardScaler().fit_transform(X_train)
    X_train = np.array(X_train)[:,mask]
    X_test = np.array(X_test)[:,mask]
    randseed = task[5]
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
    #precision = tpr / (tpr + (1-tnr))
    precision = sum(y_pred[y_test == 1]) / sum(y_pred == 1)
    if np.isnan(tpr):
        tpr = 0
    if np.isnan(tnr):
        tnr = 0
    if np.isnan(precision):
        precision = 0
    acc_score = (tpr, tnr, precision)
    scores = (best_esti_score, test_score, acc_score)
    return foldnr, scores, best_esti, task[3], task[4], y_labels
