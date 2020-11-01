import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.preprocessing import StandardScaler
import optimization.randomsearch as  rs
import traceback as tb


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


def execute_classifier_string(clfname):
    """
    Note: For this to work clfname NEEDS to include a part with 'clf = ClassifierName()'
    """
    exec(clfname, globals())
    return clf


def random_param_search(mask, clfname, foldxy, n_jobs, df, randseed, debug):
    """
    Args:
      task (list): A task made by maketasks().
      foldxy (list): [X_train, X_test, y_train, y_test]
      n_jobs (int): Number of parallel jobs used by score().
      debug (bool): True if debug mode.
    """
    X_train, X_test, y_train, y_test = foldxy
    X_train = StandardScaler().fit_transform(X_train)
    X_train = np.array(X_train)[:,mask]
    X_test = np.array(X_test)[:,mask]
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
    ######
    try:
        coefs = list(zip(df.columns[mask], clf.coef_[0]))
    except:
        print("Given classifier does not have 'coef_' function")
        coefs = ""
    ######
    try:
        y_labels = (y_test, list(clf.predict_proba(X_test)[:,1]))
    except:
        y_labels = ([1,1,1,1,1], [1,1,1,1,1])
        tb.print_stack()
        print("Classifier does not support predict_proba()")
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
    #return foldnr, scores, best_esti, task[3], task[4], y_labels
    return scores, best_esti, y_labels, coefs #####
