import pandas as pd
import copy
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split  # tmp
import other.loadfiles as loadfiles
import os

def scorer(esti,x,y):
    yh = esti.predict(x)
    return f1_score(y,yh)

def load_data(debug=False):
    fn = "tmp/pnd.json" if debug else "tmp/pn.json" # Different file for debug mode.
    if os.path.isfile(fn):
        p, n = loadfile(fn)
    else:
        p, n = loadfiles.loaddata("data", numneg=3000 if not debug else 200, pos='1' if debug else 'both', seed=9)
        dumpfile((p,n), fn)
    return p, n


def clean(di, oklist):
    for k in list(di.keys()):
        if k not in oklist:
            di.pop(k)
    return di


def makeXY(featurelist, p, n):
    asd = [clean(e, featurelist) for e in copy.deepcopy(p+n)]
    df = pd.DataFrame(asd)
    X = df.to_numpy()
    y = [1]*len(p)+[0]*len(n)
    return X, y, df


def pd_dataframe(p, n):
    allfeatures = list(p[1].keys())  # the filenames are the last one and we dont need that (for now)
    allfeatures.remove("name")
    X, Y, df = makeXY(allfeatures, p, n)
    return X, Y, df


###################
# KFold Cross Validation.
###################

def kfold(X, y, n_splits=2, randseed=None, shuffle=True):
    """Applies KFold Cross Validation to the given data.
    Returns:
      splirs (List): A list where each entry represents each fold with [X_train, X_test, y_train, y_test]
    """
    #X = StandardScaler().fit_transform(X)
    splits = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    for train, test in kf.split(X, y):
        splits.append([X[train], X[test],
                      [y[i] for i in train], [y[i] for i in test]])
    return splits


##################
# Loading and dumping p and n data files in JSON
##################

def dumpfile(data, fn):
     """saves with loaddata() loaded data as a JSON file."""
     import json
     with open(fn, "w") as f:
         json.dump(data, f)

def loadfile(fn):
    """loads with save_loadeddata() saved files
    and returns its p and n again."""
    import json
    with open(fn, "r") as f:
        return json.load(f)
