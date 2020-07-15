import pandas as pd
import copy
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import other.loadfiles as loadfiles
import os

def scorer(esti,x,y):
    yh = esti.predict(x)
    return f1_score(y,yh)

def load_data(debug, randseed, use_rnaz):
    fn = "tmp/pnd.json" if debug else "tmp/pn.json" # Different file for debug mode.
    if os.path.isfile(fn):
        p, n = loadfile(fn)
    else:
        p, n = loadfiles.loaddata("data", numneg=3000 if not debug else 200, pos='1' if debug else 'both', seed=randseed, use_rnaz=use_rnaz)
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

##
##def pd_dataframe(p, n):
##    allfeatures = list(p[1].keys())  # the filenames are the last one and we dont need that (for now)
##    allfeatures.remove("name")
##    X, Y, df = makeXY(allfeatures, p, n)
##    return X, Y, df


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
# Loading and dumping data files in JSON
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

###################
# Printing out Results
###################

def showresults(args=""):
    from collections import Counter
    from pprint import pprint
    results = loadfile("results.json")
    estimators = {}
    ftlists = []
    c = Counter()
    #for best_score, best_esti, ftlist, fname in results.values():
    for scores, best_esti, ftlist, fname in results.values():
        esti_name, params = best_esti
        best_esti_score, test_score, accuracy_score = scores
        params["test_score"] = round(test_score, 4)
        params["best_esti_score"] = round(best_esti_score, 4)
        params["accuracy_score"] = [round(acc, 4) for acc in accuracy_score]
        if esti_name not in estimators:
            estimators[esti_name] = {}
        for key, value in params.items():
            if key in estimators[esti_name]:
                estimators[esti_name][key].append(value)
            else:
                estimators[esti_name][key] = [value]
        ftlists.append((fname, ftlist)) # ?
        c.update(ftlist)
    print_help = True
    if "f" in args:
        pprint(c.most_common())
        print("\n")
        print_help = False
    if "e" in args:
        for key in estimators.keys():
            print(f"{key}:")
            print("-" * (len(key)+1))
            for param in estimators[key].items():
                print(f"{param[0]}: {param[1]}")
            print("\n")
        print_help = False
    if "n" in args:
        for x in ftlists:
            pprint((x[0], len(x[1]), x[1]))
        print_help = False
    if print_help:
        print("Usage: pig.py showresults {fen}\n", \
              "f - featurelists with number of occurences\n e - estimators\n", \
              "n - Shows ALL featurelists with the info used to create them")
