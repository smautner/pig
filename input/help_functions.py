import pandas as pd
import copy
from sklearn.model_selection import StratifiedKFold


def clean(di, oklist):
    for k in list(di.keys()):
        if k not in oklist:
            di.pop(k)
    return di


def makeXY(featurelist, p, n):
    asd = [clean(e, featurelist) for e in copy.deepcopy(p+n)]
    df = pd.DataFrame(asd)
    df = df.transpose().drop_duplicates().transpose() # Remove Duplicates
    X = df.to_numpy()
    y = [1]*len(p)+[0]*len(n)
    return X, y, df


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
     """Saves given data as a JSON file."""
     import json
     with open(fn, "w") as f:
         json.dump(data, f)

def loadfile(fn):
    """Loads a JSON file and returns its values."""
    import json
    with open(fn, "r") as f:
        return json.load(f)

###################
# Printing out Results
###################

def showresults(args=""):
    from collections import Counter, defaultdict
    from pprint import pprint
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    results = loadfile("results.json")
    estimators = defaultdict(lambda: defaultdict(list))
    ftlists = []
    c = Counter()
    y_true = []
    y_score = []
    for scores, best_esti, ftlist, fname, y_labels in results.values():
        esti_name, params = best_esti
        best_esti_score, test_score, accuracy_score = scores
        params["test_score"] = round(test_score, 4)
        params["best_esti_score"] = round(best_esti_score, 4)
        params["accuracy_score"] = [round(acc, 4) for acc in accuracy_score]
        for key, value in params.items():
            estimators[esti_name][key].append(value)
        ftlists.append((fname, ftlist)) # ?
        c.update(ftlist)
        y_true.extend(y_labels[0])
        y_score.extend(y_labels[1])
    if "f" in args:
        pprint(c.most_common())
        print("\n")
    if "e" in args:
        for key in estimators.keys():
            print(f"{key}:")
            print("-" * (len(key)+1))
            for param in estimators[key].items():
                print(f"{param[0]}: {param[1]}")
            print("\n")
    if "n" in args:
        for x in ftlists:
            pprint((x[0], len(x[1]), x[1]))
    if "r" in args:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=f"{round(auc, 4)}")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='best')
        plt.show()
    if  "h" in args:
        print("Usage: pig.py showresults {fen}\n", \
              "f - featurelists with number of occurences\n", \
              "e - estimators\n", \
              "n - Shows ALL featurelists with the info used to create them\n", \
              "r - Creates and plots the roc_curve")
