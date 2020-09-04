lmap = lambda *x: list(map(*x))
lzip = lambda *x: list(zip(*x))


import dill
import time
import pandas as pd
import copy
from collections import Counter, defaultdict
from pprint import pprint
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

def mpmap(func, iterable, chunksize=10, poolsize=2):
    import multiprocessing as mp
    """pmap."""
    pool = mp.Pool(poolsize)
    result = pool.map(func, iterable, chunksize=chunksize)
    pool.close()
    pool.join()
    return list(result)

# functools partial can set some arguments...
def mpmap_prog(func, iterable, chunksize=10, poolsize=2):
    import multiprocessing as mp
    import tqdm
    """pmap."""
    pool = mp.Pool(poolsize)
    result = list(tqdm.tqdm( pool.imap(func, iterable, chunksize=chunksize), total=len(iterable)))
    pool.close()
    pool.join()
    return result

def shexec(cmd):
    import subprocess
    '''
    :param cmd:
    :return: (exit-code, stderr, stdout)
    the subprocess module is chogeum.. here is a workaround
    '''
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, stderr = process.communicate()
    retcode = process.poll()
    return (retcode,stderr,output) # .decode('utf-8') something like this might be necessary for py3

def shexec_and_wait(cmd):
    ret,stderr,out = shexec(cmd)
    taskid = out.split()[2][:7]
    print("taskid:", int(taskid))
    while taskid in shexec("qstat")[2]:
        time.sleep(10)

def interact():
    import code
    code.interact(local=dict(globals(), **locals()))


###################
# Create X and y
###################

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

def showresults(args, resultfile):
    results = loadfile(resultfile)
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
            pprint((x[0], len(x[1]), sorted(x[1])))
    if "r" in args:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=f"{round(auc, 4)}")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='best')
        plt.show()
    if "p" in args:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
    if  "h" in args:
        print("Usage: pig.py -r {fenrp}\n", \
              "f - featurelists with number of occurences\n", \
              "e - estimators\n", \
              "n - Shows ALL featurelists with the info used to create them\n", \
              "r - Creates and plots the roc_curve\n", \
              "p - Creates and plots the precision_recall_curve")
