import sys
import numpy as np
import other.randomsearch as  rs
import os
import other.help_functions as h
from feature_selection import feature_selection
import rps

############
#
#############


def getfeaturelist(n_splits, randseed, debug):
    p, n = h.load_data(debug)
    X, Y, df = h.pd_dataframe(p, n)
    folds = h.kfold(X, Y, n_splits=n_splits, randseed=randseed)
    featurelists = []
    func_names = []
    for X_train, X_test, y_train, y_test in folds:
        fs = feature_selection(X_train, y_train, df, debug=debug)
        featurelists.append(fs[0])
        func_names.append(fs[1])
    #h.dumpfile(featurelists, "tmp/ftlist.json") #Not really required but maybe would help?
    #h.dumpfile(func_names, "tmp/funcnames.json") #Same as above.
    rps.maketasks(featurelists, func_names, p, n, randseed, n_splits) # Creates "tmp/tasks" file

def getnumtasks():
    tasks  = np.load("tmp/tasks", allow_pickle=True)
    return len(tasks)

def gettasks_annotated():
    ftlist  = h.loadfile("tmp/ftlist.json")
    tasks = [(clf,param,ftli, clfname,ftid)for clf,param,clfname in zip(rs.classifiers,rs.param_lists, rs.clfnames) for ftid,ftli in enumerate(ftlist)] 
    return tasks

def doajob(idd, debug):
    tasks = np.load("tmp/tasks", allow_pickle=True) ###
    foldnr, best_score, best_esti, ftlist, fname = rps.random_param_search(tasks[idd], n_jobs=4, debug=debug)
    #np.array([best_score, best_esti, ftlist, fname]).dump(f"tmp/results/{idd}")
    h.dumpfile([foldnr, best_score, str(best_esti), ftlist, fname], f"tmp/results/{idd}.json")
    #print(best_score, best_esti, ftlist, fname)

def getresults():
    results = {}
    for rfile in os.listdir("tmp/results"):
        f = h.loadfile(f"tmp/results/{rfile}")
        if f[0] in results:
            if f[1] > results[f[0]][1]:
                results[f[0]] = f
        else:
            results[f[0]] = f
    h.dumpfile(results, "results.json")
    return results


if __name__ == "__main__":
    debug = True
    n_splits = 2
    randseed = 42

    if not os.path.exists("tmp/results"):
        print("Creating tmp/results directory")
        os.makedirs("tmp/results")
    
    if sys.argv[1] == 'maxtask':
        print(getnumtasks()) # for use in qsub.. 

    elif sys.argv[1] == 'getresults':
        getresults()

    elif sys.argv[1] == 'showresult':
        print(h.loadfile("results.json")[sys.argv[2]])

    elif sys.argv[1] == 'makeftlist':
        getfeaturelist(n_splits, randseed, debug)

    else:
        idd = int(sys.argv[1])-1
        doajob(idd, debug)


