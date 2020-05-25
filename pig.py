import sys
import numpy as np
import other.randomsearch as  rs
#import basics as b#####
import other.help_functions as h
from feature_selection import feature_selection as fs
import rps

############
#
#############

def getfeaturelist(n_splits=2, randseed=42, debug=True):
    
    p, n = h.load_data(debug)
    X, Y, df = h.pd_dataframe(p, n)
    folds = h.kfold(X, Y, n_splits=n_splits, randseed=randseed)
    featurelists = []
    for X_train, X_test, y_train, y_test in folds:
        featurelists += fs(X_train, y_train, df, debug=debug)
    h.dumpfile(featurelists, "ftlist.json")
    rps.maketasks(featurelists, p, n, randseed, n_splits)
    return featurelists
    

def gettasks():
    ftlist  = h.loadfile("ftlist.json")
    tasks = [(clf,param,ftli)for clf,param in zip(rs.classifiers,rs.param_lists) for ftli in ftlist]
    return tasks

def gettasks_annotated():
    ftlist  = h.loadfile("ftlist.json")
    tasks = [(clf,param,ftli, clfname,ftid)for clf,param,clfname in zip(rs.classifiers,rs.param_lists, rs.clfnames) for ftid,ftli in enumerate(ftlist)] 
    return tasks

def doajob(idd, debug):
    tasks = np.load("tasks", allow_pickle=True)
    best_esti = rps.random_param_search(tasks[idd], n_jobs=4, debug=debug)
    print(best_esti)
        

def readresult(fname):
    a= open(fname,"r").read()
    dic,numerics = a.split("}")
    params = eval(dic+"}")
    f1,perf,jid,time = [float(a) for a in numerics.split()]
    return f1,perf, dic , jid , time
    


if __name__ == "__main__":
    debug = True
    if sys.argv[1] == 'maxtask':
        print(len( gettasks_annotated() )) # for use in qsub.. 

    elif sys.argv[1] == 'showft':
        features = h.loadfile("ftlist.json")
        ft= list(features[int(sys.argv[2])])
        import pprint
        pprint.pprint(ft)
        print("numft:",len(ft))
    elif sys.argv[1] == 'makeftlist':
        getfeaturelist()
    else:
        idd = int(sys.argv[1])-1
        doajob(idd, debug)


