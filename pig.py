import sys
import time
import numpy as np
import os
import other.help_functions as h
import other.basics as b
import core.feature_selection as fs
import core.rps as rps
from sklearn.preprocessing import StandardScaler



def cleanup(pn=False):
    """Cleans up the tmp folder to prevent inconsistencies when toggling debug.
    """
    import shutil
    if os.path.exists("tmp/fs_results"):
        shutil.rmtree("tmp/fs_results")
    if os.path.exists("tmp/rps_results"):
        shutil.rmtree("tmp/rps_results")
    for file in os.listdir("tmp"):
        if not (file == "pn.json" or file == "pnd.json") or pn:
            os.remove(f"tmp/{file}")
            print(f"Removed tmp/{file}")
        

#############
# Make Featurelists
#############


def makefltasks(n_splits, randseed, debug):
    """Creates tasks the cluster uses to create featurelists."""
    p, n = h.load_data(debug)
    X, Y, df = h.pd_dataframe(p, n)
    X = StandardScaler().fit_transform(X)
    folds = h.kfold(X, Y, n_splits=n_splits, randseed=randseed)
    tasks = fs.maketasks(folds, df, debug)
    print(f"Created {len(tasks)} FS tasks.")
    return len(tasks)


def calculate_featurelists(idd):
    """Executed by Cluster"""
    foldnr, fl, fname = fs.feature_selection(idd)
    h.dumpfile([foldnr, fl, fname], f"tmp/fs_results/{idd}.json")


def gather_featurelists(debug):
    """Collect results to create the proper featurelists.
    Also creates tmp/rps_tasks for RPS.
    Note: The debug variable needs to be the same value as makefltasks uses."""
    featurelists = {}
    for ftfile in os.listdir("tmp/fs_results"):
        foldnr, fl, fname = h.loadfile(f"tmp/fs_results/{ftfile}")
        if foldnr in featurelists:
            featurelists[foldnr].append((fl, fname))
        else:
            featurelists[foldnr] = [(fl, fname)]
    h.dumpfile(featurelists, "tmp/ftlists.json") # Used to rank features
    p, n = h.load_data(debug)
    tasks = rps.maketasks(featurelists, p, n, randseed, n_splits) # Creates "tmp/rps_tasks" file
    print(f"Created {len(tasks)} RPS tasks.")
    return len(tasks)


def get_top_features(k=None):
    """Optional function. Returns the k most used features by ALL featurelists.
    Not very efficient but probably does the job for now."""
    from collections import Counter
    featurelists = h.loadfile("tmp/ftlists.json")
    fl = list(featurelists.values())
    c = Counter()
    for fold in fl:
        for ftlist in fold:
            c.update(ftlist[0])
    return c.most_common(k)


#############
# Random Parameter Search
#############


def doajob(idd, debug):
    tasks = np.load("tmp/rps_tasks", allow_pickle=True)
    foldnr, best_score, best_esti, ftlist, fname = rps.random_param_search(tasks[idd], n_jobs=24, debug=debug)
    best_esti = (type(best_esti).__name__, best_esti.get_params()) # Creates readable tuple that can be dumped.
    h.dumpfile([foldnr, best_score, best_esti, ftlist, fname], f"tmp/rps_results/{idd}.json")
    return best_esti

def getresults():
    results = {}
    for rfile in os.listdir("tmp/rps_results"):
        f = h.loadfile(f"tmp/rps_results/{rfile}")
        if f[0] in results:
            if f[1] > results[f[0]][0]:
                results[f[0]] = f[1:]
        else:
            results[f[0]] = f[1:]
    h.dumpfile(results, "results.json")
    return results

#############
# Additional Options
#############

def makeall(n_splits, randseed, debug):
    # FL tasks
    print("Making Featurelist tasks...")
    fstasklen = makefltasks(n_splits, randseed, debug)
    #Calc FL Part -> Cluster
    print(f"Sending {fstasklen} FS tasks to cluster...")
    ret,stderr,out = b.shexec(f"qsub -V -t 1-{fstasklen} runall_fs_sge.sh")
    taskid = out.split()[2][:7]
    print("taskid:", int(taskid))
    while taskid in b.shexec("qstat")[2]: # "not ba.shexc("qstat"[2]" would be enough if user only runs 1 thing at a time
        time.sleep(10)
    print("...Cluster finished")
    # RPS tasks
    print("Assembling FS lists and RPS tasks...")
    rpstasklen = gather_featurelists(debug)
    print(f"Sending {rpstasklen} RPS tasks to cluster...")
    #Calc RPS Part -> Cluster
    ret,stderr,out = b.shexec(f"qsub -V -t 1-{rpstasklen} runall_rps_sge.sh")
    taskid = out.split()[2][:7]
    print("taskid:", int(taskid))
    while taskid in b.shexec("qstat")[2]:
        time.sleep(10)
    # Results
    print("Gathering results...")
    getresults()
    print("Done")


#############
# Main Function
#############

if __name__ == "__main__":
    debug = False
    n_splits = 10 if not debug else 2
    randseed = 42

    if not os.path.exists("tmp"):
        print("Creating tmp directory")
        os.makedirs("tmp")
    if not os.path.exists("tmp/fs_results"):
        print("Creating tmp/fs_results directory")
        os.makedirs("tmp/fs_results")
    if not os.path.exists("tmp/rps_results"):
        print("Creating tmp/rps_results directory")
        os.makedirs("tmp/rps_results") 


    if sys.argv[1] == 'makefltasks':
        makefltasks(n_splits, randseed, debug)

    elif sys.argv[1] == 'calcfl':
        idd = int(sys.argv[2])-1
        calculate_featurelists(idd)

    elif sys.argv[1] == 'gatherfl':
        gather_featurelists(debug)

    elif sys.argv[1] == 'calcrps':
        idd = int(sys.argv[2])-1
        doajob(idd, debug)

    elif sys.argv[1] == 'getresults':
        getresults()

    elif sys.argv[1] == 'makeall':
        makeall(n_splits, randseed, debug)

    elif sys.argv[1] == 'showresults':
        if len(sys.argv) == 3:
            h.showresults(sys.argv[2])
        else:
            h.showresults("")

    elif sys.argv[1] == 'topfeatures':
        if len(sys.argv) == 2:
            get_top_features()
        else:
            get_top_features(sys.argv[2])

    elif sys.argv[1] == 'cleanup':
        if len(sys.argv) == 2:
            cleanup()
        elif sys.argv[2] == 'True':
            cleanup(True)
        else:
            print("Usage: cleanup (True if pn and pnd should also be removed)")

    else:
        print("Usage: makefltasks -> calcfl(Cluster) -> gatherfl -> calcrps(Cluster) -> getresults -> showresults")


