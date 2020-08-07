import sys
import numpy as np
import os
import input.basics as b
import input.loadfiles as loadfiles
import optimization.feature_selection as fs
import optimization.estimator_parameter as rps
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def cleanup(pn=False):
    """Cleans up the tmp folder to prevent inconsistencies when toggling debug.
    """
    import shutil
    if os.path.exists("tmp/fs_results"):
        shutil.rmtree("tmp/fs_results")
    if os.path.exists("tmp/rps_results"):
        shutil.rmtree("tmp/rps_results")
    for folder in ["pig_o", "pig_e"]:
        for file in os.listdir(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}"):
            os.remove(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}/{file}")
        print(f"Cleaned up {folder}")
    for file in os.listdir("tmp"):
        if not file == "blacklist.json": 
            if not (file == "pn.json" or file == "pnd.json") or pn:
                os.remove(f"tmp/{file}")
                print(f"Removed tmp/{file}")

#############
# KFold Cross Validation.
#############


def kfold(X, y, n_splits=2, randseed=None, shuffle=True):
    """Applies KFold Cross Validation to the given data.
    Returns:
      splirs (List): A list where each entry represents each fold with [X_train, X_test, y_train, y_test]
    """
    from sklearn.model_selection import StratifiedKFold
    splits = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    for train, test in kf.split(X, y):
        splits.append([X[train], X[test],
                      [y[i] for i in train], [y[i] for i in test]])
    return splits


#############
# Make Featurelists
#############


def makefltasks(use_rnaz, use_relief, use_filters, n_splits, numneg, randseed, debug):
    """Creates tasks the cluster uses to create featurelists."""
    fn = "tmp/pnd.json" if debug else "tmp/pnf.json" if use_filters else "tmp/pn.json" # Different file for debug mode.

    # If a file with the loaded files already exists, skip loadfiles.loaddata()
    if os.path.isfile(fn):
        p, n = b.loadfile(fn) # pos, neg from loaded file
    else:
        if use_filters:
            p, n = loadfiles.loaddata("data", debug, numneg, randseed, use_rnaz)
        else:
            p, n = loadfiles.loaddata("data", debug, numneg, randseed, use_rnaz, 'both', blacklist_file="noblacklist")
        b.dumpfile((p, n), fn)

    allfeatures = list(p[1].keys())
    allfeatures.remove("name")  # We dont need the filenames (for now)
    X, Y, df = b.makeXY(allfeatures, p, n)
    X = StandardScaler().fit_transform(X)
    folds = kfold(X, Y, n_splits=n_splits, randseed=randseed)
    tasks = fs.maketasks(folds, df, use_relief, debug)
    numtasks = len(tasks)
    print(f"Created {numtasks} FS tasks.")
    return numtasks


def calculate_featurelists(idd):
    """Executes FS for a given task. Executed by cluster."""
    foldnr, fl, mask, fname, FOLDXY = fs.feature_selection(idd)
    FOLDXY = (FOLDXY[0].tolist(), FOLDXY[1].tolist(), FOLDXY[2], FOLDXY[3])
    b.dumpfile((foldnr, fl, mask, fname, FOLDXY), f"tmp/fs_results/{idd}.json")


def gather_featurelists(clfnames, debug):
    """Collect results to create the proper featurelists.
    Also creates tmp/rps_tasks for RPS.
    Note: The debug variable needs to be the same value as makefltasks uses."""
    featurelists = defaultdict(list)
    for ftfile in os.listdir("tmp/fs_results"):
        foldnr, fl, mask, fname, FOLDXY = b.loadfile(f"tmp/fs_results/{ftfile}")
        # Append the Featurelists to a dict with their fold number as key
        featurelists[foldnr].append((fl, mask, fname, FOLDXY))
    tasks = rps.maketasks(featurelists, clfnames)
    numtasks = len(tasks)
    print(f"Created {numtasks} RPS tasks.")
    return numtasks


#############
# Random Parameter Search
#############


def calcrps(idd, n_jobs, debug, randseed):
    """Executes RPS for a given task. Executed by cluster."""

    tasks = np.load("tmp/rps_tasks", allow_pickle=True)
    foldnr, scores, best_esti, ftlist, fname, y_labels = rps.random_param_search(tasks[idd], n_jobs, debug, randseed)
    best_esti = (type(best_esti).__name__, best_esti.get_params()) # Creates readable tuple that can be dumped.
    b.dumpfile([foldnr, scores, best_esti, ftlist, fname, y_labels], f"tmp/rps_results/{idd}.json")

def getresults():
    """Analyzes the result files in rps_results and
    returns only the ones with the best best_esti_score in each fold.
    """
    results = defaultdict(lambda: [[0]])
    for rfile in os.listdir("tmp/rps_results"):
        f = b.loadfile(f"tmp/rps_results/{rfile}")
        if f[1][0] > results[f[0]][0][0]:
            # For each fold the result with the best best_esti_score is saved
            results[f[0]] = f[1:]
        for rfile in os.listdir("tmp/rps_results"):
            f = b.loadfile(f"tmp/rps_results/{rfile}")
        b.dumpfile(results, "results.json")

#############
# Additional Options
#############

def makeall(use_rnaz, use_relief, use_filters, clfnames, n_splits, numneg, randseed, debug):
    from time import time
    starttime = time()
    # FL tasks
    print("Making Featurelist tasks...")
    fstasklen = makefltasks(use_rnaz, use_relief, use_filters, n_splits, numneg, randseed, debug)
    print(f"Time since start: {time() - starttime}")
    # Calc FL part -> Cluster
    print(f"Sending {fstasklen} FS tasks to cluster...")
    b.shexec_and_wait(f"qsub -V -t 1-{fstasklen} runall_fs_sge.sh")
    print("...Cluster finished")
    print(f"Time since start: {time() - starttime}")
    # RPS tasks
    print("Assembling FS lists and RPS tasks...")
    rpstasklen = gather_featurelists(clfnames, debug)
    # Calc RPS part -> Cluster
    print(f"Time since start: {time() - starttime}")
    print(f"Sending {rpstasklen} RPS tasks to cluster...")
    b.shexec_and_wait(f"qsub -V -t 1-{rpstasklen} runall_rps_sge.sh")
    # Results
    print(f"Time since start: {time() - starttime}")
    print("Gathering results...")
    getresults()
    print("Done")
    print(f"Time since start: {time() - starttime}")

def makeall_all(use_rnaz, use_relief, use_filters, n_splits, numneg, randseed, debug):
    """Temporary function because of lazyness.
    Executes parameter combinations and saves the results.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    if not os.path.exists("results"):
        os.makedirs("results")
    else:
        for file in os.listdir("results"):
            os.remove(f"results/{file}")
    plt.figure(figsize=(12.8, 9.6))
    plt.plot([0, 1], [0, 1], 'k--')
    for clfname in [['gradientboosting'], ['os_gradientboosting'], ['neuralnet'], ['os_neuralnet']]:
        name = clfname[0]
        cleanup(True)
        create_directories()
        print(f"----- {name} -----")
        makeall(use_rnaz, use_relief, use_filters, clfname, n_splits, numneg, randseed, debug)
        b.shexec(f"python pig.py showresults fen > results/output_{name}.txt")
        ### The following part saves the data needed for drawing roc curves
        ### For each parameter combination.
        y_true, y_score = [], []
        for sc, be, ft, fn, y_labels in b.loadfile("results.json").values():
            y_true.extend(y_labels[0])
            y_score.extend(y_labels[1])
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} - {round(auc,4)}")
        b.dumpfile((y_true, y_score), f"results/y_data_{name}")
        os.rename("results.json", f"results/results_{name}.json")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.savefig("results/roc_curves")

def create_directories():
    if not os.path.exists("tmp"):
        print("Creating tmp directory")
        os.makedirs("tmp")
    if not os.path.exists("tmp/fs_results"):
        print("Creating tmp/fs_results directory")
        os.makedirs("tmp/fs_results")
    if not os.path.exists("tmp/rps_results"):
        print("Creating tmp/rps_results directory")
        os.makedirs("tmp/rps_results") 

#############
# Main Function
#############

if __name__ == "__main__":
    debug = True
    use_rnaz = False # If True RNAz scores will be added as a feature
    use_relief = False # If True relief and RFECV will be used
    use_mlpc = True # If True MLPClassifier will be used
    use_filters = True # If False, blacklist in loadfiles will not be used.
    use_oversampling = False # If True, oversampling versions of classifiers will be used instead
    numneg = 80000 if not debug else 200 # Number of negative files beeing read by b.loaddata()
    n_splits = 5 if not debug else 2 # Number of splits kfold makes
    n_jobs = 24
    randseed = 42

    if use_oversampling: # Sets the used classifiers based on the parameters above
        clfnames = ['os_xtratrees', 'os_gradientboosting']
        if use_mlpc:
            clfnames.append('os_neuralnet')
    else:
        clfnames = ['xtratrees', 'gradientboosting']
        if use_mlpc:
            clfnames.append('neuralnet')

    create_directories()

    if sys.argv[1] == 'makefltasks':
        makefltasks(use_rnaz, use_relief, use_filters, n_splits, numneg, randseed, debug)

    elif sys.argv[1] == 'calcfl':
        idd = int(sys.argv[2])-1
        calculate_featurelists(idd)

    elif sys.argv[1] == 'gatherfl':
        gather_featurelists(clfnames, debug)

    elif sys.argv[1] == 'calcrps':
        idd = int(sys.argv[2])-1
        calcrps(idd, n_jobs, debug, randseed)

    elif sys.argv[1] == 'getresults':
        getresults()

    elif sys.argv[1] == 'makeall':
        if len(sys.argv) == 2:
            makeall(use_rnaz, use_relief, use_filters, clfnames, n_splits, numneg, randseed, debug)
        elif sys.argv[2] == 'lazy': ### TMP
            makeall_all(use_rnaz, use_relief, use_filters, n_splits, numneg, randseed, debug) ###

    elif sys.argv[1] == 'showresults':
        if len(sys.argv) == 3:
            b.showresults(sys.argv[2])
        else:
            b.showresults("")

    elif sys.argv[1] == 'cleanup':
        if len(sys.argv) == 2:
            cleanup()
        elif sys.argv[2] == 'True':
            cleanup(True)
        else:
            print("Usage: cleanup (True if pn and pnd should also be removed)")

    else:
        print("Usage: makefltasks -> calcfl(Cluster) -> gatherfl -> calcrps(Cluster) -> getresults -> showresults")
