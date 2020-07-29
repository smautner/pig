import sys
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
# Make Featurelists
#############


def makefltasks(use_rnaz, use_relief, n_splits, numneg, randseed, debug):
    """Creates tasks the cluster uses to create featurelists."""
    p, n = h.load_data(debug, numneg, randseed, use_rnaz)
    allfeatures = list(p[1].keys())  # the filenames are the last one and we dont need that (for now)
    allfeatures.remove("name")
    X, Y, df = h.makeXY(allfeatures, p, n)
    X = StandardScaler().fit_transform(X)
    folds = h.kfold(X, Y, n_splits=n_splits, randseed=randseed)
    tasks = fs.maketasks(folds, df, use_relief, debug)
    numtasks = len(tasks)
    print(f"Created {numtasks} FS tasks.")
    return numtasks


def calculate_featurelists(idd):
    """Executes FS for a given task. Executed by cluster."""
    foldnr, fl, mask, fname, FOLDXY = fs.feature_selection(idd)
    FOLDXY = (FOLDXY[0].tolist(), FOLDXY[1].tolist(), FOLDXY[2], FOLDXY[3])
    h.dumpfile((foldnr, fl, mask, fname, FOLDXY), f"tmp/fs_results/{idd}.json")


def gather_featurelists(use_mlpc, debug):
    """Collect results to create the proper featurelists.
    Also creates tmp/rps_tasks for RPS.
    Note: The debug variable needs to be the same value as makefltasks uses."""
    featurelists = {}
    for ftfile in os.listdir("tmp/fs_results"):
        foldnr, fl, mask, fname, FOLDXY = h.loadfile(f"tmp/fs_results/{ftfile}")
        if foldnr in featurelists: # Append the Featurelists to a dict with their fold number as key
            featurelists[foldnr].append((fl, mask, fname, FOLDXY))
        else:
            featurelists[foldnr] = [(fl, mask, fname, FOLDXY)]
    tasks = rps.maketasks(featurelists, use_mlpc) # Creates "tmp/rps_tasks"
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
    h.dumpfile([foldnr, scores, best_esti, ftlist, fname, y_labels], f"tmp/rps_results/{idd}.json")
    return best_esti

def getresults():
    """Analyzes the result files in rps_results and
    returns only the ones with the best best_esti_score in each fold.
    """
    results = {}
    for rfile in os.listdir("tmp/rps_results"):
        f = h.loadfile(f"tmp/rps_results/{rfile}")
        if f[0] in results:
            if f[1][0] > results[f[0]][0][0]: # best_esti_score
                results[f[0]] = f[1:]
        else:
            results[f[0]] = f[1:]
    h.dumpfile(results, "results.json")

#############
# Additional Options
#############

def makeall(use_rnaz, use_relief, use_mlpc, n_splits, numneg, randseed, debug):
    # FL tasks
    print("Making Featurelist tasks...")
    fstasklen = makefltasks(use_rnaz, use_relief, n_splits, numneg, randseed, debug)
    # Calc FL part -> Cluster
    print(f"Sending {fstasklen} FS tasks to cluster...")
    b.shexec_and_wait(f"qsub -V -t 1-{fstasklen} runall_fs_sge.sh")
    print("...Cluster finished")
    # RPS tasks
    print("Assembling FS lists and RPS tasks...")
    rpstasklen = gather_featurelists(use_mlpc, debug)
    # Calc RPS part -> Cluster
    print(f"Sending {rpstasklen} RPS tasks to cluster...")
    b.shexec_and_wait(f"qsub -V -t 1-{rpstasklen} runall_rps_sge.sh")
    # Results
    print("Gathering results...")
    getresults()
    print("Done")

def makeall_all(n_splits, numneg, randseed, debug):
    """Temporary function because of lazyness.
    Executes all parameter combinations and saves the results.
    """
    from itertools import product
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    if not os.path.exists("results"):
        os.makedirs("results")
    else:
        for file in os.listdir("results"):
            os.remove(f"results/{file}")
    plt.figure(figsize=(12.8, 9.6))
    plt.plot([0, 1], [0, 1], 'k--')
    for use_rnaz, use_relief, use_mlpc  in product([0, 1], repeat=3):
        cleanup(True)
        create_directories()
        name = f"RNAz-{use_rnaz}_Relief-{use_relief}_MLPC-{use_mlpc}"
        print(f"----- {name} -----")
        makeall(use_rnaz, use_relief, use_mlpc, n_splits, numneg, randseed, debug)
        b.shexec(f"python pig.py showresults fen > results/output_{name}.txt")
        ### The following part saves the data needed for drawing roc curves
        ### For each parameter combination.
        y_true, y_score = [], []
        for sc, be, ft, fn, y_labels in h.loadfile("results.json").values():
            y_true.extend(y_labels[0])
            y_score.extend(y_labels[1])
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} - {round(auc,4)}")
        h.dumpfile((y_true, y_score), f"results/y_data_{name}")
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
    use_mlpc = False # If True MLPClassifier will be used
    use_rnaz = False # If True RNAz scores will be added as a feature
    use_relief = False # If True relief and RFECV will be used
    numneg = 3000 # Number of negative files beeing read by h.loaddata()
    n_splits = 5 if not debug else 2 # Number of splits kfold makes
    n_jobs = 24
    randseed = 42

    create_directories()

    if sys.argv[1] == 'makefltasks':
        makefltasks(use_rnaz, use_relief, n_splits, numneg, randseed, debug)

    elif sys.argv[1] == 'calcfl':
        idd = int(sys.argv[2])-1
        calculate_featurelists(idd)

    elif sys.argv[1] == 'gatherfl':
        gather_featurelists(use_mlpc, debug)

    elif sys.argv[1] == 'calcrps':
        idd = int(sys.argv[2])-1
        calcrps(idd, n_jobs, debug, randseed)

    elif sys.argv[1] == 'getresults':
        getresults()

    elif sys.argv[1] == 'makeall':
        if len(sys.argv) == 2:
            makeall(use_rnaz, use_relief, use_mlpc, n_splits, numneg, randseed, debug)
        elif sys.argv[2] == 'lazy': ### TMP
            makeall_all(n_splits, numneg, randseed, debug) ###

    elif sys.argv[1] == 'showresults':
        if len(sys.argv) == 3:
            h.showresults(sys.argv[2])
        else:
            h.showresults("")

    elif sys.argv[1] == 'cleanup':
        if len(sys.argv) == 2:
            cleanup()
        elif sys.argv[2] == 'True':
            cleanup(True)
        else:
            print("Usage: cleanup (True if pn and pnd should also be removed)")

    else:
        print("Usage: makefltasks -> calcfl(Cluster) -> gatherfl -> calcrps(Cluster) -> getresults -> showresults")
