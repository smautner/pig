import os
import sys
import argparse
import numpy as np
import input.basics as b
import input.loadfiles as loadfiles
import optimization.feature_selection as fs
import optimization.estimator_parameter as rps
import data.blacklist as blacklist
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


def cleanup(pn=False):
    """Cleans up the tmp folder to prevent inconsistencies when toggling debug.
    Other users will need to change the path with /scratch/bi01/...
    """
    import shutil
    if os.path.exists("tmp/fs_results"):
        shutil.rmtree("tmp/fs_results")
    if os.path.exists("tmp/rps_results"):
        shutil.rmtree("tmp/rps_results")
    for folder in ["pig_o", "pig_e"]:
        if os.path.exists(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}"):
            for file in os.listdir(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}"):
                os.remove(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}/{file}")
            print(f"Cleaned up {folder}")
    for file in os.listdir("tmp"):
        if not (file.startswith("pn_")) or pn:
            os.remove(f"tmp/{file}")
            print(f"Removed tmp/{file}")

#############
# KFold Cross Validation.
#############


def kfold(X, y, n_splits=2, randseed=None, shuffle=True):
    """Applies KFold Cross Validation to the given data.
    Returns:
      splits (List): A list where each entry represents each fold with [X_train, X_test, y_train, y_test]
    """
    from sklearn.model_selection import StratifiedKFold
    splits = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    for train, test in kf.split(X, y):
        splits.append([X[train], X[test],
                      [y[i] for i in train], [y[i] for i in test]])
    return splits


def makefolds(p, n, n_splits, randseed):
    allfeatures = list(p[1].keys())
    allfeatures.remove("name")  # We dont need the filenames (for now)
    X, Y, df = b.makeXY(allfeatures, p, n)
    X = StandardScaler().fit_transform(X)
    folds = kfold(X, Y, n_splits=n_splits, randseed=randseed)
    return folds, df

#############
# Make Featurelists
#############
def load_pn_files(use_rnaz, use_filters, numneg, randseed, debug):
    fn=f"tmp/pn_{use_rnaz}_{use_filters}_{numneg}_{randseed}_{debug}.json"
 
    # If a file with the loaded files already exists, skip loadfiles.loaddata()
    if os.path.isfile(fn):
        p, n = b.loadfile(fn) # pos, neg from loaded file
    else:
        if use_filters:
            p, n = loadfiles.loaddata("data", numneg, randseed, use_rnaz)
        else:
            p, n = loadfiles.loaddata("data", numneg, randseed, use_rnaz, 'both', blacklist_file="noblacklist")
        b.dumpfile((p, n), fn)
    return p, n


def makefltasks(p, n, selection_methods, n_splits, randseed, debug):
    """Creates tasks the cluster uses to create featurelists."""

    folds, df = makefolds(p, n, n_splits, randseed)
    tasks = fs.maketasks(folds, df, selection_methods, randseed, debug)
    numtasks = len(tasks)
    print(f"Created {numtasks} FS tasks.")
    return numtasks


def calculate_featurelists(idd):
    """Executes FS for a given task. Executed by cluster."""
    foldnr, fl, mask, fname, FOLDXY = fs.feature_selection(idd)
    FOLDXY = (FOLDXY[0].tolist(), FOLDXY[1].tolist(), FOLDXY[2], FOLDXY[3])
    b.dumpfile((foldnr, fl, mask, fname, FOLDXY), f"tmp/fs_results/{idd}.json")


def gather_featurelists(clfnames, debug, randseed):
    """Collect results to create the proper featurelists.
    Also creates tmp/rps_tasks for RPS.
    Note: The debug variable needs to be the same value as makefltasks uses.

    Args:
      clfnames (list): What classifiers are beeing used
      debug (bool): Debug mode
      randseed (int): Used seed
    """
    featurelists = defaultdict(list)
    for ftfile in os.listdir("tmp/fs_results"):
        foldnr, fl, mask, fname, FOLDXY = b.loadfile(f"tmp/fs_results/{ftfile}")
        # Append the Featurelists to a dict with their fold number as key
        featurelists[foldnr].append((fl, mask, fname, FOLDXY))
    tasks = rps.maketasks(featurelists, clfnames, randseed)
    numtasks = len(tasks)
    print(f"Created {numtasks} RPS tasks.")
    return numtasks


#############
# Random Parameter Search
#############


def calculate_rps(idd, n_jobs, debug):
    """Executes RPS for a given task. Executed by cluster.
    """
    tasks = np.load("tmp/rps_tasks", allow_pickle=True)
    foldnr, scores, best_esti, ftlist, fname, y_labels = rps.random_param_search(tasks[idd], n_jobs, debug)
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

def skip_feature_selection(set_fl, p, n, clfnames, n_splits, randseed):
    """Skips feature selection if set_fl is not empty.
    To be more precise it skips "makefltasks()", "calculate_featurelists()" and
    "gather_featurelists()" and uses its own method instead.
    Args:
      set_fl(list): A set featurelist that will be used for all folds
      p and n: The loaded filedata created by b.loadfiles()
      clfnames (list): The used classifiers for the random parameter search
      n_splits (int): Number of folds created by kfold
      randseed (int): Random seed used
    """
    folds, df = makefolds(p, n, n_splits, randseed)
    print("Skipping Featureselection using set Featurelist")
    foldnr = 0
    featurelists = {}
    mask = [True if f in set_fl else False for f in df.columns]
    for X_train, X_test, y_train, y_test in folds:
        FOLDXY = [X_train, X_test, y_train, y_test]
        featurelists[foldnr] = [(set_fl, mask, "Set Featurelist", FOLDXY)]
        foldnr += 1
    tasks = rps.maketasks(featurelists, clfnames, randseed)
    numtasks = len(tasks)
    print(f"Created {numtasks} RPS tasks.")
    return numtasks


def makeall(use_rnaz, use_filters, selection_methods, clfnames, n_splits, numneg, randseed, debug, set_fl = []):
    from time import time
    starttime = time()
    # Load files
    print("Loading p and n files")
    p, n = load_pn_files(use_rnaz, use_filters, numneg, randseed, debug)

    if set_fl: # Skip Feature selection process
        rpstasklen = skip_feature_selection(set_fl, p, n, clfnames, n_splits, randseed)
    else:
        # FL tasks
        print(f"{time() - starttime}: Making Featurelist tasks...")
        fstasklen = makefltasks(p, n, selection_methods, n_splits, randseed, debug)
        # Calc FL part -> Cluster
        print(f"{time() - starttime}: Sending {fstasklen} FS tasks to cluster...")
        b.shexec_and_wait(f"qsub -V -t 1-{fstasklen} runall_fs_sge.sh")
        print(f"{time() - starttime}: ...Cluster finished")
        # RPS tasks
        print("Assembling FS lists and RPS tasks...")
        rpstasklen = gather_featurelists(clfnames, debug, randseed)
    if rpstasklen == 0:
        print("No RPS tasks were created. Possibly used wrong parameters.")
        return
    # Calc RPS part -> Cluster
    print(f"{time() - starttime}: Sending {rpstasklen} RPS tasks to cluster...")
    b.shexec_and_wait(f"qsub -V -t 1-{rpstasklen} runall_rps_sge.sh")
    # Results
    print(f"{time() - starttime}: Gathering results...")
    getresults()
    print(f"{time() - starttime}: Done")

def lazymake(use_rnaz, use_filters, selection_methods, n_splits, numneg, randseed, debug, set_fl=[]):
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
        cleanup(False)
        create_directories()
        print(f"----- {name} -----")
        makeall(use_rnaz, use_filters, selection_methods, clfname, n_splits, numneg, randseed, debug, set_fl)
        b.shexec(f"python pig.py --results fen > results/output_{name}.txt")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', type=int, default=0, help='0-no cleanup, 1-normal cleanup, 2-cleanup also removes pn files')
    parser.add_argument('-b', '--blacklist', action='store_true', help='If selected the blacklist will be (re-)created first')
    parser.add_argument('--calcfl', type=int, help='Feature selection process for the cluster')
    parser.add_argument('--calcrps', type=int, help='Random Parameter Search process for the cluster')
    parser.add_argument('-a', '--makeall', action='store_true', help='Ignore given clfnames (--clf) and execute the program with every classifier seperately. Also plots and saves all useful data')
    parser.add_argument('-d', '--debug', action='store_true', help='Use if debug. Overwrites FS arguments')
    parser.add_argument('-r', '--rnaz', action='store_true', help='If used RNAz scores will be added as a feature')
    parser.add_argument('-f', '--filters', action='store_true', help='If used, blacklist will be used in loadfiles')
    parser.add_argument('-o', '--oversample', action='store_true', help='If used, oversampling versions of classifiers will be used')
    parser.add_argument('--lasso', nargs='+', type=float, default=[], help='Lasso for Feature Selection. Warning: Probably cant handle the full 70k+ files and in turn select 0 features and break the program')
    parser.add_argument('--varthresh', nargs='+', type=float, default=[], help='Variance Treshold for Feature Selection. Recommended values: .99 .995 1 1.005 1.01')
    parser.add_argument('--kbest', nargs='+', type=int, default=[], help='Select-K-Best for Feature Selection. Uses Chi2')
    parser.add_argument('--relief', nargs='+', type=int, default=[], help='Relief for Feature Selection. Recommended values: 40 60 80. Warning: Needs very high memory (with the 70k files >12gb)')
    parser.add_argument('--rfecv', nargs='+', type=int, default=[], help='RFECV for Feature Selection. Warning: Insane runtime. Might never end')
    parser.add_argument('--svc1', nargs='+', type=float, default=[], help='SVC with L1 Regularization')
    parser.add_argument('--svc2', nargs='+', type=float, default=[], help='SVC with L2 Regularization')
    parser.add_argument('--forest', nargs='+', type=int, default=[], help='RandomForestClassifier for Feature Selection')
    parser.add_argument('--featurelist', nargs='+', type=str, default=[], help='A optional set featurelist. If this is not empty the feature selection methods will be ignored')
    parser.add_argument('--clf', nargs='+', type=str, choices=('xtratrees', 'gradientboosting', 'neuralnet'), default=['xtratrees', 'gradientboosting', 'neuralnet'], help='Needs to be any of: xtratrees, gradientboosting, neuralnet')
    parser.add_argument('-n', '--nsplits', type=int, default=5, help='Number of splits kfold creates')
    parser.add_argument('--numneg', type=int, default=10000, help='Number of negative (and max of positive) files beeing loaded')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random Seed used for execution')
    parser.add_argument('--results', type=str, default="", help='If used ignore all other arguments and show selected results (options: fenr)')

    args = vars(parser.parse_args())
    debug = args['debug']
    use_rnaz = args['rnaz']
    use_filters = args['filters']
    use_oversampling = args['oversample']
    selection_methods = {'Lasso': args['lasso'], 'VarThresh': args['varthresh'],
                         'SelKBest': args['kbest'], 'Relief': args['relief'],
                         'RFECV': args['rfecv'], 'SVC1': args['svc1'], 'SVC2': args['svc2'], 'Forest': args['forest']}
    clfnames = args['clf']
    n_splits = args['nsplits']
    randseed = args['seed']
    numneg = args['numneg'] # Number of negative files beeing read by b.loaddata()
    n_jobs = 24 # Number of parallel jobs used by RandomizedSearchCV
    set_fl = args['featurelist']
    print("FL:", set_fl)

    if args['blacklist']:
        blacklist.create_blacklist("data")

    if use_oversampling: # Turns the used classifiers into their oversampling equivalents
        clfnames = ['os_' + s for s in clfnames]

    create_directories() # Create all needed directories

    if args['results']: # Instead of executing other functions show previous results
        b.showresults(args['results'], "results.json")
    elif args['calcfl']:
        idd = args['calcfl'] - 1
        calculate_featurelists(idd)
    elif args['calcrps']:
        idd = args['calcrps'] - 1
        calculate_rps(idd, n_jobs, debug)
    else:
        if args['clean']==1: # Removes previously created temporary files to prevent issues
            cleanup(False)
        elif args['clean']==2: # Also removes pn.json or pnd.json so they will need to be reloaded
            cleanup(True)
        if not any(selection_methods.values()):
            pass # No Selection method was given so just return
        elif args['makeall']:
            print(selection_methods)
            print(f"Used Seed: {randseed}")
            lazymake(use_rnaz, use_filters, selection_methods, n_splits, numneg, randseed, debug, set_fl) ###
        else:
            makeall(use_rnaz, use_filters, selection_methods, clfnames, n_splits, numneg, randseed, debug, set_fl)


