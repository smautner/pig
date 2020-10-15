import os
import argparse
import pandas as pd
import numpy as np
import input.basics as b
import input.showresults as res
import input.loadfiles as loadfiles
import optimization.feature_selection as fs
import optimization.estimator_parameter as rps
import data.blacklist as blacklist
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Location for temporary files. Remember to clean the old location before changing.
tmpdirectory = "/scratch/bi01/mautner/guest10/tmp"

def cleanup(pn=False):
    """Cleans up the tmp folder to prevent inconsistencies when toggling debug.
    Other users will need to change the path with /scratch/bi01/...
    """
    import shutil
    for folder in ["pig_o", "pig_e"]:
        if os.path.exists(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}"):
            for file in os.listdir(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}"):
                os.remove(f"/scratch/bi01/mautner/guest10/JOBZ/{folder}/{file}")
            print(f"Cleaned up {folder}")
    if os.path.exists(f"{tmpdirectory}"):
        if os.path.exists(f"{tmpdirectory}/fs_results"):
            shutil.rmtree(f"{tmpdirectory}/fs_results")
        if os.path.exists(f"{tmpdirectory}/rps_results"):
            shutil.rmtree(f"{tmpdirectory}/rps_results")
        for file in os.listdir(f"{tmpdirectory}"):
            if not (file.startswith("pn_")) or pn:
                os.remove(f"{tmpdirectory}/{file}")
                print(f"Removed {tmpdirectory}/{file}")

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


#############
# Load Data
#############


def load_pn_files(use_rnaz, use_filters, numneg, randseed, debug):
    fn=f"{tmpdirectory}/pn_{use_rnaz}_{use_filters}_{numneg}_{randseed}_{debug}.json"
 
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


def makefltasks(p, n, selection_methods, n_splits, randseed, debug):
    """Creates tasks the cluster uses to create featurelists."""

    folds, df = makefolds(p, n, n_splits, randseed)
    tasks = fs.maketasks(len(folds), selection_methods, randseed, debug)
    numtasks = len(tasks)
    np.array(folds, dtype=object).dump(f"{tmpdirectory}/folds.pkl")
    df.to_pickle(f"{tmpdirectory}/dataframe.pkl")
    np.array(tasks, dtype=object).dump(f"{tmpdirectory}/fs_tasks.pkl")
    print(f"Created {numtasks} FS tasks.")
    return numtasks, df


def calculate_featurelists(idd):
    """Executes FS for a given task. Executed by cluster."""
    task = np.load(f"{tmpdirectory}/fs_tasks.pkl", allow_pickle=True)[idd]
    foldxy = np.load(f"{tmpdirectory}/folds.pkl", allow_pickle=True)[task[0]] # task[0] = foldnr
    df = pd.read_pickle(f"{tmpdirectory}/dataframe.pkl")
    foldnr, fl, mask, fname = fs.feature_selection(idd, task, foldxy, df)
    b.dumpfile((foldnr, fl, mask, fname), f"{tmpdirectory}/fs_results/{idd}.json")


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
    for ftfile in os.listdir(f"{tmpdirectory}/fs_results"):
        foldnr, fl, mask, fname = b.loadfile(f"{tmpdirectory}/fs_results/{ftfile}")
        # Append the Featurelists to a dict with their fold number as key
        featurelists[foldnr].append((fl, mask, fname))
    tasks = rps.maketasks(featurelists, clfnames, randseed)
    numtasks = len(tasks)
    tasks.dump(f"{tmpdirectory}/rps_tasks.pkl")
    print(f"Created {numtasks} RPS tasks.")
    return numtasks


#############
# Random Parameter Search
#############


def calculate_rps(idd, n_jobs, debug):
    """Executes RPS for a given task. Executed by cluster.
    """
    task = np.load(f"{tmpdirectory}/rps_tasks.pkl", allow_pickle=True)[idd]
    foldxy = np.load(f"{tmpdirectory}/folds.pkl", allow_pickle=True)[task[0]] # task[0] = foldnr
    foldnr, scores, best_esti, ftlist, fname, y_labels = rps.random_param_search(task, foldxy, n_jobs, debug)
    best_esti = (type(best_esti).__name__, best_esti.get_params()) # Creates readable tuple that can be dumped.
    b.dumpfile([foldnr, scores, best_esti, ftlist, fname, y_labels], f"{tmpdirectory}/rps_results/{idd}.json")

def getresults():
    """Analyzes the result files in rps_results and
    returns only the ones with the best best_esti_score in each fold.
    """
    results = defaultdict(lambda: [[0]])
    for rfile in os.listdir(f"{tmpdirectory}/rps_results"):
        f = b.loadfile(f"{tmpdirectory}/rps_results/{rfile}")
        if f[1][0] > results[f[0]][0][0] or f[1][0] == -1: # Remove this == -1 part
            # For each fold the result with the best best_esti_score is saved
            # If the best_esti_score is -1 it means a set classifier was used.
            results[f[0]] = f[1:]
    b.dumpfile(results, f"results/results.json")


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
    np.array(folds, dtype=object).dump(f"{tmpdirectory}/folds.pkl")
    df.to_pickle(f"{tmpdirectory}/dataframe.pkl") # ?
    print("Skipping Featureselection using set Featurelist")
    featurelists = {}
    mask = [True if f in set_fl else False for f in df.columns]
    for foldnr in range(0, len(folds)):
        featurelists[foldnr] = [(set_fl, mask, "Set Featurelist")]
    tasks = rps.maketasks(featurelists, clfnames, randseed)
    numtasks = len(tasks)
    print(f"Created {numtasks} RPS tasks.")
    return numtasks, df


def makeall(use_rnaz, use_filters, selection_methods, clfnames, n_splits, numneg, randseed, debug, set_fl = []):
    from time import time
    starttime = time()
    # Load files
    print("Loading p and n files")
    p, n = load_pn_files(use_rnaz, use_filters, numneg, randseed, debug)

    if set_fl: # Skip Feature selection process
        rpstasklen, df = skip_feature_selection(set_fl, p, n, clfnames, n_splits, randseed)
    else:
        # FL tasks
        print(f"{time() - starttime}: Making Featurelist tasks...")
        fstasklen, df = makefltasks(p, n, selection_methods, n_splits, randseed, debug)
        # Calc FL part -> Cluster
        print(f"{time() - starttime}: Sending {fstasklen} FS tasks to cluster...")
        i = 0
        starttask = 1
        endtask = 75000
        while fstasklen > endtask: # This is neccessary because the cluster doesnt accept more than 75000 tasks at once.
            b.shexec_and_wait(f"qsub -V -t {starttask}-{endtask} runall_fs_sge.sh")
            print(f"Finished tasks {starttask}-{endtask}")
            i += 1
            starttask += 75000
            endtask += 75000
        b.shexec_and_wait(f"qsub -V -t {starttask}-{fstasklen} runall_fs_sge.sh")
        print(f"{time() - starttime}: ...Cluster finished")
        # RPS tasks
        print("Assembling FS lists and RPS tasks...")
        rpstasklen = gather_featurelists(clfnames, debug, randseed)
    if rpstasklen == 0:
        print("No RPS tasks were created. Possibly used wrong parameters.")
        return
    # Calc RPS part -> Cluster
    print(f"{time() - starttime}: Sending {rpstasklen} RPS tasks to cluster...")
    i = 0
    starttask = 1
    endtask = 75000
    while rpstasklen > endtask:
        b.shexec_and_wait(f"qsub -V -t {starttask}-{endtask} runall_rps_sge.sh")
        print(f"Finished tasks {starttask}-{endtask}")
        i += 1
        starttask += 75000
        endtask += 75000
    b.shexec_and_wait(f"qsub -V -t {starttask}-{rpstasklen} runall_rps_sge.sh")
    # Results
    print(f"{time() - starttime}: Gathering results...")
    getresults()
    # Write results in a readable file and save ROC and precision recall graphs
    b.shexec(f"python pig.py --results fel > results/output_results.txt")
    res.showresults("rp", "results/results.json", showplots=False)
    df.to_pickle("results/dataframe.pkl")
    print(f"{time() - starttime}: Done")


def create_directories():
    if not os.path.exists(f"{tmpdirectory}"):
        print(f"Creating {tmpdirectory} directory")
        os.makedirs(f"{tmpdirectory}")
    if not os.path.exists(f"{tmpdirectory}/fs_results"):
        print(f"Creating {tmpdirectory}/fs_results directory")
        os.makedirs(f"{tmpdirectory}/fs_results")
    if not os.path.exists(f"{tmpdirectory}/rps_results"):
        print(f"Creating {tmpdirectory}/rps_results directory")
        os.makedirs(f"{tmpdirectory}/rps_results")
    if not os.path.exists("results"):
        os.makedirs("results")

#############
# Main Function
#############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true', help='If used it will remove pn files before execution. So they need to be reloaded.')
    parser.add_argument('-b', '--blacklist', action='store_true', help='If selected the blacklist will be (re-)created first')
    parser.add_argument('--calcfl', type=int, help='Feature selection process for the cluster')
    parser.add_argument('--calcrps', type=int, help='Random Parameter Search process for the cluster')
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
    parser.add_argument('--random', nargs='+', type=int, default=[], help='Randomly select 40 features. Given number decides number of tasks.')
    parser.add_argument('--featurelist', nargs='+', type=str, default=[], help='A optional set featurelist. If this is not empty the feature selection methods will be ignored')
    parser.add_argument('--clf', nargs='+', type=str, default=['xtratrees', 'gradientboosting', 'neuralnet'], help='Either needs to be the name of a classifier or an executeable string that returns a classifier (in which case the inner cross validation is disabled).')
    parser.add_argument('-n', '--nsplits', type=int, default=5, help='Number of splits kfold creates')
    parser.add_argument('--numneg', type=int, default=10000, help='Number of negative (and max of positive) files beeing loaded')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random Seed used for execution')
    parser.add_argument('--results', type=str, default="", help='If used ignore all other arguments and show selected results (options: hfelrp)')

    args = vars(parser.parse_args())
    debug = args['debug']
    use_rnaz = args['rnaz']
    use_filters = args['filters']
    use_oversampling = args['oversample']
    selection_methods = {'Lasso': args['lasso'], 'VarThresh': args['varthresh'],
                         'SelKBest': args['kbest'], 'Relief': args['relief'],
                         'RFECV': args['rfecv'], 'SVC1': args['svc1'],
                         'SVC2': args['svc2'], 'Forest': args['forest'], 'Random': args['random']}
    clfnames = args['clf']
    n_splits = args['nsplits']
    randseed = args['seed']
    numneg = args['numneg'] # Number of negative files beeing read by b.loaddata()
    n_jobs = 24 # Number of parallel jobs used by RandomizedSearchCV
    set_fl = args['featurelist']

    if args['blacklist']:
        blacklist.create_blacklist("data")

    if use_oversampling: # Turns the used classifiers into their oversampling equivalents
        clfnames = ['os_' + s for s in clfnames]

    if args['results']: # Instead of executing other functions show previous results
        res.showresults(args['results'], "results/results.json", showplots=True)
    elif args['calcfl']:
        idd = args['calcfl'] - 1
        calculate_featurelists(idd)
    elif args['calcrps']:
        idd = args['calcrps'] - 1
        calculate_rps(idd, n_jobs, debug)
    elif len(selection_methods['Random']) > 1:
             raise ValueError("Using Random with more than one argument would lead to false results")
    else:
        cleanup(args['clean'])
        create_directories() # Create all needed directories
        if not (any(selection_methods.values()) or set_fl):
            print("No features or feature selection methods were given.")
            pass # No Selection method was given so just return
        else:
            makeall(use_rnaz, use_filters, selection_methods, clfnames, n_splits, numneg, randseed, debug, set_fl)
