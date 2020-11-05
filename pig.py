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
#tmpdirectory = "tmp"

def cleanup(keep_pn=True, keep_featurelists=False):
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
##        if keep_featurelists:
##            print("Kept featurelists from previous run")
##        else:
##            if os.path.exists(f"{tmpdirectory}/fs_results"):
##                shutil.rmtree(f"{tmpdirectory}/fs_results")
        if os.path.exists(f"{tmpdirectory}/task_results"):
            shutil.rmtree(f"{tmpdirectory}/task_results")
        for file in os.listdir(f"{tmpdirectory}"):
            if not (file.startswith("pn_")) or not keep_pn:
                if not (keep_featurelists and (file.startswith("fs_") or file.startswith("folds") or file.startswith("dataframe"))):
                    os.remove(f"{tmpdirectory}/{file}")
                    print(f"Removed {tmpdirectory}/{file}")



def create_directories():
    if not os.path.exists(f"{tmpdirectory}"):
        print(f"Creating {tmpdirectory} directory")
        os.makedirs(f"{tmpdirectory}")
    if not os.path.exists(f"{tmpdirectory}/task_results"):
        print(f"Creating {tmpdirectory}/task_results directory")
        os.makedirs(f"{tmpdirectory}/task_results")
    if not os.path.exists("results"):
        os.makedirs("results")

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


def makefolds(p, n, n_folds, randseed):
    allfeatures = list(p[1].keys())
    allfeatures.remove("name")  # We dont need the filenames (for now)
    X, Y, df = b.makeXY(allfeatures, p, n)
    X = StandardScaler().fit_transform(X)
    folds = kfold(X, Y, n_splits=n_folds, randseed=randseed)
    return folds, df

#############
# Make Featurelists
#############


#############
# Random Parameter Search
#############

def maketasks(p, n, fs_selection_methods, clfnames, n_folds, randseed, debug):
    """
    Creates and dumps tasks, dataframe and the folds created by kfold
    that are then read and executed by the cluster.

    Args:
      p (list): A list containing all the feature values
                for all the positive samples used.
      n (list): A list containing all the feature values
                for all the negative samples used.
      fs_selection_methods (dict): The dictionary of the Feature selection
                                   methods and their arguments.
      clfnames (list(string)): A list containing all the classifiernames used
                               for random parameter search or specific classifiers
                               to fit to.
      n_folds (int): Number of folds stratified K-Fold creates
      randseed (int): Randomseed used by the whole program
      debug (bool): Debug Mode (Might not actually do anything atm.)

    Returns:
      len(tasks): The total number of tasks.
    
    """
    tasks = []
    folds, df = makefolds(p, n, n_folds, randseed) #numfolds = n_splits
    for foldnr in range(n_folds):
        for clfname in clfnames:
            for fstype, parameters in fs_selection_methods.items():
                for args in parameters:
                    if fstype == "Random":
                        num_features, num_random_tasks = args
                        for seed in range(num_random_tasks): # Keep in mind this seed IS NOT randseed
                            tasks.append((foldnr, fstype, (num_features, seed), clfname, randseed))
                    elif fstype == "Forest" or fstype == "SVC1" or fstype == "SVC2":
                        tasks.append((foldnr, fstype, (randseed, args), clfname, randseed))
                    else:
                        tasks.append((foldnr, fstype, args, clfname, randseed))
    b.dumpfile(tasks, f"{tmpdirectory}/tasks.json")
    np.array(folds, dtype=object).dump(f"{tmpdirectory}/folds.pkl")
    df.to_pickle(f"{tmpdirectory}/dataframe.pkl")
    return len(tasks)


def make_set_fl_tasks(p, n, set_fl, clfnames, n_folds, randseed):
    """
    Similar to maketasks. Used if a set featurelist is used
    and the feature selection process is skipped.

    Args:
      p (list): A list containing all the feature values
                for all the positive samples used.
      n (list): A list containing all the feature values
                for all the negative samples used.
      set_fl (list): The feature list used.
      clfnames (list(string)): A list containing all the classifiernames used
                               for random parameter search or specific classifiers
                               to fit to.
      n_folds (int): Number of folds stratified K-Fold creates
      randseed (int): Randomseed used by the whole program
      debug (bool): Debug Mode (Might not actually do anything atm.)

    Returns:
      len(tasks): The total number of tasks.
    """
    set_fl_tasks = []
    folds, df = makefolds(p, n, n_folds, randseed)
    for foldnr in range(n_folds):
        for clfname in clfnames:
            set_fl_tasks.append((foldnr, clfname, randseed))
    b.dumpfile(set_fl, f"{tmpdirectory}/set_fl.json")
    b.dumpfile(set_fl_tasks, f"{tmpdirectory}/tasks.json")
    np.array(folds, dtype=object).dump(f"{tmpdirectory}/folds.pkl")
    df.to_pickle(f"{tmpdirectory}/dataframe.pkl")
    return len(set_fl_tasks)


def calculate(idd, n_jobs, debug):
    """Executes FS and RPS for a given task. Executed by cluster.

    Args:
      idd (int): Jobid. Used to find the right task
      n_jobs (int): Number of parallel jobs made by the
                    random parameter search. Does nothing otherwise.
    """
    # task = Foldnr, mask, clfname, ftlist, fname, randseed
    task = b.loadfile(f"{tmpdirectory}/tasks.json")[idd] # = (foldnr, fstype, args, clfname, randseed) or (foldnr, clfname, randseed)
    foldxy = np.load(f"{tmpdirectory}/folds.pkl", allow_pickle=True)[task[0]]
    df = pd.read_pickle(f"{tmpdirectory}/dataframe.pkl")
    if len(task) == 5: # Normal procedure with Feature Selection first.
        foldnr, fstype, args, clfname, randseed = task
        ftlist, mask, fname = fs.feature_selection(foldxy, fstype, args, df) # FS - Done.
    elif len(task) == 3: # A set featurelist was used.
        foldnr, clfname, randseed = task
        ftlist = b.loadfile(f"{tmpdirectory}/set_fl.json")
        mask = [True if f in ftlist else False for f in df.columns]
        fname = "Set Featurelist"
    else:
        raise ValueError("Incorrect number of arguments in the taskfile: {len(task)} should be 5 or 3")
    scores, best_esti, y_labels, coefs = rps.random_param_search(mask, clfname, foldxy, n_jobs, df, randseed, debug)######
    best_esti_params = best_esti.get_params()
    best_esti = (type(best_esti).__name__, best_esti_params) # Creates readable tuple that can be dumped.
    b.dumpfile([foldnr, scores, best_esti, ftlist, fname, y_labels], f"{tmpdirectory}/task_results/{idd}.json")
    #b.dumpfile(coefs, f"results/coef_{idd}.json")########### Only use this with single


def getresults():
    """Analyzes the result files in rps_results and
    returns only the ones with the best best_esti_score in each fold.
    """
    results = defaultdict(lambda: [[0]])
    for rfile in os.listdir(f"{tmpdirectory}/task_results"):
        f = b.loadfile(f"{tmpdirectory}/task_results/{rfile}")
        if f[1][0] > results[f[0]][0][0] or f[1][0] == -1: # Remove this == -1 part
            # For each fold the result with the best best_esti_score is saved
            # If the best_esti_score is -1 it means a set classifier was used.
            results[f[0]] = f[1:]
    b.dumpfile(results, f"results/results.json")


#############
# Additional Options
#############



def makeall(use_rnaz, use_filters, fs_selection_methods, clfnames, n_folds, numneg, randseed, debug, keep_featurelists, set_fl = []):
    """
    Note: keep_featurelists is no longer working after code has been restructured - TODO.
    """
    from time import time
    starttime = time()
    if keep_featurelists and set_fl:
        raise ValueError("Using --keepfl and --featurelist at once will lead to errors and is not allowed")
    # Load files
    print("Loading p and n files")
    p, n = load_pn_files(use_rnaz, use_filters, numneg, randseed, debug)

    if set_fl: # Skip Feature selection process
        tasklen = make_set_fl_tasks(p, n, set_fl, clfnames, n_folds, randseed)
        print("Skip feature selection using set featurelist")
    else:
        tasklen = maketasks(p, n, fs_selection_methods, clfnames, n_folds, randseed, debug)
    if tasklen == 0:
        print("No tasks were created. Possibly used wrong parameters.")
        return
    # Calculation part -> Cluster
    print(f"{time() - starttime}: Sending {tasklen} tasks to cluster...")
    i = 0
    starttask = 1
    endtask = 75000
    while tasklen > endtask:
        b.shexec_and_wait(f"qsub -V -t {starttask}-{endtask} runall_tasks_sge.sh")
        print(f"Finished tasks {starttask}-{endtask}")
        i += 1
        starttask += 75000
        endtask += 75000
    b.shexec_and_wait(f"qsub -V -t {starttask}-{tasklen} runall_tasks_sge.sh")
    # Results
    print(f"{time() - starttime}: Gathering results...")
    getresults()
    # Write results in a readable file and save ROC and precision recall graphs
    b.shexec(f"python pig.py --results fel > results/output_results.txt")
    res.showresults("rp", "results/results.json", showplots=False)
    print(f"{time() - starttime}: Done")

#############
# Main Function
#############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_false', help='If used it will remove pn files before execution. So they need to be reloaded.')
    parser.add_argument('-b', '--blacklist', action='store_true', help='If selected the blacklist will be (re-)created first')
    parser.add_argument('--calc', type=int, help='Task execution. Used for the cluster')
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
    parser.add_argument('--random', nargs=2, type=int, default=[], help='Must be exactly 2 numbers X, Y. X is number of features per random and Y the number of different random selections.')
    parser.add_argument('--featurelist', nargs=1, type=str, default="", help='A optional set featurelist. If this is not empty the feature selection methods will be ignored')
    parser.add_argument('--clf', nargs='+', type=str, default=['gradientboosting'], help='Either needs to be the name of a classifier (xtratrees, gradientboosting, neuralnet) or an executeable string that returns a classifier (in which case the inner cross validation is disabled).')
    parser.add_argument('-n', '--nfolds', type=int, default=5, help='Number of folds kfold creates')
    parser.add_argument('--numneg', type=int, default=10000, help='Number of negative (and max of positive) files beeing loaded')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random Seed used for execution')
    parser.add_argument('--results', type=str, default="", help='If used ignore all other arguments and show selected results (options: hfelrp)')
    parser.add_argument('--keepfl', action='store_true', help='HANDLE WITH CARE: If used, the existing featurelist solution from previous runs will not be deleted and instead be used for this run to speed up the process. Cant be used together with --featurelists')

    args = vars(parser.parse_args())
    debug = args['debug']
    use_rnaz = args['rnaz']
    use_filters = args['filters']
    use_oversampling = args['oversample']
    if args['random']:
        randargs = [(args['random'][0], args['random'][1])]
    else:
        randargs = []
    fs_selection_methods = {'Lasso': args['lasso'], 'VarThresh': args['varthresh'],
                         'SelKBest': args['kbest'], 'Relief': args['relief'],
                         'RFECV': args['rfecv'], 'SVC1': args['svc1'],
                         'SVC2': args['svc2'], 'Forest': args['forest'], 'Random': randargs}
    clfnames = args['clf']
    n_folds = args['nfolds']
    randseed = args['seed']
    numneg = args['numneg'] # Number of negative files beeing read by b.loaddata()
    n_jobs = 24 # Number of parallel jobs used by RandomizedSearchCV
    if args['featurelist']:
        set_fl = args['featurelist'][0].strip("'[]").split("', '")
    else:
        set_fl = []

    if args['blacklist']:
        blacklist.create_blacklist("data")

    if use_oversampling: # Turns the used classifiers into their oversampling equivalents
        clfnames = ['os_' + s for s in clfnames]

    if args['results']: # Instead of executing other functions show previous results
        res.showresults(args['results'], "results/results.json", showplots=True)
    elif args['calc']:
        idd = args['calc'] - 1
        calculate(idd, n_jobs, debug)
    elif len(fs_selection_methods['Random']) > 1:
             raise ValueError("Using Random with more than one argument would lead to false results")
    else:
        cleanup(args['clean'], args['keepfl']) # Remove data from previous runs
        create_directories() # Create all needed directories
        if not (any(fs_selection_methods.values()) or set_fl or args['keepfl']):
            print("No features or feature selection methods were given.")
            pass # No Selection method was given so just return
        else:
            makeall(use_rnaz, use_filters, fs_selection_methods, clfnames, n_folds, numneg, randseed, debug, args['keepfl'], set_fl)
