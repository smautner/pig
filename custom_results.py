import os
import argparse
import numpy as np
import input.basics as b
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter


def getresults2(numrandomtasks=10000, n_best=10, tmpdirectory="tmp"):
    """
    Calculates the average F1 scores random featurelist over every fold.
    Collects these into a histogram and dumps them into the results directory.
    Also takes the n_best Featurelists with the best average F1-Scores and dumps them too.

    Example:
      - Executed pig.py --random 40 10000 -n 7
      => Results in 70000 files with 40 features each.
      File 0-9999 beeing 10000 different featurelists for the 1. fold.
      Files 10000-19999 beeing the same 10000 featurelists for the 2. fold.
      ...
      - This code will take taskid % numrandomtasks and calculate the
        average F1 score for each of these featurelists over every fold.
      => Takes 7 files each and calculates their average F1 score
      => Results in 10000 scores in total.

    Args:
      numrandomtasks(int): Needs to be the same number as the 2. argument of
                           --random in pig.py, so the number of
                           random featurelists per fold.
      n_best(int): Number of best featurelists that should be saved seperately
      tmpdirectory(String): Location of the "tmp" directory.
    """
    score_d = defaultdict(list)
    avg_score_d = defaultdict(list)
    featurelist_d = defaultdict(list)

    i = 0
    j = 0
    for rfile in os.listdir(f"{tmpdirectory}/task_results"):
        taskid = int(rfile.split(".")[0])
        f = b.loadfile(f"{tmpdirectory}/task_results/{rfile}")
        scores = f[1]
        fl = f[3] # Featurelist
        tpr, precision = scores[2][0], scores[2][2]
        if np.isnan(precision):
            i += 1
        score_d[taskid % numrandomtasks].append((tpr, precision))
        # 10.000 Dictionary Entries mit 7 score tuples
        featurelist_d[taskid % numrandomtasks] = fl
        # 10.000 different Featurelists

    # Calculate average F1-Scores of each entry
    best_f1_score = 0
    best_key = 0
    f1_list = [] # Used for Histogram
    for key in score_d:
        sum_tpr, sum_precision = 0, 0
        for tpr, precision in score_d[key]:
            sum_tpr += tpr
            sum_precision += precision
        avg_tpr, avg_precision = sum_tpr/len(score_d[key]), sum_precision/len(score_d[key])
        f1 = 2*((avg_precision*avg_tpr)/(avg_precision+avg_tpr))
        if np.isnan(f1):
            j += 1
        f1_list.append(f1)
        avg_score_d[key] = f1

    # Get the best n_best featurelists
    best_featurelists = {}
    for key in dict(sorted(avg_score_d.items(), key = itemgetter(1), reverse = True)[:n_best]).keys():
        best_featurelists[key] = (avg_score_d[key], featurelist_d[key])
    b.dumpfile(best_featurelists, f"results/best_featurelists.json")

    # Draw the histogram
    fontsize=18 # Size of the text for the labels and legend.
    plt.figure(figsize=(12.8, 9.6))
    plt.xlabel("F1-Score", fontsize=fontsize)
    plt.ylabel("Number of Scores", fontsize=fontsize)
    plt.hist(f1_list, bins=100)
    plt.savefig("results/f1_histogram.png")


def plotall_roc(new_plots_dir, old_plots_dir, exclude_string=None):
    """Experimental function that allows to easily draw ROC figures from different runs into a single file.
    Optionally can also draw plots from specific older files.
    Note that these older files need to have a very specific file format.
    (This function was only added for convenience use to compare results of pig.py with different programs)

    Args:
      new_plots_dir(string): Directory of runs to plot. This directory should
                             contain subdirectories with their runnames as
                             directorynames and those need to contain a "results.json" file.
                             Example: new_plots_dir/gradientboosting42/results.json
      old_plots_dir(string): Directory containing old roc files. Those files should start with "roc-all."
      exclude_string(string): If existing it will ignore subdirectories that contain this string in their name.
                              Example: If you have gradientboosting and neuralnet runs set it to "neural"
                                       to only plot gradientboosting runs.

    Returns:
      Nothing but saves the resulting ROC figure into the new_plots_dir.
    """
    fontsize=18 # Size of the text for the labels and legend.
    plt.figure(figsize=(12.8, 9.6))
    plt.plot([0, 1], [0, 1], 'k--')
    for dirname in os.listdir(new_plots_dir):
        if not os.path.isdir(f"{new_plots_dir}/{dirname}"):
            continue
        elif not exclude_string == None and exclude_string in dirname:
            continue
        for filename in os.listdir(f"{new_plots_dir}/{dirname}"):
            if filename.startswith("results.json"):
                y_true, y_score = [], []
                for sc, be, ft, fn, y_labels in b.loadfile(f"{new_plots_dir}/{dirname}/{filename}").values():
                    y_true.extend(y_labels[0])
                    y_score.extend(y_labels[1])
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                auc = roc_auc_score(y_true, y_score)
                plt.plot(fpr, tpr, label=f"{dirname} - {round(auc,4)}")
    if old_plots_dir:
        for filename in os.listdir(old_plots_dir):
            if filename.startswith("roc-all"):
                with open(f"{old_plots_dir}/{filename}") as file:
                    x, y = [], []
                    for line in file:
                        sp = line.split()[:2]
                        x.append(float(sp[0]))
                        y.append(float(sp[1]))
                    plt.plot(y, x, label=f"{filename}")
    plt.xlabel('False positive rate', fontsize=fontsize)
    plt.ylabel('True positive rate', fontsize=fontsize)
    plt.legend(loc='best')
    plt.savefig(f"{new_plots_dir}/all_roc")
    plt.show()


def plotall_precision_recall(new_plots_dir, exclude_string=None):
    """Similar to plotall_roc() but for precision-recall curves.
    Does not include the option to include files from different programs.
    The rest of the comments are the same.
    """
    fontsize=18 # Size of the text for the labels and legend.
    plt.figure(figsize=(12.8, 9.6))
    for dirname in os.listdir(new_plots_dir):
        if not os.path.isdir(f"{new_plots_dir}/{dirname}"):
            continue
        elif not exclude_string == None and exclude_string in dirname:
            continue
        for filename in os.listdir(f"{new_plots_dir}/{dirname}"):
            if filename.startswith("results.json"):
                y_true, y_score = [], []
                for sc, be, ft, fn, y_labels in b.loadfile(f"{new_plots_dir}/{dirname}/{filename}").values():
                    y_true.extend(y_labels[0])
                    y_score.extend(y_labels[1])
                precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                plt.plot(recall, precision, label=f"{dirname}")
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.legend(loc='best')
    plt.savefig(f"{new_plots_dir}/all_precision_recall")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--histo', action='store_true', help='Draws F1-histogram after executing pig.py with the '
                                                             '--random method.')
    parser.add_argument('--numrandom', nargs=1, type=int, default=10000, help='Used for --histo. Needs to be the same '
                                                                              'number as the 2. argument of --random '
                                                                              'in pig.py')
    parser.add_argument('--numbest', nargs=1, type=int, default=10, help='Used for --histo. Sets the amount of best '
                                                                         'featurelists stored.')
    parser.add_argument('--tmp', nargs=1, type=str, default="tmp", help='Used for --histo. Location of the tmp '
                                                                        'directory from pig.py')
    args = vars(parser.parse_args())

    tmpdirectory = "/scratch/bi01/mautner/guest10/tmp" ##########

    if args['histo']:
        numrandomtasks = args['numrandom'][0]
        n_best = args['numbest'][0]
        tmpdirectory = args['tmp'][0]
        getresults2(numrandomtasks, n_best, tmpdirectory)


    if False: # Draw all ROC curves in a single file
        old_plots_dir = "results/runs/old_plots"
        new_plots_dir = "results/runs/new"
        plotall_roc(new_plots_dir, old_plots_dir, exclude_string="neural")

    if False: # Draw all Precision-Recall curves in a single file
        new_plots_dir = "results/runs/new"
        plotall_precision_recall(new_plots_dir, exclude_string="neural")


