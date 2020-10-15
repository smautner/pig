import os
import sys
from shutil import move
import numpy as np
import input.basics as b
import input.showresults as res
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pprint import pprint

directory = "results/runs/10-fold-10000"
filetype = "results.json"
seed = "42"
label = ""
figsize = (12.8, 9.6)
plt.rcParams.update({'font.size': 22})

def draw_precision_recall(directory, filetype):
    plt.figure(figsize=figsize)
    for methodname in os.listdir(directory):
        if not methodname == seed:
            continue
        for filename in os.listdir(f"{directory}/{methodname}"):
            if filename.startswith(filetype):
                y_true, y_score = [], []
                for sc, be, ft, fn, y_labels in b.loadfile(f"{directory}/{methodname}/{filename}").values():
                    y_true.extend(y_labels[0])
                    y_score.extend(y_labels[1])
                precision, recall, thresholds = precision_recall_curve(y_true, y_score)
                plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.savefig("results/precision_recall")
    plt.show()

def draw_roc(directory, filetype):
    plt.figure(figsize=figsize)
    for methodname in os.listdir(directory):
        for filename in os.listdir(f"{directory}/{methodname}"):
            if filename.startswith(filetype):
                y_true, y_score = [], []
                for sc, be, ft, fn, y_labels in b.loadfile(f"{directory}/{methodname}/{filename}").values():
                    y_true.extend(y_labels[0])
                    y_score.extend(y_labels[1])
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                auc = roc_auc_score(y_true, y_score)
                plt.plot(fpr, tpr, label=f"{methodname} - {round(auc, 4)}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.savefig("results/roc")
    plt.show()

def plotall_precision_recall(new_plots_dir, exclude_string=None):
    plt.figure(figsize=figsize)
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
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.savefig(f"{new_plots_dir}/all_precision_recall")
    plt.show()

def plotall_roc(old_plots_dir, new_plots_dir, exclude_string=None):
    plt.figure(figsize=figsize)
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
    for filename in os.listdir(old_plots_dir):
        if filename.startswith("roc-all"):
            with open(f"{old_plots_dir}/{filename}") as file:
                x, y = [], []
                for line in file:
                    sp = line.split()[:2]
                    x.append(float(sp[0]))
                    y.append(float(sp[1]))
                plt.plot(y, x, label=f"{filename}")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.savefig(f"{new_plots_dir}/all_roc")
    plt.show()

def print_best_features():
    directory = "results/runs/testing/svc_gradientboosting_only"
    resultslist = []
    c = Counter()
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            for sc, be, ftlist, fn, y_labels in b.loadfile(f"{directory}/{filename}").values():
                c.update(ftlist)
    pprint(c.most_common())
    print([item[0] for item in c.most_common()])

def check_featurelists(numrandomtasks = 10000):
    tmpdirectory = "/scratch/bi01/mautner/guest10/tmp"
    numfolds = 7
    d = defaultdict(list)
    for ftfile in os.listdir(f"{tmpdirectory}/fs_results"):
        foldnr, fl, mask, fname = b.loadfile(f"{tmpdirectory}/fs_results/{ftfile}")
        taskid = int(ftfile.split(".")[0])
        d[taskid % numrandomtasks] += [fl]
    for i in range(0, len(d)):
        if all(elem == d[i][0] for elem in d[i]):
            print("YAY")
        else:
            print("NU")
            for x in d[i]:
                print(x)

def getresults2(numrandomtasks = 10000):
    """
    taskid % 10000 => 1-7 => F1 von 7 zusammen
    list F1 scores => 10000 elemente
    list features = 10000 featurelisten
    Beste Featureliste zurück geben
    Histogram über alle 10000 F1 scores
    """
    results = []
    score_d = defaultdict(list)
    featurelist_d = defaultdict(list)
    tmpdirectory = "/scratch/bi01/mautner/guest10/tmp"
    i = 0#####
    j = 0######
    for rfile in os.listdir(f"{tmpdirectory}/rps_results"):
        taskid = int(rfile.split(".")[0])
        f = b.loadfile(f"{tmpdirectory}/rps_results/{rfile}")
        scores = f[1]
        fl = f[3]
        tpr, precision = scores[2][0], scores[2][2]
        if np.isnan(precision):
            i += 1########
        score_d[taskid % numrandomtasks].append((tpr, precision))
        # 10.000 Dictionary Entries mit 7 score tuples
        featurelist_d[taskid % numrandomtasks] = fl 
        # 10.000 verschiedene Featurelists
    

    print(f"SCORE_D: {len(score_d)}")
    print(f"FEATURELIST_D: {len(featurelist_d)}")
    print(f"RIPCOUNTER: {i}")
    


    # Calculate average F1-Scores of each entry
    avg_tpr, avg_precision = 0, 0
    best_f1_score = 0
    best_key = 0
    f1_list = [] # Used for Histogram
    for key in score_d:
        for tpr, precision in score_d[key]:
            avg_tpr += tpr
            avg_precision += precision
        avg_tpr, avg_precision = avg_tpr/len(score_d[key]), avg_precision/len(score_d[key])
        f1 = 2*((avg_precision*avg_tpr)/(avg_precision+avg_tpr))
        if np.isnan(f1):
            j += 1
        f1_list.append(f1)
        if f1 > best_f1_score: # return best of 10000 average F1-Scores
            best_f1_score = f1
            best_key = key
    print(f"F1_LIST: {len(f1_list)}")
    print(f"RIPCOUNTER 2: {j}")

    best_featurelist = featurelist_d[best_key]
    #results[best_file[0]] = best_file[1:]
    #b.dumpfile(results, f"results/results.json")
    print(best_featurelist)
    b.dumpfile(best_featurelist, f"results/best_featurelist.json")

    fontsize=18 # Size of the text for the labels and legend.
    plt.figure(figsize=(12.8, 9.6))
    plt.xlabel("F1-Score", fontsize=fontsize)
    plt.ylabel("Number of Scores", fontsize=fontsize)
    plt.hist(f1_list, bins=100)
    plt.savefig("results/f1_histogram.png")


if __name__ == "__main__":
    #draw_roc(directory, filetype)
    #draw_precision_recall(directory, filetype)
    #res.showresults("e", "results/results.json")
    getresults2(10000)


    old_plots_dir = "results/runs/old_plots"
    new_plots_dir = "results/runs/new"
    if False: # Draw ROC and Precision/Recall for each results file
        runs = new_plots_dir
        for runname in os.listdir(runs):
            if not os.path.isdir(f"{runs}/{runname}"):
                continue
            path = f"{runs}/{runname}"
            print(f"----- {runname} -----")
            #res.showresults("e", f"{path}/results.json")
            orig_stdout = sys.stdout # Write Output in a file instead of printing it
            f = open(f"{path}/output_results.txt", "w")
            sys.stdout = f
            res.showresults("fel", f"{path}/results.json")
            sys.stdout = orig_stdout
            res.showresults("rp", f"{path}/results.json", showplots=False)
            move("results/results_roc.png", f"{path}/results_roc.png")
            move("results/results_precision_recall.png", f"{path}/results_precision_recall.png")
        #print_best_features()

    elif False: # Draw all ROC curves in a single file
        plotall_roc(old_plots_dir, new_plots_dir, exclude_string="neural")

    elif False: # Draw all Precision-Recall curves in a single file
        plotall_precision_recall(new_plots_dir, exclude_string="neural")

