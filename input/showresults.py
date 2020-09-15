from collections import Counter, defaultdict
from pprint import pprint
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import input.basics as b


###################
# Printing out Results
###################

def showresults(args, resultfile="results/results.json"):
    results = b.loadfile(resultfile)
    estimators = defaultdict(lambda: defaultdict(list))
    ftlists = []
    c = Counter()
    y_true = []
    y_score = []
    for scores, best_esti, ftlist, fname, y_labels in results.values():
        esti_name, params = best_esti
        best_esti_score, test_score, accuracy_score = scores
        if best_esti_score == -1:
            params["best_esti_score"] = None
        else:
            params["best_esti_score"] = round(best_esti_score, 4)
        params["test_score"] = round(test_score, 4)
        params["accuracy_score"] = [round(acc, 4) for acc in accuracy_score]
        for key, value in params.items():
            estimators[esti_name][key].append(value)
        ftlists.append((fname, ftlist)) # ?
        c.update(ftlist)
        y_true.extend(y_labels[0])
        y_score.extend(y_labels[1])
    if "f" in args:
        pprint(c.most_common())
        print("\n")
    if "e" in args:
        for key in estimators.keys():
            print(f"{key}:")
            print("-" * (len(key)+1))
            for param in estimators[key].items():
                print(f"{param[0]}: {param[1]}")
            print("\n")
    if "n" in args:
        for x in ftlists:
            pprint((x[0], len(x[1]), sorted(x[1])))
    if "r" in args:
        plt.figure(figsize=(12.8, 9.6))
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=f"{round(auc, 4)}")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='best')
        plt.savefig("results/results_roc")
        plt.show()
    if "p" in args:
        plt.figure(figsize=(12.8, 9.6))
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig("results/results_precision_recall")
        plt.show()
    if  "h" in args:
        print("Usage: pig.py -r {fenrp}\n", \
              "f - featurelists with number of occurences\n", \
              "e - estimators\n", \
              "n - Shows ALL featurelists with the info used to create them\n", \
              "r - Creates and plots the roc_curve\n", \
              "p - Creates and plots the precision_recall_curve")

