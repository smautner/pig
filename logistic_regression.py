from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from numpy import reshape, concatenate
from numpy import chararray
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
#from loadfiles import loaddata


# Currently not needed:
# def save_loadeddata(p, n):
#     """saves with loaddata() loaded data as a JSON file."""
#     import json
#     with open("p_loaded_data.json", "w") as f:
#         json.dump(p, f)
#     with open("n_loaded_data.json", "w") as f:
#         json.dump(n, f)
#
# def load_loadeddata():
#     """loads with save_loadeddata() saved files
#     and returns its p and n again."""
#     import json
#     with open("p_loaded_data.json", "r") as f:
#         p = json.load(f)
#     with open("n_loaded_data.json", "r") as f:
#         n = json.load(f)
#     return p, n

def execute_cmfinder04(path="cmfinder-0.4.1.18/bin/cmfinder04",
                       data="data", output="", overwrite=False):
    """Executes the cmfinder program to calculate the yao-score
       with all samples in the given data folder.
    Args:
      path(String): The path for the cmfinder04 tool.
      data(String): The path of the datafolder.
      output(String): Path for the output files.
      overwrite(Bool): Set True if existing output files should be overwritten.
    """
    if output == "":
        output = "yaoscores"
        os.system("mkdir -p " + output)
    else:
        output = output + "/yaoscores"
        os.system("mkdir -p " + output)
    for directory in os.listdir(data):
        if overwrite == True:
            os.remove(output + "/" + directory)
        for filename in os.listdir(data + "/" + directory):
            os.system(path + " --summarize --summarize-gsc \
                      --summarize-no-de-tag --fragmentary " + data + "/" +
                      directory + "/" + filename +
                      " | grep -oP '(?<=yaoFormulaScore=)[0-9]+.[0-9]+' >> " +
                      output + "/" + directory)


def log_reg(pos, neg):
    """Builds a Logistic Regression model using the Yao-score
       of the given sample data as the feature and returns it.
    Args:
      pos (Array[String]): An array of paths to proven positive yao-scores.
      neg (Array[String]): An array of paths to proven negative yao-scores.

    Returns:
      lr (LogisticRegression): The trained model.
    """
    X = []
    for i in neg:
        with open(i) as negfile:
            X = reshape(negfile.read().split(), (-1, 1))
    negf = len(X)
    for j in pos:
        with open(j) as posfile:
            X = concatenate((X, reshape(posfile.read().split(), (-1, 1))))
    posf = len(X) - negf
    y = [0]*negf + [1]*posf
    lr = LogisticRegression().fit(X, y)
    # X_scaled = StandardScaler().fit_transform(X)
    cv = cross_val_score(lr, chararray.astype(X, float), y)
    return lr, cv 


if __name__ == "__main__":
    # execute_cmfinder04() #-- Takes a LONG time
    lr, cv = log_reg(["yaoscores/pos", "yaoscores/pos2"], ["yaoscores/neg"])
    # test_ali("data/pos/8-1281-0-1.sto", lr)
