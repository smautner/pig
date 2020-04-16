from sklearn.linear_model import LogisticRegression
from math import sqrt, log
from loadfiles import loaddata


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

def log_reg(pos, neg):
    """Builds a Logistic Regression model using the Yao-score
       of the given sample data as the feature and returns it.
    Args:
      pos (Array[String]): An array of paths to proven positive yao-scores.
      neg (Array[String]): An array of paths to proven negative yao-scores.

    Returns:
      lr (LogisticRegression): The trained model.
    """
    yao = []
    posf, negf = 0, 0  # Number of positive and negative files
    for x in neg:
        with open(x) as negfile:
            for line in negfile:
                yao.append([line])
                negf += 1
    for y in pos:
        with open(y) as posfile:
            for line in posfile:
                yao.append([line])
                posf += 1
    states = [0]*negf + [1]*posf
    lr = LogisticRegression().fit(yao, states)
    return lr


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
    import os
    if output == "":
        output = "yaoscores"
        os.system("mkdir -p " + output)
    else:
        output = output + "/yaoscores"
        os.system("mkdir -p " + output)
    for directory in os.listdir(data):
        for filename in os.listdir(data + "/" + directory):
            os.system(path + " --summarize --summarize-gsc \
                      --summarize-no-de-tag --fragmentary " + data + "/" +
                      directory + "/" + filename +
                      " | grep -oP '(?<=yaoFormulaScore=)[0-9]+.[0-9]+' >> " +
                      output + "/" + directory)


if __name__ == "__main__":
    # execute_cmfinder04() #-- Takes a LONG time
    lr = log_reg(["yaoscores/pos", "yaoscores/pos2"], ["yaoscores/neg"])
