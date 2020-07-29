from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from numpy import reshape, concatenate
from numpy import chararray, unique
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from other.draw import matrix
import json

def log_reg(pos, neg, conf_matrix = True):
    """Builds a Logistic Regression model using the Yao-score
       of the given sample data and returns the Cross validation score.
    Args:
      pos (Array[String]): An array of paths to proven positive yao-scores.
      neg (Array[String]): An array of paths to proven negative yao-scores.
      conf_matrix (Bool): If True: Draw confusion matrix

    Returns:
      cv (ndarray[float64]): The Cross validation score
    """
    X = []
    for i in neg:
        with open(i) as negfile:
            X += json.load(negfile).values()
    negf = len(X)
    for j in pos:
        with open(j) as posfile:
            X += json.load(posfile).values()
    X = reshape(X, (-1, 1))
    posf = len(X) - negf
    y = [0]*negf + [1]*posf
    lr = LogisticRegression(class_weight="balanced")
    if conf_matrix:
        lr.fit(X,y)
        predictions = lr.predict(X)
        matrix(y, predictions, reshape([0,1], (-1,1)))
    cv = cross_val_score(lr, chararray.astype(X, float), y, cv=3, scoring="f1")
    return cv 


if __name__ == "__main__":
    cv = log_reg(["data/yaoscores/pos.json", "data/yaoscores/pos2.json"],
                 ["data/yaoscores/neg.json"])
