from  sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import logging
import ubergauss.tools as ut


def _repeat_as_column(a,n):
    return np.tile(a,(n,1)).T

def knn_accuracy(X, y,n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors+1)

    y_train= np.array(y)
    knn.fit(X,y)

    # Get the labels of the nearest neighbors for each training point
    _, indices = knn.kneighbors(X)
    # instances -> labels
    neighbor_labels = y_train[indices]
    # Exclude the label of the training point itself
    neighbor_labels = neighbor_labels[:, 1:]

    # Compute the training error
    agreement = (_repeat_as_column(y_train,n_neighbors) == neighbor_labels).mean()

    return agreement


def kmeans_ari(X,y,k=20):
    means = KMeans(n_clusters = k)
    predicted = means.fit_predict(X)
    return adjusted_rand_score(y,predicted)




def knn_f1(X,y,n_neighbors = 3,cv_strati_splits = 3):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    skf = StratifiedKFold(n_splits=cv_strati_splits, random_state=23, shuffle=True)
    y= np.array(y)
    train_test = list(skf.split(y,y))

    real = []
    pred = []
    for train, test in train_test:
        knn.fit(X[train],y[train])
        pred += list(knn.predict(X[test]))
        real += list(y[test])

    logging.captureWarnings(True)
    logging.captureWarnings(False)
    calc_jointly =  f1_score(real,pred,average=f'weighted')
    # mean of f1 scores as in rnascclust paper
    return np.mean( cross_val_score(knn, X, y, cv=skf, scoring=f'f1_weighted', n_jobs=1))


def overlap_coef(csr_a, csr_b):
    score =  len(np.intersect1d(csr_a.indices, csr_b.indices)) / min(len(csr_a.indices), len(csr_b.indices))
    # score = np.intersect1d(a, b)/ min(len(a), len(b))
    return 1-score

def overlap(csrs):
    s = csrs.shape[0]
    m = np.zeros((s,s))
    for i in range(s):
        for j in range(i,s):
            m[i,j] = m[j,i] =overlap_coef(csrs[i], csrs[j])
    return m

def overlap2(csrs):
    s = csrs.shape[0]
    def dorow(i):
        m = np.zeros(s)
        for j in range(i,s):
            m[j] = overlap_coef(csrs[i], csrs[j])
        return m

    return np.vstack(ut.xmap(dorow,range(s)))


import sklearn
def permutation_score(X,y):
    estimator = sklearn.cluster.KMeans(n_clusters = len(np.unique(y)))
    score = sklearn.model_selection.permutation_test_score(estimator, X, y, cv = 2)
    return score[2] # this is the pvalue
