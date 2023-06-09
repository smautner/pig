
from  sklearn.neighbors import KNeighborsClassifier
import numpy as np

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

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score

def kmeans_ari(X,y,k=20):
    means = KMeans(n_clusters = k)
    predicted = means.fit_predict(X)
    return adjusted_rand_score(y,predicted)



from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import logging
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
