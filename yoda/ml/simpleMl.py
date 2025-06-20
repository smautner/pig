from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RF
import sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import logging
import ubergauss.tools as ut
from scipy.sparse import csr_matrix

def _repeat_as_column(a,n):
    return np.tile(a,(n,1)).T

def knn_accuracy(X, y,n_neighbors=1, select_labels=[]):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors+1, n_jobs = 1)
    y_train= np.array(y)

    # X -= X.min()
    # np.fill_diagonal(X,0)
    knn.fit(X,y)
    # Get the labels of the nearest neighbors for each training point
    _, indices = knn.kneighbors(X)
    # instances -> labels
    neighbor_labels = y_train[indices]
    # Exclude the label of the training point itself
    neighbor_labels = neighbor_labels[:, 1:]  #!!!!!!!!!!!!!!!!!!!!!!!!!11
    # Compute the training error
    if select_labels:
        mask = [y in select_labels for y in y_train]
        y_train = y_train[mask]
        neighbor_labels = neighbor_labels[mask]
    agreement = (_repeat_as_column(y_train,n_neighbors) == neighbor_labels).mean()
    return agreement

def kmeans_ari(X,y,k=0):
    if k == 0:
        k = len(np.unique(y))
    means = KMeans(n_clusters = k)
    predicted = means.fit_predict(X)
    return adjusted_rand_score(y,predicted)

from yoda import ml
from yoda import draw

# def kmeans_bla(dist,labels):
#     X = ml.embed(dist, n_dim = 2)
#     draw.scatter(X,labels)
#     # adjusted_rand_score( KMeans(n_clusters=len(np.unique(labels))).fit_predict(X), labels)
#     return keams_ari(X, labels,k=0)


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


def overlap_coef(csr_a, csr_b, nonorm=False):
    norm = 1 if nonorm else min(len(csr_a.indices), len(csr_b.indices))
    score =  len(np.intersect1d(csr_a.indices, csr_b.indices)) / norm
    # score = np.intersect1d(a, b)/ min(len(a), len(b))
    return 1-score

def overlap(csrs, nonorm =False):
    s = csrs.shape[0]
    m = np.zeros((s,s))
    for i in range(s):
        for j in range(i,s):
            m[i,j] = m[j,i] =overlap_coef(csrs[i], csrs[j],nonorm=nonorm)
    return m

def overlap2(csrs):
    s = csrs.shape[0]
    def dorow(i):
        m = np.zeros(s)
        for j in range(i,s):
            m[j] = overlap_coef(csrs[i], csrs[j])
        return m

    return np.vstack(ut.xmap(dorow,range(s)))

def featuremask(X,y,n_ft = 100):
    # clf = RF(n_estimators = 100,max_features=None).fit(X,y)
    clf = RF(n_estimators = 100).fit(X,y)
    featscores =  clf.feature_importances_
    # print(f"{ featscores=}")
    return ut.binarize(featscores, n_ft)

def permutation_score(X,y):
    estimator = sklearn.cluster.KMeans(n_clusters = len(np.unique(y)))
    score = sklearn.model_selection.permutation_test_score(estimator, X, y, cv = 2)
    return score[2] # this is the pvalue


def clan_in_x_OLD(X, y,n_neighbors=10):
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
    # agreement = (_repeat_as_column(y_train,n_neighbors) == neighbor_labels).mean()
    return np.mean([y in row for y,row in zip(y_train,neighbor_labels)  ])


def clan_in_x(X, y,n_neighbors=10):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors+1, metric = 'precomputed')
    y_train= np.array(y)
    knn.fit(X,y)
    # Get the labels of the nearest neighbors for each training point
    _, indices = knn.kneighbors(X)
    # instances -> labels
    neighbor_labels = y_train[indices]
    # Exclude the label of the training point itself
    neighbor_labels = neighbor_labels[:, 1:]
    # Compute the training error
    # agreement = (_repeat_as_column(y_train,n_neighbors) == neighbor_labels).mean()
    return np.mean([y in row for y,row in zip(y_train,neighbor_labels)  ])

def accs(ind,y):
    '''
    neigbor indices and y -> list of contains neigh for 1..ind.shape[1]
    '''
    neighbor_labels = y[ind]
    return [ np.mean([yy in row for yy,row in zip(y,neighbor_labels[:,1:n])  ]) for n in range(1,ind.shape[1])]

def clan_in_x_corrected(X,y,n_neighbors=10):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors+1, metric = 'precomputed')
    y_train= np.array(y)
    knn.fit(X,y)
    # Get the labels of the nearest neighbors for each training point
    _, indices = knn.kneighbors(X)
    # instances -> labels
    neighbor_labels = y_train[indices]
    # Exclude the label of the training point itself
    neighbor_labels = neighbor_labels[:, 1:]
    # Compute the training error
    # agreement = (_repeat_as_column(y_train,n_neighbors) == neighbor_labels).mean()
    return np.mean([y in row for y,row in zip(y_train,neighbor_labels)  ])

from sklearn.metrics import average_precision_score

def average_precision(distances,y, skip = set()):
    '''
    distances: np.array of shape (n_samples, n_samples)
    y: np.array of shape (n_samples,)
    this function computes the average precision of the distances


    skipping is for the test set, where clans have been extended in rf15
    '''
    def score(i):
        mask = np.arange(len(y)) != i
        y_true = (y[mask] == y[i])
        y_scores = -distances[i, mask]
        return average_precision_score(y_true, y_scores)

    return np.mean(Map(score, [yy for yy in Range(y) if yy not in skip]))

def average_precision_limited(distances,indices, labels):
    '''
    distances: np.array of shape (n_samples, n_samples)
    indices: np.array of shape (n_samples,)
    labels: np.array of shape (n_samples,)
    '''
    labels = np.array(labels)
    def score(i):
        nn_labels = labels[indices[i]]
        y_true = nn_labels == labels[i]
        return average_precision_score(y_true,-distances[i])

    return np.mean(Map(score, Range(labels)))

def average_precision_srt(distances,y):

    def score(i):
        y_true = y == y[i]
        return average_precision_score(y_true,-distances[i])

    return np.sort(Map(score, Range(y)))


def average_precision_nozero(distances,y):
	nonzero = y != 0
	def score(dist, currentlabel):
		y_true = y == currentlabel

		return average_precision_score(y_true,-dist)
	return np.mean(Map(score, distances[nonzero], y[nonzero]))


def average_precision_labelsubset(distances, allLabels, labelSubset):

	# make a copy of alllabels, then set everything to 0 that is not in labelSubset

	labels = np.array(allLabels)
	labels = np.where(np.isin(labels, labelSubset), labels, 0)
	return average_precision_nozero(distances,labels)

	labels = np.array(allLabels)
	def score(i):
		y_true = labels == labels[i]
		return average_precision_score(y_true,-distances[i])
	okrows = np.where(np.isin(labels, labelSubset))[0]
	return np.mean(Map(score, okrows ))





def CSLS(csr_matrix, k=10):
	n = csr_matrix.shape[0]
	D = (csr_matrix @ csr_matrix.T).toarray()
	# Find k-nearest neighbors for each point (excluding self)
	knn = np.argpartition(-D, k+1, axis=1)[:, :k+1]  # +1 to account for self
	knn = np.array([row[row != i] for i, row in enumerate(knn)])  # remove self
	# Compute mean similarity of each point's neighborhood r(x_i)
	r = np.array([D[i, knn[i]].mean() for i in range(n)])
	# Symmetric CSLS adjustment
	csls = 2 * D - r[:, None] - r[None, :]
	return csls


