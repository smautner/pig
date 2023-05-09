

import yoda.filein as vv
import yoda.nearneigh as nn
import numpy as np
import ubergauss.tools as ut
import time
from scipy import sparse
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.utils import shuffle


from pyod import models as od



def calc():
    '''
    loads 70k data, vectorizes(eden) it and runs an outlier detection algorithm

    -- outlier detection sucks though

    Returns: true, score, classifiername

    '''
    start = time.time()
    limit = 0
    # X,y,files = vv.getXYFiles(path = '/home/stefan/WEINBERG/',limit = limit, discrete = True); ut.dumpfile((X,y,files), f'{limit}delme.dmp')
    (X,y,files)  = ut.loadfile(f'{limit}delme.dmp')
    print(f"vectorized  used so far:{(time.time()-start)/60}")

    X= sparse.vstack(X)
    y= np.array(y)
    neg = X[y==0]
    pos = X[np.logical_or(y==2,y==1)]

    neg = shuffle(neg)
    neg_train = neg[:neg.shape[0]//2]
    neg_test = neg[neg.shape[0]//2:]

    test = sparse.vstack([neg_test,pos])
    test_y = [0]*neg_test.shape[0] + [1]* pos.shape[0]

    # clf = OneClassSVM(gamma='auto').fit(neg_train) ; name = 'rbf OneClassSVM '
    # clf = SGDOneClassSVM().fit(neg_train); name = 'sgd OneClassSVM'
    # clf = LocalOutlierFactor().fit(neg_train) ; name = 'localoutlierfactor' # too slow
    # clf = IsolationForest().fit(neg_train); name = 'IsolationForest'
    # scores = -clf.score_samples(test)

    # also tryable lmdd lscp abod
    # clf = od.ECOD(); clf.fit(neg_train.toarray()) ; name = 'ecod'
    # return test_y, -clf.decision_function(test.toarray()), name

    # CBLOF also tryable lmdd lscp abod
    import pyod.models.cblof as c
    clf = c.CBLOF(); clf.fit(neg_train.toarray()) ; name = 'CBLOF()'
    scores = -clf.decision_function(test.toarray())

    return test_y, scores, name






