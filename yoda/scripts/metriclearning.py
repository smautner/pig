from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from lmz import Map
from sklearn.metrics import silhouette_score
import numpy as np
from yoda.ml import simpleMl
from ubergauss import tools as ut
from yoda.graphs import ali2graph
from sklearn.metrics import pairwise
import matplotlib
from yoda import draw
# from yoda.graphs import vectorize_alignments, grakel_vectorize
import yoda.graphs as graphs
from yoda.ml import embed, get_distance_method
matplotlib.use('module://matplotlib-sixel')
import ubergauss.optimization as uo
from yoda import alignments


########################
# so, first we cluster all the right alignments
##########################
# data = alignments.load_rfam()

def eval_ft(X_new,_,y):
    # X=X.todense()
    # X = pairwise.euclidean_distances(X_new)
    # X = embed(X, n_dim= 6)
    #return silhouette_score(X_new, y, metric= f'precomputed')
    return silhouette_score(X_new, y)

def add_vector_attributes(ali):
    return  ut.xmap( lambda ali:ali2graph.rfam_graph_decoration(ali,
                       RY_thresh = .7,nuc_thresh = .65, conservation = [],
                        covariance = 0.05, sloppy = False, fake_nodes = False, progress = True,
                        nucleotideRNAFM = False),
                ali)



import metric_learn
from collections import Counter
def supervised(alis,labels, n_ft = 100):
    a = add_vector_attributes(alis)

    # i think this samples sequences
    # a,labels = Transpose( Flatten ( ut.xmap(lambda x: ali2graph.manifest_subgraphs(x, maxgraphs = 10),zip(a,labels))))
    # a, labels = Transpose(Flatten ([ (newali,label)  for ali,label in zip(a,labels) for newali in ali2graph.manifest_subgraphs(ali,100) ] ))
    # there is manifest_sequences now !
    a, labels = ali2graph.manifest_sequences(a,labels, instances = 5  )

    labels = np.array(labels)
    X = graphs.vectorize_alignments(a, min_rd=1, mp= True)

    # remove empty columns
    X=X.toarray()
    import yoda.ml.simpleMl as ml
    ft = ml.featuremask(X, labels, n_ft = 1000)
    X = X[:,ft==1]
    print(f"{X.shape=}")

    # empty_columns = np.all(X == 0, axis=0)
    # X = X[:, ~empty_columns]


    test_scores = []
    for train, test in uo.groupedCV(n_splits = 3).split(X,labels,labels):
        tr = X[train], labels[train]
        te = X[test], labels[test]
        print(f"{ Counter(tr[1])=}")
        model = metric_learn.LFDA(n_components = 6) # LFDA and MLKR should be tried
        train_transformed = model.fit_transform(*tr)
        test_transformed = model.transform(te[0])
        test_scores.append( (eval_ft(test_transformed, *te)) )
        print(f"{ eval_ft(train_transformed, *tr) = }")
    print(f"{ np.mean(test_scores)= }")


if __name__ == f"__main__":
    data = alignments.load_rfam()
    supervised(*data)


from scipy import sparse

def supervised_graphaverage(alis,labels, n_ft = 100):
    a = add_vector_attributes(alis)

    def stacksamples(ali):
        aliss, _  = Transpose(ali2graph.manifest_subgraphs((ali,False),maxgraphs = 10) )
        X = graphs.vectorize_alignments(aliss, min_rd=1, mp= False)
        return np.mean(X, axis = 0)

    X =  ut.xmap(stacksamples , a)
    X = np.vstack(X)

    print(f"{X.shape=}")
    test_scores = []
    for train, test in uo.groupedCV(n_splits = 3).split(X,labels,labels):
        tr = X[train], labels[train]
        te = X[test], labels[test]
        ft = simpleMl.featuremask(*tr, n_ft)
        test_scores.append( (eval_ft(ft, *te)) )
        print(f"{ eval_ft(ft, *tr) = }")
    print(f"{ np.mean(test_scores)= }")

