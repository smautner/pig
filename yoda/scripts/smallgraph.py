from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten



'''

we want to compres the alignment into a small graph:
a whole stem might be 1 node.
we need diverse labels to make the graphkernel work
there fore we just use the number of nodes and diskretize to 3?
values -> stem_small might be a label
then we use conservation levels and (nuc distributions?) as vector labels


'''


# load data
from yoda import alignments
import ubergauss.optimization as uo
import structout as so
from yoda.graphs import ali2graph
import networkx as nx
import matplotlib.pyplot as plt

def makedata():
    alis, labels = alignments.load_rfam()
    # 3 way split
    train, test = next( uo.groupedCV(n_splits = 3).split(alis,labels,labels))
    te = [alis[x] for x in test], [labels[x] for x in test]
    a,l = [alis[x] for x in train], [labels[x] for x in train]
    return a,l, te

from  yoda.graphs.ali2graph import decorateAbstractgraph as abstract
import eden.graph as eg
from yoda.ml import simpleMl as sml
space = {
    'sa' : Range(6,20,3),
    'sb' : Range(20,30,3),
    'da' : Range(0,15,3),
    'db' : Range(24,36,3)
}
taskfilter=lambda x: x['sb'] > x['sa'] and x['db'] > x['da']
import umap
def eval(X,y, sa=0,sb=0,da=0,db=0):

    if sa > sb or da > db:
        return None

    alig = Map( abstract, X,len_single=[sa,sb], len_double=[da,db])


    X = eg.vectorize(alig, discrete = False, normalization=False, inner_normalization=False)
    X = umap.UMAP(n_components = 4).fit_transform(X)

    return sml.knn_accuracy(X,y) + sml.kmeans_ari(X,y)

import ubergauss.optimization as uo
if __name__ == f"__main__":
    d1,d2,_ = makedata()
    uo.gridsearch(eval, space, data=[d1,d2], taskfilter=taskfilter)



# make abstract graphs -> optimization function (blabla)
## lets do this for testing a bit...

# optimize on 2/3

# eval onthe last 1/3


