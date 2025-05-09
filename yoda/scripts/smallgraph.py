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
import ubergauss.hyperopt as ho
import structout as so
from yoda.graphs import ali2graph
import networkx as nx
import matplotlib.pyplot as plt
import umap

def makedata(splits = 3,add_cov=''):
    alis, labels = alignments.load_rfam(add_cov='')
    if splits <= 2:
        return alis, labels
    # 3 way split
    train, test = next( uo.groupedCV(n_splits = splits).split(alis,labels,labels))
    te = [alis[x] for x in test], [labels[x] for x in test]
    a,l = [alis[x] for x in train], [labels[x] for x in train]
    return a,l, te

from  yoda.graphs.ali2graph import decorateAbstractgraph as abstract, get_coarse
import eden.graph as eg
from yoda.ml import simpleMl as sml



space =  '''RY_thresh .5 1
dillution_fac1 0 1
dillution_fac2 0 1
cons_thresh 0.5 1
cutS1 0 20 1
cutS2 1 40 1
cutD1 0 20 1
norm 0 1 1
cutD2 1 40 1'''
space = ho.spaceship(space)




def string_to_space(space):
    return ho.spaceship(space)



def getvalues(val):
    if val[0].isdigit():
        integers= Map(int, val.split())
        return Range(*integers)
    else:
        return eval(val)

def string_to_paramd(text):
    result = {}
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        key, *values = line.strip().split(maxsplit=1)
        result[key] = getvalues(values[0])
    return result







taskfilter=lambda x: x['cutS2'] > x['cutS1'] and x['cutD2'] > x['cutD1'] and x['dillution_fac1'] > x['dillution_fac2']

def evaleval(X, y, **kwargs):

    alig = Map( get_coarse, X, **kwargs)
    X = eg.vectorize(alig, discrete = True, normalization=kwargs['norm'], inner_normalization=kwargs['norm'])
    X = umap.UMAP(n_components = 4).fit_transform(X)

    return sml.knn_accuracy(X,y) + sml.kmeans_ari(X,y)

import ubergauss.optimization as uo
if __name__ == f"__main__":
    d1,d2,_ = makedata()
    uo.gridsearch(evaleval, space, data=[d1, d2], taskfilter=taskfilter)
    # get besst
    # score trainset
    # score testset


def scorefeatures(vector,labels=[]):
    pass




