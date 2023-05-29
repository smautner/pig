from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from yoda import filein, alignment, ali2graph , simpleMl
from ubergauss import tools as ut
import eden.graph as eg


def countbla():
    import glob
    from collections import Counter
    f = glob.glob(f'../../rfamome/*.stk')
    m = map(alignment.grepfamily, f)
    c = Counter(m)
    print(sum([v for v in c.values() if v > 2]))
    print(c)


def runbaseline():
    alis = filein.loadrfamome(path = f'../../rfamome')
    alis = filein.addstructure(alis) # a.struct
    alis = filein.addcov(alis)       # a.rscape [(start,stop,e-valus)]
    alis = alignment.process_cov(alis, debug = False)

    graphs = Map(ali2graph.scclust,alis) # not done!
    X = eg.vectorize(graphs)
    y = [a.label for a in alis]

    # 1nn check
    print(f"{simpleMl.knn_accuracy(X,y,n_neighbors = 1)=}")
    print(f"{simpleMl.knn_f1(X,y,n_neighbors = 1,cv_strati_splits = 3)=}")
    # cv kmeans clustering
    print(f"{simpleMl.kmeans_ari(X,y,k=28)=}")


    # loguru!

    # so.grpint pos RNA
    # https://github.com/fabriziocosta/EDeN/blob/master/eden/display/__init__.py

    # vectorizer should have many options... why are the fake nodes decreasing performance?
    # especially removing the fakenodes is improving performance, with the so.rna stuff
    # i should be able to debug and see if i implemented it correctly


    # experiment a bit with this dataset...
    # then build the rfam tree or cluster the clan ones!
