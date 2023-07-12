from lmz import Map
from sklearn.metrics import silhouette_score
import numpy as np
from yoda.ml import simpleMl
from ubergauss import tools as ut
from yoda.graphs import ali2graph
from sklearn.metrics import pairwise
import matplotlib
from yoda.draw import asciigraph
from yoda.graphs import vectorize_alignments, grakel_vectorize
from yoda.ml import embed, get_distance_method
matplotlib.use('module://matplotlib-sixel')
import ubergauss.optimization as uo

from yoda import alignments
# def get_ali_dist():
#     # iconv -f ISO-8859-1 -t UTF-8 Rfam.seed > Rfam.seed.utf8
#     alignments = filein.readseedfile(f'/home/ubuntu/Rfam.seed.utf8')
#     graphs = ut.xmap(ali2graph.rfamseedfilebasic, alignments)
#     vectors = vectorize_alignments(graphs)
#     dist = pairwise.linear_kernel(vectors)
#     return alignments, dist , vectors

########################
# so, first we cluster all the right alignments
##########################
# data = alignments.load_rfam()




def run( alignments,labels,
        RY_thresh = .1, conservation = [.50,.75,.90,.95],
        covariance = .05, sloppy = False, fake_nodes = False, vectorizer = 'WeisfeilerLehman',
        distance_measure = 'euclidean',min_rd = 2,n_dim = 6, mp = False):

    mapper = ut.xmap if mp else Map
    alignments = mapper( lambda ali:ali2graph.rfam_graph_decoration(ali, RY_thresh = RY_thresh ,
                          conservation = conservation,
                          covariance = covariance,
                          sloppy = sloppy,
                          fake_nodes = fake_nodes),
                          alignments)

    # asciigraph(alignments[4])

    # breakpoint()
    # vectors = vectorize_debug(alignments)

    # vectors = vectorize(alignments,
    #                     min_rd = min_rd,
    #                     ignorevectorlabels= False,
    #                     mp=mp)

    # vectors  = mapper(lambda vectorizer: grakel_vectorize(x, vectorizer), alignments)
    vectors = vectorize_alignments(alignments, mp=mp, min_rd= min_rd)
    di = get_distance_method(distance_measure)
    return silhouette_score(di(vectors), labels, metric=f'precomputed')

    # dist = di(vectors)
    # X = embed(dist, n_dim= n_dim)
    # # plotneighbors(alignments,dist, alignment_id = 5, num  = 4)
    # # scatter(X, labels)
    # # print(f"{ simpleMl.permutation_score(X, labels) = }")
    # return silhouette_score(X, labels)
    # # return silhouette_score(X, labels),


############
# get goodd features
#####################


def eval_ft(ft,X,y):
    # X=X.todense()
    X = X[:,ft==1]
    X = pairwise.euclidean_distances(X)
    # X = embed(X, n_dim= 6)
    return silhouette_score(X, y, metrix= f'precomputed')

def add_vector_attributes(ali):
    return  ut.xmap( lambda ali:ali2graph.rfam_graph_decoration(ali,
                          RY_thresh = .2,
                          covariance = 0.05,
                          sloppy = False,
                          fake_nodes = False),
                          ali)

def supervised(a,l, n_ft = 100):
    # a = ut.xmap(ali2graph.rfam_graph_structure_deco, a)
    a = add_vector_attributes(a)

    X = vectorize_alignments(a, min_rd=1, mp= True)
    test_scores = []

    for train, test in uo.groupCV(n_splits = 3).split(X,l,l):
        ft = simpleMl.featuremask(*train, n_ft)
        test_scores.append( (eval_ft(ft, *test)) )
        print(f"{ eval_ft(ft, *train) = }")

    print(f"{ np.mean(test_scores)= }")






############
#   grip optimization
##########
from ubergauss import optimization as opti
grid = {
        'RY_thresh' : np.linspace(0.1, 0.3, 3),
        'covariance': [.05, .1],
        # 'distance_measure': 'euclidean '.split(),
        # 'fake_nodes': [False,True],
        # 'sloppy': [False,True],
        'min_rd': [1,2]}

def optimize():
    data = alignments.load_rfam(full = False)
    df =  opti.gridsearch(run, grid, data)
    print(df.corr(method='pearson'))
    opti.print(df)
    print(df.sort_values(by = 'score'))

