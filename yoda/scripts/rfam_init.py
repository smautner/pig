from lmz import Map
from sklearn.metrics import silhouette_score
import numpy as np
from yoda.ml import simpleMl
from ubergauss import tools as ut
from yoda.graphs import ali2graph
from sklearn.metrics import pairwise
import matplotlib
from yoda.draw import asciigraph
# from yoda.graphs import vectorize_alignments, grakel_vectorize
import yoda.graphs as graphs
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
        covariance = .05, sloppy = False, fake_nodes = False,
        vectorizer = f'', # 'WeisfeilerLehman',
        distance_measure = 'euclidean',min_rd = 2,n_dim = 6, mp = False, **kwargs):

    mapper = ut.xmap if mp else Map

    # alignments = mapper( lambda ali:ali2graph.rfam_graph_decoration(ali, RY_thresh = RY_thresh ,
    #                       conservation = conservation,
    #                       covariance = covariance,
    #                       sloppy = sloppy,
    #                       fake_nodes = fake_nodes),
    #                       alignments)

    alignments = mapper( lambda ali:ali2graph.set_base_label(ali,**kwargs), alignments)



    # asciigraph(alignments[4])

    # breakpoint()
    # vectors = vectorize_debug(alignments)

    # vectors = vectorize(alignments,
    #                     min_rd = min_rd,
    #                     ignorevectorlabels= False,
    #                     mp=mp)

    if vectorizer:
        distance_matrix  = graphs.grakel_vectorize(alignments, vectorizer)
    else:
        vectors = graphs.vectorize_alignments(alignments, mp=mp, min_rd= min_rd)
        distance_matrix = get_distance_method(distance_measure)(vectors)

    return silhouette_score(distance_matrix, labels, metric=f'precomputed')

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
        # 'RYlimit' : np.linspace(0.9, 0.99, 10),
        # 'nuclimit': np.linspace(.74,.99,1),
        # 'distance_measure': 'euclidean '.split(),
        # 'fake_nodes': [False,True],
        # 'sloppy': [False,True],
        # 'min_rd': [1,2]
        f'vectorizer' : graphs.grakel_kernels
        }

def optimize():
    data = alignments.load_rfam(full = False)[:50]
    df =  opti.gridsearch(run, grid, data)
    print(df.corr(method='spearman'))
    opti.dfprint(df)
    ut.dumpfile(df,f'lastopti.delme')

def drawoptidf():
    import seaborn as sns
    import matplotlib.pyplot as plt
    # sns.scatterplot(data = df, x = f'RYlimit', y = f'nuclimit', c = list(-df['score']))
    df = ut.loadfile(f'lastopti.delme')
    mydf = df.pivot(f'RYlimit', f'nuclimit', f'score')
    sns.heatmap(mydf)
    plt.tight_layout()
    plt.show()
    plt.close()



