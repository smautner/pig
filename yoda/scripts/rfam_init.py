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



def find_working_grakel_kernels(alignments,i):
    """
    this is just to figure out which kernels work in the first place
    """
    grakel_kernels = graphs.grakel_kernels
    alignments = add_vector_attributes(alignments[:2])
    graphs.grakel_vectorize(alignments,grakel_kernels[i])
    oklist = []
    for i,v in enumerate(grakel_kernels):
        try:
            dist = graphs.grakel_vectorize(alignments,v)
            # print(f"{ dist=}")
            print(f"{ v=}{i=} ok")
            oklist.append(v)
        except:
            print(f"{ v=}{i=} failed")
    print(oklist)

import structout as so
def run( alignments,labels,
        RY_thresh = .5,nuc_thresh = .5, conservation = [],
        covariance = False, sloppy = False, fake_nodes = False, progress = False,
        kernel = f'', # 'WeisfeilerLehman',
        nucleotideRNAFM = False,
        distance_measure = 'euclidean',min_rd = 2,n_dim = 6, mp = False):

    mapper = ut.xmap if mp else Map

    alignments = mapper( lambda ali:ali2graph.rfam_graph_decoration(ali,
                          RY_thresh = RY_thresh ,
                          nuc_thresh=nuc_thresh,
                          conservation = conservation,
                          nucleotideRNAFM = nucleotideRNAFM,
                          covariance = covariance,
                          sloppy = sloppy,
                          progress= progress,
                          fake_nodes = fake_nodes),
                          alignments)


    # draw.asciigraph(alignments[4])
    # draw.shownodes(alignments[4])
    # return


    if kernel:
        sim_matrix  = graphs.grakel_vectorize(alignments, kernel)
        if np.max(sim_matrix)==0:
            return -1
        distance_matrix = 1-sim_matrix/np.max(sim_matrix)
        np.fill_diagonal(distance_matrix, 0)
    else:
        vectors = graphs.vectorize_alignments(alignments, mp=mp, min_rd= min_rd)
        distance_matrix = get_distance_method(distance_measure)(vectors)

    return silhouette_score(distance_matrix, labels, metric='precomputed')

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
    return silhouette_score(X, y, metric= f'precomputed')

def add_vector_attributes(ali):
    return  ut.xmap( lambda ali:ali2graph.rfam_graph_decoration(ali,
                       RY_thresh = .7,nuc_thresh = .65, conservation = [],
                        covariance = 0.05, sloppy = False, fake_nodes = False, progress = False,
                        nucleotideRNAFM = False),
                ali)

def supervised(alis,labels, n_ft = 100):
    a = add_vector_attributes(alis)

    a,labels = Transpose( Flatten ( ut.xmap(lambda x: ali2graph.manifest_subgraphs(x,maxgraphs = 10),zip(a,labels))))
    # a, labels = Transpose(Flatten ([ (newali,label)  for ali,label in zip(a,labels) for newali in ali2graph.manifest_subgraphs(ali,100) ] ))
    labels = np.array(labels)

    X = graphs.vectorize_alignments(a, min_rd=1, mp= True)
    for n_ft in [1000,3000]:
        test_scores = []
        for train, test in uo.groupedCV(n_splits = 3).split(X,labels,labels):
            tr = X[train], labels[train]
            te = X[test], labels[test]
            ft = simpleMl.featuremask(*tr, n_ft)
            test_scores.append( (eval_ft(ft, *te)) )
            print(f"{ eval_ft(ft, *tr) = }")

        print(f"{ np.mean(test_scores)= }")

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




############
#   grip optimization
##########
from ubergauss import optimization as opti
grid = {
        # 'RY_thresh' : np.linspace(0.4, 0.75, 10),
        # 'nuc_thresh': np.linspace(.36,.7,10),
        # 'distance_measure': 'euclidean '.split(),
        'fake_nodes': [False, True],
        'sloppy': [False,True],
        # 'min_rd': [1,2],
        'progress' : Range(2,40,10),
        f'covariance': [False, .05],
        # 'nucleotideRNAFM':[True, False],
        # 'vectorizer' : ['ShortestPath', 'PyramidMatch', 'NeighborhoodHash', 'GraphletSampling', 'WeisfeilerLehman', 'SvmTheta', 'Propagation', 'OddSth', 'VertexHistogram', 'EdgeHistogram', 'CoreFramework', 'WeisfeilerLehmanOptimalAssignment']
        }

def optimize():
    data = alignments.load_rfam(full = False)
    # data, label  = data
    df =  opti.gridsearch(run, grid, data)
    print(df.corr(method='spearman'))
    opti.dfprint(df)
    ut.dumpfile(df,f'lastopti.delme')

def drawoptidf():
    import seaborn as sns
    import matplotlib.pyplot as plt
    # sns.scatterplot(data = df, x = f'RYlimit', y = f'nuclimit', c = list(-df['score']))
    df = ut.loadfile(f'lastopti.delme')

    mydf = df.pivot(f'RY_thresh', f'nuc_thresh', f'score')
    sns.heatmap(mydf)
    plt.tight_layout()
    plt.show()
    plt.close()

