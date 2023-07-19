import networkx as nx
import numpy as np
from lmz import Map
from scipy import sparse
from ubergauss import tools as ut
from eden import graph as eg
import grakel
import time

def vectorize_alignments(alignments,min_rd = 1, ignorevectorlabels = False, mp = False):
    vectorizer = lambda x: eg.vectorize([x.graph],
                                        discrete = ignorevectorlabels,
                                        min_r = min_rd,
                                        min_d = min_rd) # normalization=False, inner_normalization=False)
    mapper = ut.xmap if mp else Map
    vectors = mapper(vectorizer, alignments)
    vectors = sparse.vstack(vectors)
    return vectors


def convert_to_grakel_graph(alignment) -> grakel.Graph:

    graph = alignment.graph
    dol = nx.convert.to_dict_of_lists(graph)
    node_attr = dict()
    for nid, attr in graph.nodes(data=True):
        # node_attr[nid] = np.asarray([attr['label']]+attr['vec'].tolist())
        node_attr[nid] = np.asarray([attr['label']]+attr['vec'])

    result = grakel.Graph(dol, node_labels=node_attr, graph_format='dictionary')
    for node in [n for n in dol.keys() if n not in result.edge_dictionary.keys()]:
        result.edge_dictionary[node] = {}
    return result


def grakel_vectorize(alignments, vectorizer= 'WeisfeilerLehman'):
    g =  Map(convert_to_grakel_graph, alignments )
    d = {} # 'normalize' : False, 'sparse' : True}
    starttime= time.time()
    res =  eval(f'grakel.{vectorizer}')(**d).fit_transform(g)
    return time.time()-starttime

grakel_kernels = 'RandomWalk RandomWalkLabeled PyramidMatch NeighborhoodHash ShortestPath ShortestPathAttr GraphletSampling SubgraphMatching WeisfeilerLehman HadamardCode NeighborhoodSubgraphPairwiseDistance LovaszTheta SvmTheta Propagation PropagationAttr OddSth MultiscaleLaplacian HadamardCode VertexHistogram EdgeHistogram GraphHopper CoreFramework WeisfeilerLehmanOptimalAssignment'.split(f' ')
