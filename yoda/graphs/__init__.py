import networkx as nx
import numpy as np
from lmz import Map
from scipy import sparse
from ubergauss import tools as ut
from eden import graph as eg
import grakel
import time

def vectorize_alignments(alignments,**kwargs):
    return vectorize_graphs([x.graph for x in alignments], **kwargs)

def vectorize_graphs(graphs,**kwargs):

    vectors = ut.xxmap(eg.vectorize, [[g] for g in graphs], **kwargs)
    vectors = sparse.vstack(vectors)
    return vectors

def convert_to_grakel_graph(alignment) -> grakel.Graph:

    graph = alignment.graph
    dol = nx.convert.to_dict_of_lists(graph)
    node_attr = dict()
    for nid, attr in graph.nodes(data=True):
        # node_attr[nid] = np.asarray([attr['label']]+attr['vec'].tolist())
        node_attr[nid] = ord(attr[f'label'])# np.asarray([ord(attr['label'])]+attr['vec'])

    edge_attr = {}
    for e in graph.edges:
        a,b = e
        edge_attr[e] = ord(graph.edges[a,b]['label'])

    result = grakel.Graph(dol, node_labels=node_attr, edge_labels=edge_attr, graph_format='dictionary')
    for node in [n for n in dol.keys() if n not in result.edge_dictionary.keys()]:
        result.edge_dictionary[node] = {}
    return result



def grakel_vectorize(alignments, kernel= 'WeisfeilerLehman'):
    g =  Map(convert_to_grakel_graph, alignments )
    d = {'normalize' : False} # , 'sparse' : True}
    # jstarttime = time.time()
    if kernel == f'OddSth':
        d[f'h'] = 4
    res =  eval(f'grakel.{kernel}')(**d).fit_transform(g)
    # return time.time()-starttime
    return res
# GraphHopper RandomWalkLabeled
grakel_kernels = 'ShortestPath RandomWalk PyramidMatch NeighborhoodHash GraphletSampling SubgraphMatching WeisfeilerLehman HadamardCode NeighborhoodSubgraphPairwiseDistance LovaszTheta SvmTheta Propagation PropagationAttr OddSth MultiscaleLaplacian HadamardCode VertexHistogram EdgeHistogram CoreFramework WeisfeilerLehmanOptimalAssignment'.split(f' ')



