import networkx as nx
import numpy as np
from lmz import Map
from scipy import sparse
from ubergauss import tools as ut
from eden import graph as eg


def vectorize_alignments(alignments,min_rd = 1, ignorevectorlabels = False, mp = False):
    vectorizer = lambda x: eg.vectorize([x.graph],
                                        discrete = ignorevectorlabels,
                                        min_r = min_rd,
                                        min_d = min_rd) # normalization=False, inner_normalization=False)
    mapper = ut.xmap if mp else Map
    vectors = mapper(vectorizer, alignments)
    vectors = sparse.vstack(vectors)
    return vectors


def convert_to_grakel_graph(graph: nx.DiGraph) -> grakel.Graph:
    dol = nx.convert.to_dict_of_lists(graph)
    node_attr = dict()
    for nid, attr in graph.nodes(data=True):
        node_attr[nid] = np.asarray([attr['label']]+attr['vec'].tolist())

    result = grakel.Graph(dol, node_labels=node_attr, graph_format='dictionary')
    for node in [n for n in dol.keys() if n not in result.edge_dictionary.keys()]:
        result.edge_dictionary[node] = {}
    return result


def grakel_vectorize(alignment, vectorizer= 'WeisfeilerLehman'):
    g =  convert_to_grakel_graph( alignment.graph)
    d = {'normalize' : False, 'sparse' : True}
    return eval('grakel'.vectorizer)(**d).fit_transform(g)
