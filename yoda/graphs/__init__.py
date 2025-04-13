import networkx as nx
import numpy as np
from lmz import Map
from scipy import sparse
from ubergauss import tools as ut
from eden import graph as eg
import grakel
import time


from yoda.graphs import ali2graph

def alignment_to_vectors(alignments,
                         RYthresh=0,
                         d1=0.25,
                         d2=0.75,
                         fix_edges=True,
                         ct=0.97,
                         bad_weight=0.15,
                         min_r=2,
                         min_d=1,
                         normalization=True,
                         inner_normalization=True,
                         clusterSize=0,
                         maxclust=10,
                         simplegraph=False,
                         nest=True):
    """
    Process RNA alignments into feature vectors.

    This function performs the following steps:
    1. Preprocess alignments into graphs
    2. Vectorize the graphs

    Parameters:
    -----------
    alignments : list
        List of alignment objects
    RYthresh : float, default=0
        Threshold used for R/Y base pairing in the set_weight_label function
    ct : float, default=0.97
        Conservation threshold for set_weight function
    d1 : float, default=0.25
        First dillution parameter for dillute function
    d2 : float, default=0.75
        Second dillution parameter for dillute function
    bad_weight : float, default=0.15
        Weight for bad edges in set_weight function
    fix_edges : bool, default=True
        Whether to fix edges in dillute function
    min_r : int, default=2
        Minimum radius parameter for vectorization
    min_d : int, default=1
        Minimum distance parameter for vectorization
    normalization : bool, default=True
        Whether to apply normalization during vectorization
    inner_normalization : bool, default=True
        Whether to apply inner normalization during vectorization
    clusterSize : int, default=0
        Size of clusters for multiGraph function. If 0, multiGraph is not used
    maxclust : int, default=10
        Maximum number of clusters for multiGraph function
    simplegraph : bool, default=False
        Whether to use simplegraph option in multiGraph
    nest : bool, default=True
        Whether to apply donest function

    Returns:
    --------
    matrix : sparse matrix
        Feature vectors representing the alignments
    """


    def preprocess(ali):
        graph = ali2graph.set_weight_label(ali, RYthresh=RYthresh)
        graph = ali2graph.dillute(graph, dilute1=d1, dilute2=d2, fix_edges=fix_edges)
        graph = ali2graph.set_weight(graph, bad_weight=bad_weight, consThresh=ct)
        if clusterSize:
            graph = ali2graph.multiGraph(ali, clusterSize=clusterSize, maxclust=maxclust, simplegraph=simplegraph)
        if nest:
            graph = ali2graph.donest(graph)
        return graph

    # Process all alignments in parallel
    graphs = ut.xxmap(preprocess, alignments)

    # Vectorize the graphs
    # matrix = eg.vectorize(graphs, normalization=normalization,) THIS IS MEGA SLOW oO
    matrix = vectorize_graphs(graphs, normalization=normalization,
                          min_r=min_r, min_d=min_d,
                          inner_normalization=inner_normalization)

    return matrix



def vectorize_alignments(alignments,**kwargs):
    return vectorize_graphs([x.graph for x in alignments], **kwargs)


def vectorize_graphs(graphs,**kwargs):
    vectors = ut.xxmap(eg.vectorize, [[g] for g in graphs], **kwargs)
    vectors = sparse.vstack(vectors)
    return vectors


def vectorize_graphs_hack(graphs,**kwargs):
    vectors = ut.xxmap(hackvectorize, [[g] for g in graphs], **kwargs)
    vectors = sparse.vstack(vectors)
    return vectors

class hackvectorizer(eg.Vectorizer):
    def __init__(self,disthack =2,**kwargs):
        super().__init__(**kwargs)
        self.distance_hack_cutoff = disthack
    def _transform_vertex_pair_valid(self,
                                     graph,
                                     vertex_v,
                                     vertex_u,
                                     radius,
                                     distance,
                                     feature_list,
                                     connection_weight=1):
        cw = connection_weight
        if distance < self.distance_hack_cutoff:
            distance = 1
        else:
            distance = 2
        # we need to revert to r/2 and d/2
        radius_dist_key = (radius / 2, distance / 2)
        # reweight using external weight dictionary
        len_v = len(graph.nodes[vertex_v]['neigh_graph_hash'])
        len_u = len(graph.nodes[vertex_u]['neigh_graph_hash'])
        if radius < len_v and radius < len_u:
            # feature as a pair of neighborhoods at a radius,distance
            # canonicalization of pair of neighborhoods
            vertex_v_labels = graph.nodes[vertex_v]['neigh_graph_hash']
            vertex_v_hash = vertex_v_labels[radius]
            vertex_u_labels = graph.nodes[vertex_u]['neigh_graph_hash']
            vertex_u_hash = vertex_u_labels[radius]
            if vertex_v_hash < vertex_u_hash:
                first_hash, second_hash = (vertex_v_hash, vertex_u_hash)
            else:
                first_hash, second_hash = (vertex_u_hash, vertex_v_hash)
            feature = eg.fast_hash_4(
                first_hash, second_hash, radius, distance, self.bitmask)
            # half features are those that ignore the central vertex v
            # the reason to have those is to help model the context
            # independently from the identity of the vertex itself
            half_feature = eg.fast_hash_3(vertex_u_hash,
                                       radius, distance, self.bitmask)
            if graph.graph.get('weighted', False) is False:
                if self.use_only_context is False:
                    feature_list[radius_dist_key][feature] += cw
                feature_list[radius_dist_key][half_feature] += cw
            else:
                weight_v = graph.nodes[vertex_v]['neigh_graph_weight']
                weight_u = graph.nodes[vertex_u]['neigh_graph_weight']
                weight_vu_radius = weight_v[radius] + weight_u[radius]
                val = cw * weight_vu_radius
                # Note: add a feature only if the value is not 0
                if val != 0:
                    if self.use_only_context is False:
                        feature_list[radius_dist_key][feature] += val
                    half_val = cw * weight_u[radius]
                    feature_list[radius_dist_key][half_feature] += half_val


def hackvectorize(thing,**kwargs):
    v = hackvectorizer(**kwargs)
    return v.transform(thing)




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



