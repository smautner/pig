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



