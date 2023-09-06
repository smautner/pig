from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import ubergauss.tools as ut

import networkx.linalg as nxlin
import numpy as np
def makematrix(graph, maxlen = None):

    # add padding nodes
    missing_nodes = maxlen - len(graph)
    for i in range(missing_nodes):
        graph.add_node(max(graph.nodes)+1 , label = 'N')

    adjacency_matrix = nxlin.adjacency_matrix(graph).todense()
    # now we need to add another axis and add node infos
    res = np.zeros((11, maxlen, maxlen))
    indices = np.repeat ( Range(maxlen), maxlen).reshape((maxlen,maxlen))
    def makeonehot(char):
        return np.repeat ( [ 1 if graph.nodes[n]['label'] == char else 0 for n in graph.nodes ] , maxlen).reshape((maxlen,maxlen))

    ONEHOT = np.array( Map(makeonehot, 'AUGC'))
    res[:4] = ONEHOT
    res[4] = indices
    res[5] = adjacency_matrix
    res[6] = indices.T
    res[7:] = np.transpose(ONEHOT, (0,2,1))

    return res


def toarray(alis):
    graphs = [a.graph for a in alis]
    maxlen = max(map(len, graphs))
    return ut.xmap(makematrix, graphs, maxlen = maxlen)


def test_toarray():
    from yoda import alignments
    a,l = alignments.load_rfam()
    atensor = toarray(a[:2])[0]
    print(atensor)


