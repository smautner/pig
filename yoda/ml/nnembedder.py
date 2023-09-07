from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import ubergauss.tools as ut

import networkx.linalg as nxlin
import numpy as np
def makematrix(graph, maxlen = None):

    startpadding = 5
    endpadding = 5

    # add padding nodes
    missing_nodes = maxlen - len(graph)
    missing_nodes += endpadding # extra padding
    maxlen += startpadding+endpadding

    for i in range(missing_nodes):
        graph.add_node(max(graph.nodes)+1 , label = 'N')

    for i in range(startpadding):
        graph.add_node( -i-1 , label = 'N')

    assert len(graph)== maxlen, f"{len(graph)=}, {maxlen=}"
    sorted_nodes = sorted(graph.nodes)
    adjacency_matrix = nxlin.adjacency_matrix(graph,nodelist=sorted_nodes).todense()
    # now we need to add another axis and add node infos
    res = np.zeros((11, maxlen, maxlen))
    indices = np.repeat ( Range(maxlen), maxlen).reshape((maxlen,maxlen))
    def makeonehot(char):
        return np.repeat ( [ 1 if graph.nodes[n]['label'] == char else 0 for n in sorted_nodes ] , maxlen).reshape((maxlen,maxlen))

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


    import structout as so
    for layer in atensor:
        so.heatmap(layer)


