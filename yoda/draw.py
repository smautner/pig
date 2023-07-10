import networkx as nx
import numpy as np
from lmz import Range
from matplotlib import pyplot as plt


def scatter(X, y):
    plt.scatter(*X.T, c = y)
    plt.show()


def asciigraph(ali):
    graph = nx.relabel_nodes(ali.graph,mapping = dict(zip(ali.graphnodes, Range(ali.graphnodes))),copy = True)
    RNAprint(graph, size = 2)
    return ali.graph


def shownodes(ali):
    for n in ali.graph.nodes:
        print(ali.graph.nodes[n])


def plotneighbors(alignments,distance_matrix, alignment_id = 5, num = 3):
    def plotalign(a1,a2):
        # _fancyalignment(seq1,seq2,str1,str2)
        nn._fancyalignment(a1.graph.graph['sequence'],
                           a2.graph.graph['sequence'],
                           a1.graph.graph['structure'],
                           a2.graph.graph['structure'])

    for i in np.argsort(distance_matrix[alignment_id])[1:num+1]:
        plotalign(alignments[alignment_id], alignments[i])
