import networkx as nx
import numpy as np
import structout as so
from lmz import Range, Map
from matplotlib import pyplot as plt
from structout.rnagraph import RNAprint

import yoda.alignments

def scatter(X, y):
    plt.scatter(*X.T, c = y)
    plt.show()
    plt.close()


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
        _fancyalignment(a1.graph.graph['sequence'],
                        a2.graph.graph['sequence'],
                        a1.graph.graph['structure'],
                        a2.graph.graph['structure'])

    for i in np.argsort(distance_matrix[alignment_id])[1:num+1]:
        plotalign(alignments[alignment_id], alignments[i])


def plot_NN(dists, classes=0):
    NN = dists[:,1]
    if isinstance(classes, int):
            so.hist(NN,bins= 40)
    else:
        for e in np.unique(classes):
            so.hist(NN[classes == e],bins= 40)
    print(f"closest {min(NN)}")


def _lsearch(seq):
    # returns index of first char
    for i,e in enumerate(seq):
        if e != '-':
            return i


def _rsearch(seq):
    numfucks = _lsearch(seq[::-1])
    return len(seq) - numfucks


def _fancyalignment(seq1,seq2,str1,str2):

    al1, al2 = yoda.pairwise_alignments.needle(seq1, seq2)

    def adjuststruct(al1,str1):
        # 1. insert dashes into the str
        str1=list(str1)
        re = ''
        for e in al1[::-1]:
            if e == '-':
                re+=e
            else:
                re+=str1.pop()
        return re[::-1]

    str1 =  adjuststruct(al1,str1)
    str2 =  adjuststruct(al2,str2)


    '''
    here we have st1+2 and al1+2 that all have the same lengh
    ... next we want to format it...
    everything is aligned, we just cut it into 3 pieces
    '''

    # cutoff
    left = max(Map(_lsearch, (al1,al2)))
    right = min(Map(_rsearch, (al1,al2)))


    def cutprint(item):
        nust = item[left:right]
        left1  = _countfmt(item[:left])
        right1  = _countfmt(item[right:])
        nust = f'{left1} {nust} {right1}'
        print(nust)

    Map(cutprint,(al1,str1,al2,str2))
    print()
    print()


def _countfmt(item):
    # count '-' and gives the number a pading so the string has length 4
    cnt = str(len([a for a in item if a !='-']))
    return cnt+' '*(4-len(cnt))


def test_fancy_alignment():
    ab = ('----asdasda asd as da sd---', '---asdasdasd asd a sd---', )
    _fancyalignment(*ab,*ab)
