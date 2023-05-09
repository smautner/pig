import numpy as np
import networkx as nx
from lmz import *
import eden.graph as eg # eden-kernel in pip
from collections import defaultdict, Counter

open_brace_string={")":"(",
                "]":"[",
                ">":"<","}":"{"}



def nested_frag_encoder(ali):
    '''
    - adds entropy info to into vectorlabels
    - builds backbone along the main structure,
            if there are side-structures (indicated by dots in ss_cons)
            they get connected via darkedges
    -> hence the name nested fragment encoder
    '''
    graph = nx.Graph()
    lifo = defaultdict(list)

    stru = ali.gc['SS_cons']
    for i, ssc in enumerate(stru):
        # FIND NODE LABEL
        # most common:
        ct = Counter( ali.alignment[:,i].tolist())
        for k,v in ct.most_common():
            if k in "ACGU":
                nodelabel = k
                break
        # ADD NODE,
        # with a vector annotation,,, am i using that?
        myv = [ord(ali.gc[k][i]) for k in ali.gc.keys()]
        graph.add_node(i, label=k, vec=myv)
        # handle hydrogen bonds
        if ssc in ['(','[','<']:
            lifo['x'].append(i)
        if ssc in [')',']','>']:
            j = lifo['x'].pop()
            graph.add_edge(i, j, label='=', type='basepair', len=1)

    # ADD BACKBONE
    lastgoodnode =  0
    for i in range(len(stru)-1):
        a,b = stru[i]=='.', stru[i+1]=='.'
        if a == b: # if a and b are the same we can just insert a normal edge
            graph.add_edge(i,i+1, label='-', type='backbone', len=1)
        elif a and not b: #  .-
            graph.add_edge(i,i+1, label='zz', nesting=True) #nesting are dark edges in eden
            if lastgoodnode:
                graph.add_edge(lastgoodnode, i+1, label='-', type='backbone', len=1)
        elif b and not a: #  -.
            lastgoodnode = i
            graph.add_edge(i,i+1, label='zz', nesting=True) #nesting are dark edges in eden
    return graph




def _most_common_nucs(ali):
    ''' goes through the alignments columnwise and finds the most common ACGU'''
    def getch(x):
        col = ali.alignment[:,x].tolist()
        counts = Counter( col )
        for k,v in counts.most_common():
            if k in "ACGU":
                return k
    return Map(getch, Range(ali.alignment.shape[1]))




def mainchainentropy(ali, structure  = 'SS_cons'):
    '''
        cleaning up the graph generation a bit,
        will keep a main alignment and just encode that together with the entropy...
        hopefully will keep a consensus to use for printing later..
        maybe i can also use sscons as vec-features
    '''
    graph = nx.Graph()
    lifo = defaultdict(list)
    nucs = _most_common_nucs(ali)
    simple_sscons = ali.gc[structure]
    simple_sscons = simple_sscons.replace(',',':')
    simple_sscons = simple_sscons.replace('-',':')
    simple_sscons = simple_sscons.replace('_',':')
    ali.gc[structure] = simple_sscons

    sequence = ''
    conSS = ''

    for i, (struct,nuc) in enumerate(zip(simple_sscons,nucs)):
        if struct != '.' and nuc != None: # ATTENTION! some structures have a :  but there is not even one nucleotide listed
            try:
                conSS += struct
                sequence+=nuc
                myv  = [ ord(ali.gc[k][i]) for k in ali.gc.keys() ]
                graph.add_node(i, label=nuc, vec=myv)
                # handle hydrogen bonds
                if struct in ['(','[','<']:
                    lifo['x'].append(i)
                if struct in [')',']','>']:
                    j = lifo['x'].pop()
                    graph.add_edge(i, j, label='=', type='basepair', len=1)
            except:
                print("ERROR IN FILE", ali.fname)

    # ADD BACKBONE
    nodes = list(graph)
    nodes.sort()
    for i in Range(len(nodes)-1):
            a,b = nodes[i], nodes[i+1]
            graph.add_edge(a,b, label='-', type='backbone', len=1)

    graph.graph = {}
    graph.graph['structure'] = conSS
    graph.graph['sequence'] = sequence
    return graph



