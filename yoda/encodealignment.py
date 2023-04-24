import numpy as np
import networkx as nx
from lmz import *
import eden.graph as eg # eden-kernel in pip
from collections import defaultdict, Counter

open_brace_string={")":"(",
                "]":"[",
                ">":"<","}":"{"}

def nested_frag_encoder(fname, alignment, annot):
    '''
    - adds entropy info to into vectorlabels
    - builds backbone along the main structure, if there are side-structures (indicated by dots in ss_cons) they get connected via darkedges
    -> hence the name nested fragment encoder
    '''
    graph = nx.Graph()
    lifo = defaultdict(list)


    for i, ssc in enumerate(annot[0]):
        # FIND NODE LABEL
        # most common:
        ct = Counter( alignment[:,i].tolist())
        for k,v in ct.most_common():
            if k in "ACGU":
                nodelabel = k
                break
        # ADD NODE,
        # with a vector annotation,,, am i using that?
        myv  = [ ord(rr[i]) for rr in annot ]
        graph.add_node(i, label=k, vec=myv)
        # handle hydrogen bonds
        if ssc in ['(','[','<']:
            lifo['x'].append(i)
        if ssc in [')',']','>']:
            j = lifo['x'].pop()
            graph.add_edge(i, j, label='=', type='basepair', len=1)

    # ADD BACKBONE
    lastgoodnode =  0
    for i in range(len(annot[0])-1):
        a,b = annot[0][i]=='.', annot[0][i+1]=='.'
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




def most_common_nucs(alignment):
    def getch(x):
        col = alignment[:,x].tolist()
        counts = Counter( col )
        for k,v in counts.most_common():
            if k in "ACGU":
                return k
    return Map(getch, Range(alignment.shape[1]))




def mainchainentropy(fname, alignment, annot):
    '''
        cleaning up the graph generation a bit,
        will keep a main alignment and just encode that together with the entropy...
        hopefully will keep a consensus to use for printing later..
        maybe i can also use sscons as vec-features
    '''
    graph = nx.Graph()
    lifo = defaultdict(list)
    nucs = most_common_nucs(alignment)
    simple_sscons = annot[0].replace(',',':')
    simple_sscons = simple_sscons.replace('-',':')
    simple_sscons = simple_sscons.replace('_',':')


    annot[0] = simple_sscons # this might be weird in the future hmmmm
    sequence = ''
    conSS = ''
    for i, (struct,nuc) in enumerate(zip(simple_sscons,nucs)):
        if struct != '.' and nuc != None: # ATTENTION! some structures have a :  but there is not even one nucleotide listed
            try:
                conSS += struct
                sequence+=nuc
                myv  = [ ord(rr[i]) for rr in annot ]
                graph.add_node(i, label=nuc, vec=myv)
                # handle hydrogen bonds
                if struct in ['(','[','<']:
                    lifo['x'].append(i)
                if struct in [')',']','>']:
                    j = lifo['x'].pop()
                    graph.add_edge(i, j, label='=', type='basepair', len=1)
            except:
                print("ERROR IN FILE", fname)

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



