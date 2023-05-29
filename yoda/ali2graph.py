from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import numpy as np
import networkx as nx
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
        ATTENTION: . is sometimes in preserved columns so this is  bad-ish
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

import structout as so

import warnings
warnings.filterwarnings("ignore")

def scclust(ali):
    '''
    we try to recreate rna sc clust
    '''
    graph = nx.Graph()


    # adding nodes
    for i,(e,c) in enumerate(zip(ali.alignment[0], ali.covariance)):
        if e == f'.':
            continue
        symbol = e if c < .05 else f'C'
        graph.add_node(i, label=symbol)

    # adding structure
    for i,j in ali.pairs:
        if i in graph and j in graph:
            graph.add_edge(i, j, label='=', len=1)
        else:
            ali.skippedAPair = True
    # adding backbone
    nodes = list(graph.nodes())
    nodes.sort()
    for i in range(len(nodes)-1):
        graph.add_edge(nodes[i], nodes[i+1], label='-', len=1)

    # add fakenodes
    pairdict = dict(ali.pairs)
    for i in range(len(nodes)-1):
        # i and i+1 are definitely connected...
        i,j = nodes[i],nodes[i+1]
        a = pairdict.get(i,-1)
        b = pairdict.get(j,-1)
        dict_of_a_or_empty = graph._adj.get(a,{})
        if b in dict_of_a_or_empty:
            newnode = max(nodes)+1
            graph.add_node(newnode, label=f'o')
            graph.add_edges_from([(newnode,z) for z in [i,j,a,b]],label = f'*')


    # so.gprint(graph, size = 30)
        #graph.add_edge(nodes[i], nodes[i+1], label='-', len=1)
    return graph



def rfamseedfilebasic(ali, structure  = 'SS_cons'):

    graph = nx.Graph()
    lifo = defaultdict(list)
    if f'RF' not in ali.gc:
        print(f"{ ali.gf = }")
        print(f"{ ali.gc = }")
    nucs = ali.gc[f'RF'].upper()
    simple_sscons = ali.gc[structure]
    simple_sscons = simple_sscons.replace(',',':')
    simple_sscons = simple_sscons.replace('-',':')
    simple_sscons = simple_sscons.replace('_',':')
    ali.gc[structure] = simple_sscons

    sequence = ''
    conSS = ''

    for i, (struct,nuc) in enumerate(zip(simple_sscons,nucs)):
        if nuc != f'.': # ATTENTION! some structures have a :  but there is not even one nucleotide listed
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
