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
            graph.add_node(newnode, label='o')
            graph.add_edges_from([(newnode,z) for z in [i,j,a,b]],label = '*')


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


def clean_structure(struct):
    struct = struct.replace(',',':')
    struct = struct.replace('-',':')
    struct = struct.replace('_',':')
    struct = struct.replace('[','(')
    struct = struct.replace('<','(')
    struct = struct.replace('{','(')
    struct = struct.replace(']',')')
    struct = struct.replace('>',')')
    struct = struct.replace('}',')')
    return  struct



def  rfam_clean(ali):

    graph = nx.Graph()
    lifo = []
    assert f'RF' in ali.gc

    ali.gc[f'clean_structure'] =  clean_structure (ali.gc[f'SS_cons'])
    sequence = ''
    conSS = ''

    for i, (struct,nuc) in enumerate(zip(ali.gc[f'clean_structure'],ali.gc[f'RF'].upper())):
        if nuc != '.': # ATTENTION! some structures have a :  but there is not even one nucleotide listed
            try:
                conSS += struct
                sequence+=nuc
                # myv  = [ ord(ali.gc[k][i]) for k in ali.gc.keys() ]
                graph.add_node(i, label=nuc, vec=[])
                # handle hydrogen bonds
                if struct  == f'(':
                    lifo.append(i)
                if struct == f')':
                    j = lifo.pop()
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

    ali.graph = graph
    ali.graphnodes = list(graph)

    return ali


def normalize(ctr):
    ctr.pop(f'-',None)
    su = sum(ctr.values())
    return [(k, v/su) for k,v in ctr.most_common()]

def rfam_graph_decoration(ali, RY_thresh = .1,
                          conservation = [.50,.75,.90,.95],
                          covariance = False,
                          sloppy = False,
                          fake_nodes = False):
    vec = []
    if RY_thresh or conservation or sloppy:
        nuc_distribution = {i:normalize(Counter(ali.alignment[:,i])) for i in list(ali.graph)}

    if RY_thresh:
        def assignbase(i):
            nucleotide_frequency = nuc_distribution[i]
            nucleotide_frequency+=[('',0)]
            if nucleotide_frequency[1][1]  > RY_thresh:
                if {nucleotide_frequency[0][0], nucleotide_frequency[1][0]} == set("AG"):
                    ali.graph.nodes[i]['label'] = f'R'
                if {nucleotide_frequency[0][0], nucleotide_frequency[1][0]} == set("CU"):
                    ali.graph.nodes[i]['label'] = f'Y'
        Map(assignbase, list(ali.graph))

    if conservation:
        vec+=[0]*len(conservation)
        for i in list(ali.graph):
            nf = {a:0 for a in f"ACGU"}
            nf.update(dict(nuc_distribution[i]))
            label = ali.graph.nodes[i]['label']
            percentage = nf[label] if label in 'ACGU' else (nf['A']+nf['G'] if label == 'R' else nf['C']+nf['U'])
            cons = [int(percentage > x) for x in conservation]
            ali.graph.nodes[i]['vec']+=cons

    if covariance:
            for a,b,v in ali.rscape:
                if v < .05:
                    if a in ali.graph and b in ali.graph and  ali.graph.has_edge(a,b):
                        ali.graph.nodes[a]['label']= 'N'
                        ali.graph.nodes[b]['label']= 'N'
                    else:
                        pass #print(f"cov missmatch:{ ali.fname} {a} {b}")

    if sloppy:

            vec+=[0]*2
            for i in list(ali.graph):
                ali.graph.nodes[i]['vec']+=[0,0]
            if '(' in ali.alignment:
                 print (ali.fname)
            for a,b,data in ali.graph.edges(data=True):

                if data['label'] ==  '=':
                    for z in zip(ali.alignment[:,a], ali.alignment[:,b]):

                        if sorted(z) == list('GU'):
                            ali.graph.nodes[a]['vec'][-2] = 1
                            ali.graph.nodes[b]['vec'][-2] = 1
                            break

                    for z in zip(ali.alignment[:,a], ali.alignment[:,b]):
                        if sorted(z) == list('CU') or sorted(z) == list('AG') or sorted(z) == list('AC'):
                            ali.graph.nodes[a]['vec'][-1] = 1
                            ali.graph.nodes[b]['vec'][-1] = 1
                            break
    if fake_nodes:
        # add fakenodes
        pairdict = dict(Flatten([[(a,b),(b,a)] for a,b,data in ali.graph.edges(data=True) if data['label']=='=']))
        nodes = list(ali.graph)

        for n in range(len(nodes)-1):
            # i and i+1 are definitely connected...
            i,j = nodes[n],nodes[n+1]
            a = pairdict.get(i,-1)
            b = pairdict.get(j,-1)
            dict_of_a_or_empty = ali.graph._adj.get(a,{})
            if b in dict_of_a_or_empty and i < a: # i<a prevents foring it twice
                newnode = max(ali.graph)+1
                ali.graph.add_node(newnode, label='o',vec=[])
                ali.graph.add_edges_from([(newnode,z) for z in [i,j,a,b]],label = '*')
                if len(vec) > 0:
                    ali.graph.nodes[newnode]['vec']= vec

    #if 'vec' in ali.graph.nodes[list(ali.graph.nodes)]:
    for n in ali.graph.nodes:
        ali.graph.nodes[n]['vec'] = np.array(ali.graph.nodes[n]['vec'])

    return ali
