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



'''
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
'''


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

letterpairs = { chr(97+i):chr(65+i) for i in range(26) }

def  rfam_clean(ali):

    graph = nx.Graph()
    lifo = []
    lifo_pseudoknots = defaultdict(list)
    assert 'RF' in ali.gc

    ali.gc['clean_structure'] =  clean_structure (ali.gc['SS_cons'])


    sequence = ''
    conSS = ''
    ali.pseudoknot = any([a in letterpairs.keys() for a in ali.gc['clean_structure']])

    for i, (struct,nuc) in enumerate(zip(ali.gc['clean_structure'],ali.gc['RF'].upper())):
        if nuc != '.': # ATTENTION! some structures have a :  but there is not even one nucleotide listed
            try:
                conSS += struct
                sequence+=nuc
                # myv  = [ ord(ali.gc[k][i]) for k in ali.gc.keys() ]
                graph.add_node(i, label=nuc, vec=[])
                # handle hydrogen bonds
                if struct  == '(':
                    lifo.append(i)
                if struct == ')':
                    j = lifo.pop()
                    graph.add_edge(i, j, label='=', type='basepair', len=1)
                if ali.pseudoknot:
                    if struct in letterpairs.values():
                        lifo_pseudoknots[struct].append(i)
                    if struct in letterpairs.keys():
                        j = lifo_pseudoknots[letterpairs[struct]].pop()
                        graph.add_edge(i, j, label='=', type='basepair', len=1)

            except Exception as e:
                print("ERROR IN FILE", ali.fname,e)
                print (letterpairs, ali.gc['clean_structure'])

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
    ctr.pop('-',None)
    su = sum(ctr.values())
    return [(k, v/su) for k,v in ctr.most_common()]


'''
def rfam_graph_structure_deco(ali):
    nuc_distribution = {i:normalize(Counter(ali.alignment[:,i])) for i in list(ali.graph)}

    def dist2vec(nudi):
        nudi = dict(nudi)
        nudi['R'] = sum([nudi.get(x,0) for x in 'A G'.split()])
        nudi['Y'] = sum([nudi.get(x,0) for x in 'C U'.split()])
        levels =  np.array([nudi.get(x,0) for x in f"A G C U Y R".split()])
        thresh = [.25,.5,.75,.95]
        print(f"{nudi=}")
        return np.array([ e > t for e in levels for t in thresh])

    for n in ali.graph:
        ali.graph.nodes[n]['label'] = '1'
        ali.graph.nodes[n]['vec'] = dist2vec(nuc_distribution[n])

    return ali

'''

def rfam_graph_decoration(ali, RY_thresh = .6, nuc_thresh = .85,
                          conservation = [.50,.75,.90,.95],
                          covariance = False,
                          nucleotideRNAFM = False ,
                          sloppy = False,
                          progress = 0,
                          cons_raw = False,
                          fake_nodes = False):
    vec = []
    for n in ali.graph.nodes:
        ali.graph.nodes[n]['vec'] = []
    nuc_distribution = {i:normalize(Counter(ali.alignment[:,i])) for i in list(ali.graph)}

    if RY_thresh or nuc_thresh:
        set_base_label(ali,nuc_distribution, nuc_thresh= nuc_thresh,RY_thresh= RY_thresh)

    if conservation:
        vec+=[0]*len(conservation)
        vec_add_conservation(ali, nuc_distribution, conservation)

    if covariance:
        set_base_covariance(ali, covariance)

    if sloppy:
        vec+=[0]*2
        vec_add_sloppy(ali)

    if progress:
        v_used = vec_add_progress(ali,progress)
        vec += [0]*v_used
    if cons_raw:
        vec += [0]
        vec_add_cons_raw(ali, nuc_distribution)

    if nucleotideRNAFM:
        vec += [0]*620
        vec_add_nucleotideembedding(ali)

    if fake_nodes:
        # add this last so that we know how long the vecs of the default nodes are
        add_fake_nodes(ali, vec)

    #if 'vec' in ali.graph.nodes[list(ali.graph.nodes)]:
    for n in ali.graph.nodes:
        ali.graph.nodes[n]['vec'] = np.array(ali.graph.nodes[n]['vec'])

    return ali


def vec_add_progress(ali, progress):
    proglen = progress if progress > 1 else 25
    nodes = list(ali.graph)
    for i,n in enumerate(nodes):
        prog = i/len(nodes)
        prog = int(prog*proglen)
        base = [0]*proglen
        base [prog] = 1
        ali.graph.nodes[n]['vec'] += base
    return proglen


def vec_add_cons_raw(ali, nuc_distribution):
    nodes = list(ali.graph)
    for n in nodes:
        add = 0
        if ali.graph.nodes[n]['label'] in 'ACGU':
            add = nuc_distribution[n][1]
        if ali.graph.nodes[n]['label'] == 'Y':
            nf = dict(nuc_distribution[n])
            add = nf['C'] + nf['U']
        if ali.graph.nodes[n]['label'] == 'R':
            nf = dict(nuc_distribution[n])
            add = nf['A'] + nf['G']
        ali.graph.nodes[n]['vec'] += [ add ]


def add_fake_nodes(ali,vec):
    # add fakenodes
    pairdict = dict(Flatten([[(a, b), (b, a)] for a, b, data in ali.graph.edges(data=True) if data['label'] == '=']))
    nodes = list(ali.graph)
    for n in range(len(nodes) - 1):
        # i and i+1 are definitely connected...
        i, j = nodes[n], nodes[n + 1]
        a = pairdict.get(i, -1)
        b = pairdict.get(j, -1)
        dict_of_a_or_empty = ali.graph._adj.get(a, {})
        if b in dict_of_a_or_empty and i < a:  # i<a prevents foring it twice
            newnode = max(ali.graph) + 1
            ali.graph.add_node(newnode, label='o', vec=[])
            ali.graph.add_edges_from([(newnode, z) for z in [i, j, a, b]], label='*')
            if len(vec) > 0:
                ali.graph.nodes[newnode]['vec'] = vec


def vec_add_sloppy(ali):
    for i in list(ali.graph):
        ali.graph.nodes[i]['vec'] += [0, 0]
    if '(' in ali.alignment:
        print(ali.fname)
    for a, b, data in ali.graph.edges(data=True):

        if data['label'] == '=':
            for z in zip(ali.alignment[:, a], ali.alignment[:, b]):

                if sorted(z) == list('GU'):
                    ali.graph.nodes[a]['vec'][-2] = 1
                    ali.graph.nodes[b]['vec'][-2] = 1
                    break

            for z in zip(ali.alignment[:, a], ali.alignment[:, b]):
                if sorted(z) == list('CU') or sorted(z) == list('AG') or sorted(z) == list('AC'):
                    ali.graph.nodes[a]['vec'][-1] = 1
                    ali.graph.nodes[b]['vec'][-1] = 1
                    break


def set_base_covariance(ali, covariance):
            for a,b,v in ali.rscape:
                    if v < covariance:
                        if a in ali.graph and b in ali.graph and  ali.graph.has_edge(a,b):
                            ali.graph.nodes[a]['label']= 'N'
                            ali.graph.nodes[b]['label']= 'N'
                        else:
                            pass #print(f"cov missmatch:{ ali.fname} {a} {b}")

def vec_add_conservation(ali, nuc_distribution, conservation):
    for i in list(ali.graph):
        nf = {a: 0 for a in f"ACGU"}
        nf.update(dict(nuc_distribution[i]))
        label = ali.graph.nodes[i]['label']
        percentage = nf[label] if label in 'ACGU' else (nf['A'] + nf['G'] if label == 'R' else nf['C'] + nf['U'])
        cons = [int(percentage > x) for x in conservation]
        ali.graph.nodes[i]['vec'] += cons


def set_base_label(ali, nuc_distribution,  nuc_thresh = .85 , RY_thresh = .6):
    thresh = nuc_thresh
    bonus = RY_thresh
    def chooselabel(nudi):
        nudid = dict(nudi)
        # return max(nudi.items(), key = lambda x:x[1])[0]
        k,v =  nudi[0]
        if v > thresh:
             return k

        if sum([nudid.get(x,0) for x in 'A G'.split()]) > bonus:
            return f'R'
        if sum([nudid.get(x,0) for x in 'C U'.split()]) > bonus:
            return f"Y"

        return f'0'

    for n in ali.graph:
        ali.graph.nodes[n]['label'] = chooselabel(nuc_distribution[n])
        # ali.graph.nodes[n]['vec'] = dist2vec(nuc_distribution[n])
    return ali



def vec_add_nucleotideembedding(ali):
    # assert len(ali) == 408, 'there are files called 0..407'
    vectors = np.load(f"/home/ubuntu/RNAEMBED/RNA-FM/redevelop/results/representations/{ali.gf['AC'].split()[1]}.npy")
    for n,vec in zip(list(ali.graph), vectors):
        ali.graph.nodes[n]['vec'] += vec.tolist()
    return ali


import copy, random
def manifest_subgraph(ali,n):
    ali = copy.deepcopy(ali)
    for nid in list(ali.graph):
        ali.graph.nodes[nid]['label'] = ali.alignment[n][nid]
    return ali

def manifest_subgraphs(a_l,maxgraphs = 100):
    ali, label = a_l
    ali_n_graphs = ali.alignment.shape[0]
    graphs_to_manifest = Range(ali_n_graphs)
    if ali_n_graphs > maxgraphs:
        random.shuffle(graphs_to_manifest)
        graphs_to_manifest= graphs_to_manifest[:maxgraphs]
    return [(manifest_subgraph(ali,x),label) for x in graphs_to_manifest ]

def manifest_sequences(alignments, labels, instances = 10, mp = False):
    mapper = ut.xmap if mp else Map
    a,labels = Transpose( Flatten ( mapper(manifest_subgraphs , zip(alignments,labels),
                                                maxgraphs = instances)))
    return a, labels


