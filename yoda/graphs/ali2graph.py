from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import numpy as np
import networkx as nx
import eden.graph as eg # eden-kernel in pip
from collections import defaultdict, Counter
import ubergauss.tools as ut

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
        if ali.graph.nodes[nid]['label'] in f"AGCU":
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


##################################
# abstract graph
###################################


def discretize_nodecount(cutoff_a_b,node_ids):
    count = len(node_ids)
    if count < cutoff_a_b[0]:
        return f"s"
    if count < cutoff_a_b[1]:
        return f"m"
    return 'l'

from collections import Counter

def avgcons(a):
    def cons(nucs):
        c = Counter(nucs.reshape(-1).tolist())
        all = sum([c.get(x,0) for x in f"AUCG-"])
        return max([c.get(x,0)/all for x in 'AUCG'])
    return np.mean([cons(colu) for colu in a.T])

def decorateAbstractgraph(ali, len_single =[2,4],len_double =[6,10]):
    gr = abstractgraph(ali)
    for n in gr:
        d = gr.nodes[n]
        if d['label']== 'S':
            d['label'] +=  discretize_nodecount(len_single,d['columns'])
        elif d['label']== 'D':
            d['label'] +=  discretize_nodecount(len_double,d['columns'])

        if not d['columns']:
            d['vec'] =  [0]
        else:
            d['vec'] = [avgcons(ali.alignment[:,d['columns']])]
    return gr

def abstractgraph(ali):
    g = nx.Graph()

    singleslabel = 'S'
    doublelabel = 'D'
    gid = 0
    g.add_node(gid, label = singleslabel, columns = []) # empty ss node

    col_to_gid  = {}

    for colid in ali.graph:
        # we might have visited the node before:
        get_hbond = lambda x:  [(u, v) for u, v, attr in ali.graph.edges(x, data=True) if attr['label'] == '=']
        hbond = get_hbond(colid)
        cnode = g.nodes[gid]

        # lets deal with the cases one by one:
        # 1. we are in single mode and continue
        if len(hbond) == 0 and cnode['label'] == singleslabel:
            cnode['columns'].append(colid)

        # 2. we are in singlemode and switch to double node
        elif len(hbond) > 0 and cnode['label'] == singleslabel:
            newid = len(g)
            g.add_node(newid, label = doublelabel, columns = [colid])
            g.add_edge(gid, newid,label = '-')
            gid = newid

        # 3. we are in double and switch to single
        elif len(hbond) == 0 and cnode['label'] == doublelabel:
            newid = len(g)
            g.add_node(newid, label = singleslabel, columns = [colid])
            g.add_edge(gid, newid,label = '-')
            gid = newid

        # 4. we are in double and stay double
        elif len(hbond) > 0 and cnode['label'] == doublelabel:
            # there are many cases here i think ...
            # 4.1: the stacking case should be easy:
            def part(pair,cid):
                pair = pair[0]
                assert cid in pair
                return pair[0] if cid == pair[1] else pair[1]

            def check_stack(hbond, code):
                lastcid = max(cnode['columns'])
                lasthbond = get_hbond(lastcid)
                lastpartner = part(lasthbond, lastcid)
                cpartner = part(hbond, colid)
                return ali.graph.has_edge(cpartner,lastpartner)

            if check_stack(hbond, cnode):
                cnode['columns'].append(colid)
            else:
                # 4.2 there is no stacking
                newid = len(g)
                g.add_node(newid, label = singleslabel, columns = [])
                g.add_node(newid+1, label = doublelabel, columns = [colid])
                g.add_edge(gid, newid,label = '-')
                g.add_edge(newid+1, newid,label = '-')
                gid = newid+1

        if len(hbond) >0:
            col_to_gid[colid] = gid

    # now we need to connect the doubles to each other..

    hbonds = [(u, v) for u, v, attr in ali.graph.edges(data=True) if attr['label'] == '=']
    for a,b in hbonds:
        g.add_edge(col_to_gid[a],col_to_gid[b] , label = '-' )

    return g


#############
# new abstract graph
############

def get_nuc_cons(nucdic, RYT):
    nucdic.pop('-',0)
    allnuc  = sum(nucdic.values())
    RYT_real  = allnuc*RYT
    char,cnt = nucdic.most_common(1)[0]
    if cnt > RYT_real:
        return char, cnt/allnuc
    rcnt = sum([nucdic.get(x,0) for x in 'AG'])
    if rcnt > RYT_real:
        return 'R', rcnt/allnuc
    rcnt = sum([nucdic.get(x,0) for x in 'CU'])
    if rcnt > RYT_real:
        return 'Y', rcnt/allnuc
    return char, cnt/allnuc

def dillute_cons(g, nuc_cons, dillution_fac1, dillution_fac2):
    for n in g.nodes:
        n_dist = nx.single_source_shortest_path_length(g,n, cutoff=2)
        at0 = nuc_cons[n][1]
        at1 = np.mean([nuc_cons[k][1] for k,v in n_dist.items() if v == 1])
        at2 = np.mean([nuc_cons[k][1] for k,v in n_dist.items() if v == 2])
        newcons  = (at0 + at1*dillution_fac1+ at2*dillution_fac2) / (1+dillution_fac1+dillution_fac2)
        nuc_cons[n] = (nuc_cons[n][0],newcons)
    return nuc_cons

def relabel(g,nuc_cons, cons_thresh):
    '''
    put either a letter, d or s
    '''
    isbound = lambda x:  len([(u, v) for u, v, attr in g.edges(x, data=True) if attr['label'] == '=']) > 0

    for n in g.nodes:
        if nuc_cons[n][1] > cons_thresh:
            label = nuc_cons[n][0]
        elif isbound(n):
            label = 'D'
        else:
            label = "S"
        g.nodes[n]['label'] =  label
    return g

def contract(g):
    nodes = list(g.nodes())
    for i,n in enumerate(nodes[:-1]):
        next_node = g.nodes[nodes[i+1]]
        if g.nodes[n]['label'] == 'S' == next_node['label'] or g.nodes[n]['label'] == 'D' == next_node['label']:
            se = g.nodes[n].get('cols',[])
            se.append(n)
            g = nx.contracted_nodes(g,nodes[i+1],n, self_loops= False)
            g.nodes[nodes[i+1]]['cols'] = se
    return g

def relabel_nonpreserved(g, cutS, cutD):
    for n in g.nodes:
        label = g.nodes[n]['label']
        if   label in 'SD':
             g.nodes[n]['label'] += discretize_nodecount(cutS if label =='S' else cutD , g.nodes[n].get('cols',[n]))
    return g

def get_coarse(ali, RY_thresh=.93 , dillution_fac1=.4, dillution_fac2=.2,
               cons_thresh=.8, cutS1 = 2, cutS2 =5, cutD1 =5, cutD2 = 10, **kwargs):
    nucleotide_dict = {n:Counter(ali.alignment[:,n].tolist()) for n in ali.graph.nodes}
    nuc_cons  = {k:get_nuc_cons(v, RY_thresh) for k,v in nucleotide_dict.items()}
    nuc_cons = dillute_cons(ali.graph.copy(), nuc_cons, dillution_fac1, dillution_fac2)
    g = relabel(ali.graph,nuc_cons, cons_thresh)
    g = contract(g)
    g = relabel_nonpreserved(g, (cutS1, cutS1), (cutD1, cutD2))

    for n,d in g.nodes(data=True):
        d.pop('contraction','')
    for a,b,d in g.edges(data=True):
        d.pop('contraction','')
    return g


########################################
# just cons
#################################

def writecons(ali):
    nucleotide_dict = {n:Counter(ali.alignment[:,n].tolist()) for n in ali.graph.nodes}
    for node in ali.graph.nodes():
        mynd = nucleotide_dict[node]
        mynd.pop('-',0)
        allnuc  = sum(mynd.values())
        char,cnt = mynd.most_common(1)[0]
        ali.graph.nodes[node]['weight'] = cnt/allnuc
    return ali.graph.copy()




from sklearn.metrics import pairwise_distances
from ubergauss import tools as ut
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
def multiGraphCache(alis, c = 0):
    mkMultiGraphCache(alis,c = c)
    return readMultiGraphCache(alis)

def mkMultiGraphCache(alis,c=0):
    # check if file exists
    if os.path.exists('MGCache'):
        return
    print('doing the cache')
    dists = ut.xxmap(multiGraphDistances,alis, cluster=c)
    dists = {a.label: d for a,d in zip(alis, dists)}
    if not c:
        ut.dumpfile({'MGDIST':dists}, 'MGCache')
    else:
        ut.dumpfile({'MGClusterlabels':dists}, 'MGCache')


def readMultiGraphCache(alis):
    cache = ut.loadfile('MGCache')

    for k in cache.keys():
        for a in alis:
            a.__dict__[k] = cache[k][a.label]

    return alis

def multiGraphDistances(ali, cluster = 0):
    def seq2graph(seq):
        g = ali.graph.copy()
        for n in g.nodes:
            g.nodes[n]['label'] = seq[n]
        return g
    graphs = [seq2graph(s) for s in ali.alignment]
    vectors = eg.vectorize(graphs)
    if cluster:
        n_clusters = min(int(ali.alignment.shape[0]/cluster)+1, 10)
        return KMeans(n_clusters = n_clusters ).fit_predict(vectors)
    return vectors

def multiGraph(ali, clusterSize = 15):
    '''
    some alignments are too large, i.e. contain many sequences
    since we choose only one representative, it can not catch all the variance of those many sequences
    therefore we build one representative per CLUSTERSIZE sequences.



    make sure weight annotations are there, i.e. run this after writeocons

    for each sequence in the alignment:
    - materialize a graph where the nodes are labeled according to the sequence

    vectorize graphs via the kernel, using the weight attribute for weighting the nodes
    make a distance matrix of the vectors
    cluster via scikit-learn affinity propagation

    for each cluster materialize a graph, where the labels correspond to the most frequent entry in the respective sequence column
    merge all these networkx graphs

    print the sequences of new graphs
    '''
    # assert 'weight' in ali.graph.nodes[0]


    # cluster on the sequences

    if not 'MGClusterlabels' in ali.__dict__:
        n_clusters = min(int(ali.alignment.shape[0]/clusterSize)+1, 10)
        ali.MGClusterlabels = KMeans(n_clusters = n_clusters ).fit_predict(ali.MGDIST)


    # clusterLabels = AgglomerativeClustering(n_clusters = int(ali.alignment.shape[0]/clusterSize)+1).fit_predict(ali.MGDIST)

    # deal with the clusters
    g = nx.Graph()
    # print(f"{ Counter(clusterLabels)=}")
    for clusterId in np.unique(ali.MGClusterlabels):
        # find the most common labels
        g2 = ali.graph.copy()
        sequences = ali.alignment[ali.MGClusterlabels==clusterId]
        def get_col(n):
            z= Counter(sequences[:,n])
            z.pop('-',None)
            return z.most_common(1)[0][0] if z else '-'
        labels = [ get_col(n) for n in ali.graph.nodes]

        # if its blank we better remove the node
        rmlist = [n for n,l in zip(g2.nodes, labels) if l == '-']
        for n in rmlist:
            rmnode(g2,n)
        # relabel the thing
        for n,l in zip(g2.nodes, labels):
            if l != '-':
                g2.nodes[n]['label'] = l

        # merge the graphs
        newGraph = nx.convert_node_labels_to_integers(g2, first_label=len(g))
        g = nx.compose(g, newGraph)
    return g


def rmnode(G,v):
    '''
    1. find the neighbors of v
    2. if there are 2 neighbors, delete v and connect the neighbors
    '''
    neighs = list(G.neighbors(v))

    # assert len(neighs) < 3
        #nucleotide_dict = {n:Counter(ali.alignment[:,n].tolist()) for n in ali.graph.nodes}

    neighs = [n for n in neighs if G[v][n]['label'] != '=']


    if len(neighs) == 2:
        a,b = neighs
        G.add_edge(a,b, label = '-')

    # else we have a dangling end
    G.remove_node(v)
    return G

def set_weight_label(ali, RYthresh=0):
    '''
    sets weight and label
    '''
    nucleotide_dict = {n:Counter(ali.alignment[:,n].tolist()) for n in ali.graph.nodes}

    for node in ali.graph.nodes():
        mynd = nucleotide_dict[node]
        mynd.pop('-',0)
        allnuc  = sum(mynd.values())
        char,cnt = mynd.most_common(1)[0]
        conservation = cnt/allnuc
        if conservation > RYthresh:
            # we can use the label and all is ok
            ali.graph.nodes[node]['weight'] = conservation
        else:
            # we need to relabel
            R = sum([mynd.get(x,0) for x in 'A G'.split()])
            Y = sum([mynd.get(x,0) for x in 'C U'.split()])
            cnt,nuc = (R,'R') if  R > Y else (Y,'Y')
            ali.graph.nodes[node]['weight'] = cnt/allnuc
            ali.graph.nodes[node]['label'] = nuc


    return ali.graph.copy()



def set_weight(g, consThresh= .97,  bad_weight = 0.15):
    for node in g.nodes:
        if g.nodes[node]['weight'] < consThresh:
            g.nodes[node]['weight'] = bad_weight
        else:
            g.nodes[node]['weight'] = 1 #cnt/allnuc
    return g

def nearstem(g, boost_range = 1, boost_thresh = .5, boost_weight = 1):
    isbound = lambda x:  len([(u, v) for u, v, attr in g.edges(x, data=True) if attr['label'] == '=']) > 0
    if boost_range:
        weights = [g.nodes[n]['weight'] for n in g.nodes]
        if np.mean(weights) > boost_thresh:
            bound = set([n for n in g.nodes if isbound(n)])
            for n in g.nodes:
                if n not in bound and g.nodes[n]['weight'] > .9:
                    neighs = set( nx.single_source_shortest_path_length(g,n,
                                                    cutoff=boost_range).keys())
                    # print(f"{neighs=}")
                    if neighs & bound:
                        g.nodes[n]['weight']+= boost_weight
    return g



def donest(g):
    for a, b, data in g.edges(data=True):
        if data['label'] == '=':
            data['nesting'] = True
    return g

def dillute(g, dilute1=.25, dilute2=.75, fix_edges = True):
    d = {}
    for n in g.nodes:
        n_dist = nx.single_source_shortest_path_length(g,n, cutoff=2)
        at0 = g.nodes[n]['weight']
        at1 = np.mean([g.nodes[k]['weight'] for k,v in n_dist.items() if v == 1])
        at2 = np.mean([g.nodes[k]['weight'] for k,v in n_dist.items() if v == 2])
        newcons  = (at0 + at1*dilute1+ at2*dilute2) / (1+dilute1+dilute2)
        d[n]= newcons

    for n in g.nodes:
        g.nodes[n]['weight'] = d[n]

    if fix_edges:
        for a,b in g.edges():
            neighweight = g.nodes[a]['weight'] + g.nodes[b]['weight']
            g[a][b]['weight'] = neighweight/2
    return g








def get_histograms(ali):
    nucleotide_dict = {n:Counter(ali.alignment[:,n].tolist()) for n in ali.graph.nodes}

    def hist(node):
        mynd = nucleotide_dict[node]
        mynd.pop('-',0)
        allnuc  = sum(mynd.values())
        return [ mynd.get(nuc,0)/allnuc  for nuc in 'AUGC']

    return Map(hist,ali.graph.nodes())
