from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from ubergauss import tools as ut
from yoda.graphs import ali2graph
from collections import defaultdict
from yoda.graphs.ali2graph import manifest_sequences
from yoda.alignments import filein, clans, alignment
import numpy as np

def load_rfam(seedpath = '~/Rfam.seed.utf8', full = False, add_cov = '~/rfam/test2'):
    alignments = filein.readseedfile(ut.fixpath( seedpath))
    labels = clans.getlabels(alignments)
    if not full:
        oklabel = labels != 0
        alignments = [a for a,ok in zip(alignments,oklabel) if ok > 0]
        labels = labels[oklabel]
    if add_cov:
        alignments = filein.addcov_rfam(alignments, add_cov)

    alignments = ut.xmap(ali2graph.rfam_clean, alignments)
    #check_labels(alignments,labels)
    for a,label in zip(alignments, labels):
        a.clusterlabel = label
    return alignments, labels


def size_filter(alis, labels, cutoff):
    glens = np.array([len(ali.graph) for ali in alis])
    alis = [ali for ali,cnt in zip(alis,glens) if cnt < cutoff]
    labels = labels[glens < cutoff]

    counts = np.unique(labels, return_counts= True)
    badones = [l for l, count in zip(*counts) if count == 1 ]
    alis, labels = Transpose([(ali,l) for ali,l in
                              zip(alis, labels) if l not in badones])
    return alis, labels


def filter_by_seqcount(alis):
    d = defaultdict(list)
    for  a in alis:
        if a.clusterlabel > 0:
            d[a.clusterlabel].append(a.alignment.shape[0])
    bad = [ k for k,v in d.items() if any([vv < 10 for vv in v])]
    ok = [ k for k,v in d.items() if all([vv > 9 for vv in v])]
    return ok,bad



def check_labels(alignments, labels):
    for a,l in zip(alignments[:5], labels[:5]):
        print(f"{l=}  {a.gf[f'ID'][3:]=} {a.gf[f'AC'].split()[1:]=}")


def subsample(a,l, num = 10):
    return a[:num], l[:num]


if __name__ == "__main__":
    z = load_rfam()[0]
    z = [zz for zz in z if zz.pseudoknot]
    import structout as so
    for zz in z:
        try:
            #so.rnagraph.RNAprint(zz.graph)
            so.gprint(zz.graph, size = 70)
        except:
            pass




#####################################################
# we build something like this:

    # (Pdb) df.iloc[2]
    # sequence         [G, A, A, A, U, C, U, U, U, C, C, U, G, C, U, ...
    # structure        [., ., (, (, (, (, (, (, (, (, (, (, (, (, (, ...
    # pos1id           [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1...
    # pos2id           [113, 112, 111, 110, 109, 108, 107, 106, 105, ...
    # pk               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    # msa_id                                                           2
    # set                                                          train
    # is_pdb                                                       False
    # has_pk                                                       False
    # has_multiplet                                                False
    # has_nc                                                       False
    # has_msa                                                      False
    # family                                                      MIR828
    # Id                                                               2

# noncanonical

canonical_pairs = ['GC', 'CG', 'AU', 'UA', 'GU', 'UG']
def is_nc(a,b):
    return a+b in canonical_pairs

def has_nc(s,x1,x2):
    return any([is_nc(s[xx1],s[xx2]) for xx1,xx2 in zip(x1,x2)])

import networkx as nx
def ali_to_rnaformerlinedict(id:int, thing):
    # initialize, set values that ill ignore but rnaformer might want to see
    ali, label = thing
    ret = {}
    ret['is_pdb'] = False
    ret['has_pk'] = False
    ret['has_multiplet'] = False
    ret['has_msa'] = False

    # important stuff first..
    ret['sequence'] = list(ali.graph.graph['sequence'])
    ret['structure'] = list(ali.graph.graph['structure'])
    g = nx.convert_node_labels_to_integers(ali.graph)
    try:
        ret['pos1id'], ret['pos2id'] = Transpose([(a,b) for a,b,c in g.edges(data=True) if c['type'] =='basepair'])
    except:
        # sometimes there is no structure, e.g. RF00277
        ret['pos1id'], ret['pos2id'] = [],[]
    ret['pk'] = [0]*len(ret['pos1id'])

    # not so important stuff now...
    ret['family'] = ali.get_fam_id()[3:] #ali.get_fam_name()
    ret['Id'] = id
    ret['has_nc'] = has_nc(ret['sequence'], ret['pos2id'], ret['pos1id'])
    ret['msa_id'] = 0
    ret['set'] = label
    return ret


import pandas as pd
def make_rnaformerdata(numneg = 0):
    ali,labels = load_rfam(add_cov='', full = numneg>0)
    ali,labels = size_filter(ali,labels,400)

    if numneg > 0:
        labels = np.array(labels)
        loc_pos = np.where(labels == 0)
        loc_neg = np.where(labels != 0)
        np.random.shuffle(loc_neg)
        loc_neg = loc_neg[:numneg]
        loc = np.hstack((loc_pos, loc_neg))
        ali, labels = Transpose([(ali[z], labels[z]) for z in loc.flatten() ])


    ali,labels = manifest_sequences(ali,labels,instances=100000, mp=True)
    dict_dat = Map(ali_to_rnaformerlinedict, *Transpose(enumerate(zip(ali,labels))))
    df = pd.DataFrame(dict_dat)
    print(df)
    df.to_pickle('rnaformer_rfam_100neg.plk')
    return df




