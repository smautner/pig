from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from ubergauss import tools as ut
from yoda.graphs import ali2graph
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

