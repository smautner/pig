from ubergauss import tools as ut
from yoda.graphs import ali2graph
from yoda.alignments import filein, clans, alignment


def load_rfam(full = False):
    alignments = filein.readseedfile(ut.fixpath( f'~/Rfam.seed.utf8'))
    labels = clans.getlabels(alignments)

    if not full:
        oklabel = labels != 0
        labels = labels[oklabel]
        alignments = [a for a,ok in zip(alignments,labels) if ok]

    alignments = filein.addcov_rfam(alignments)

    alignments = ut.xmap(ali2graph.rfam_clean, alignments)

    return alignments, labels



