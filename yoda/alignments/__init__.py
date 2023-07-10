from ubergauss import tools as ut
from yoda import alignments

def load_rfam(full = False):
    alignments = alignments.filein.readseedfile(ut.fixpath( f'~/Rfam.seed.utf8'))
    labels = alignments.clans.getlabels(alignments)

    if not full:
        oklabel = labels != 0
        labels = labels[oklabel]
        alignments = [a for a,ok in zip(alignments,labels) if ok]

    alignments = alignments.filein.addcov_rfam(alignments)

    alignments = ut.xmap(ali2graph.rfam_clean, alignments)

    return alignments, labels
