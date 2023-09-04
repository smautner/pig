from ubergauss import tools as ut
from yoda.graphs import ali2graph
from yoda.alignments import filein, clans, alignment


def load_rfam(full = False):
    alignments = filein.readseedfile(ut.fixpath( f'~/Rfam.seed.utf8'))
    labels = clans.getlabels(alignments)

    if not full:
        oklabel = labels != 0
        alignments = [a for a,ok in zip(alignments,oklabel) if ok > 0]
        labels = labels[oklabel]

    alignments = filein.addcov_rfam(alignments)
    alignments = ut.xmap(ali2graph.rfam_clean, alignments)

    #check_labels(alignments,labels)

    return alignments, labels


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

