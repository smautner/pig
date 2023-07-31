from lmz import Map
from sklearn.metrics import silhouette_score
import numpy as np
from ubergauss import tools as ut
from sklearn.metrics import pairwise
from yoda import alignments
from yoda.ml.pairwise_alignments import smith_waterman_score
from eden import sequence as eseq



import structout as so
def score_smithwaterman(ali, labels):
    # ali = ali[:10]
    # labels = labels[:10]
    l= len(ali)
    m = np.ndarray((l,l))
    tasks = [(i,j) for i in range(l) for j in range(i,l)]
    def sim(task):
        a,b = ali[task[0]], ali[task[1]]
        sa = a.graph.graph[f'sequence']
        sb = b.graph.graph[f'sequence']
        return 1-(smith_waterman_score(sa,sb)/min(len(sa), len(sb)))
    res = ut.xmap(sim, tasks)
    for r,(i,j) in zip(res, tasks):
        m[i,j] = r
        m[j,i] = r
    return silhouette_score(m, labels, metric='precomputed')


def gapped_kmers(ali, labels):
    ali = eseq.vectorize([a.graph.graph['sequence'] for a in ali])
    m = pairwise.cosine_distances(ali)
    return silhouette_score(m, labels, metric='precomputed')


from yoda.graphs import vectorize_alignments
from yoda.ml import get_distance_method
def simple_graph( alignments,labels):
    vectors = vectorize_alignments(alignments, mp=True)
    # di = get_distance_method(f'linkern')(vectors)
    return silhouette_score(vectors, labels)


if __name__ == f"__main__":
    data = alignments.load_rfam()
    print(f"{score_smithwaterman(*data)=}")
    print(f"{gapped_kmers(*data)=}")
    print(f"{simple_graph(*data)=}")

