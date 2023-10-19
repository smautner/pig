from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import umap
from sklearn.metrics import pairwise


def embed(distances, n_dim = 2):
    return umap.UMAP(n_components=n_dim, metric='precomputed',n_neighbors=2).fit_transform(distances)


def get_distance_method(distance_measure):
    if distance_measure == 'euclidean':
        di = pairwise.euclidean_distances
    elif distance_measure == 'overlap':
        di = simpleMl.overlap
    elif distance_measure == 'overlap_nonorm':
        di = lambda x: 1 - simpleMl.overlap(x,nonorm=True)
    elif distance_measure == 'linkern':
        di = lambda x: 10 - pairwise.linear_kernel(x)
    else:
        assert False, 'distance_measure invalid'
    return di


###############
# interpretable output and its helpers
#################
from yoda.alignments import alignment
from collections import Counter
import numpy as np
def _collapse_sequences(mtx,alis: list[alignment]):
    '''
    undoes the manifiest_sequences stuff
    '''
    fam_ids = [a.get_fam_id() for a in alis]

    # do we need to do anything?

    if np.all(np.array(Counter(fam_ids).values()) == 1):
        return mtx, alis

    # first build the matrix...
    fam_ids = np.array(fam_ids)
    fams = np.unique(fam_ids)
    res_mtx = np.zeros((len(fams), len(fams)),dtype = np.float)

    for i,fname in enumerate(fams):
        for j,fname2 in enumerate(fams):
            # breakpoint()
            # res_mtx[i,j] = np.mean(mtx[fam_ids==fname, fam_ids == fname2 ])
            res_mtx[i,j] = np.mean(mtx[fam_ids==fname][:, fam_ids == fname2 ])

    # then find the associated alignments...
    def getali(fam):
        for a in alis:
            if a.get_fam_id() == fam:
                return a

    return res_mtx, Map(getali,fams)


from sklearn.neighbors import NearestNeighbors as NN
def interpretable_output(distmatrix, alignments):
    # collapse the subsampling again
    distmatrix, alignments = _collapse_sequences(distmatrix, alignments)

    # a nearest neighbor model helps us fetch the closest instances
    np.fill_diagonal(distmatrix,0) # just making sure
    dist, indices = NN(n_neighbors = 4).fit(distmatrix).kneighbors(distmatrix)

    # report top X pairs, and say if they are right or wrong..
    dij_list = [ (d,i,j) for (i,(d,j)) in enumerate(zip(dist[:,1],indices[:,1]))]
    dij_list.sort( key = lambda x:x[0])
    donelist = []
    def fali(ali_i):
        ali = alignments[ali_i]
        return f'{ali.get_fam_name()}(cluster {ali.clusterlabel})'

    for d,i,j in dij_list[:20]:
        if (j,i) not in donelist:
            print(f"{fali(i)}, {fali(j)} {d=}")
            donelist.append((i,j))

    # report if a hit is in the top 3 for each query
    hit_in_top3 = []
    for i,a in enumerate(indices[:,1:]):
        hit_in_top3.append( any ( [ alignments[aa].clusterlabel  == alignments[i].clusterlabel for aa in  a ]))

    print(f"{np.mean(hit_in_top3)=}")

