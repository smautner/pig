import numpy as np
from kiez import Kiez
from sklearn.decomposition import PCA
import sklearn
import ubergauss as ug
# from sklearn.metrics import pairwise_distances
params = '''
kiezK 2 50 1
'''
# metric [ 'cosine', 'euclidean']
# kiezMethod  ['csls', 'dissimlocal', 'localscaling', 'mutualproximity', 'norm']
# pca 0 170 40
#kiezMethod  ['csls', 'dissimlocal', 'localscaling', 'mutualproximity', 'norm', None]
# metric ['cityblock', 'cosine', 'euclidean']

def mkdistances(csr_matrix, pca=False, metric="euclidean", kiezMethod="csls", kiezK=5, kiezPresort = 50, kiezBug = 0):
    """
    Transform a CSR matrix into a pairwise distances matrix.

    - apply pca?
    - apply a distance method
    - perform correction via kiez

    return distances, indices
    """


    X = csr_matrix.toarray()

    if pca: # pca is very slow, and bad
        n_components = pca if isinstance(pca, int) else 2
        pca_model = PCA(n_components=n_components)
        X = pca_model.fit_transform(X)

    dist = sklearn.metrics.pairwise_distances(X, metric=metric)

    if type(kiezMethod) != str:
        dist2 = ug.hubness.transform_experiments(dist.copy(), kiezK, kiezMethod, kiezpresort = kiezPresort, kiezbug = kiezBug)
        return dist2
        dist2,neigh_ind2 = ug.hubness.format_dist_ind(dist2,40)

    n_candidates = X.shape[0]-1
    k_inst = Kiez(algorithm='SklearnNN', hubness=kiezMethod, hubness_kwargs={'k': kiezK, 'n_candidates': kiezK},
                  n_candidates = n_candidates, algorithm_kwargs= {'metric' : 'precomputed'})
    k_inst.fit(dist)
    dist, neigh_ind = k_inst.kneighbors()

    # breakpoint()
    # dist = CSLS(dist,kiezK)
    # dist =dist2 + np.abs(np.min(dist, axis=1)[:, np.newaxis])
    # dist,neigh_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=40, metric='precomputed').fit(dist).kneighbors()

    return dist, neigh_ind


def kiez():
    k_inst = Kiez(n_candidates=10, algorithm="Faiss", hubness="CSLS")
    k_inst.fit(source, target)
    nn_dist, nn_ind = k_inst.kneighbors(5)


def CSLS(D, k=10):
    n = D.shape[0]
    # D = (csr_matrix @ csr_matrix.T).toarray()
    # Find k-nearest neighbors for each point (excluding self)
    knn = np.argpartition(D, k+1, axis=1)[:, :k+1]  # +1 to account for self
    knn = np.array([row[row != i] for i, row in enumerate(knn)])  # remove self
    # Compute mean similarity of each point's neighborhood r(x_i)
    r = np.array([D[i, knn[i]].mean() for i in range(n)])
    # Symmetric CSLS adjustment
    csls = 2 * D - r[:, None] - r[None, :]
    return csls

from scipy.sparse import csr_matrix
import sklearn

def test_mkdistances():
    """
    Basic test function for mkdistances.
    """
    print("Running test for mkdistances...")


    # Create a simple dummy CSR matrix
    data = np.array([1, 2, 3, 4, 5, 6])
    indices = np.array([0, 1, 1, 0, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    csr_mat = csr_matrix((data, indices, indptr), shape=(3, 3))

    np.random.seed(31337)
    testmat = np.random.rand(10, 10)
    np.set_printoptions(precision=2, floatmode='fixed')

    d,i = mkdistances(csr_matrix(testmat))
    #breakpoint()

