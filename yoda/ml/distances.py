import numpy as np
from kiez import Kiez
from sklearn.decomposition import PCA
import sklearn
# from sklearn.metrics import pairwise_distances
params = '''
kiezK 2 50 1
'''

# metric [ 'cosine', 'euclidean']
# kiezMethod  ['csls', 'dissimlocal', 'localscaling', 'mutualproximity', 'norm']
# pca 0 170 40
#kiezMethod  ['csls', 'dissimlocal', 'localscaling', 'mutualproximity', 'norm', None]
# metric ['cityblock', 'cosine', 'euclidean']

def mkdistances(csr_matrix, pca=False, metric="euclidean", kiezMethod="csls", kiezK=5):
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

    if kiezMethod == 'norm':
        # normalize the colums
        dist = sklearn.preprocessing.normalize(dist, axis=0, norm='l2')
        kiezMethod = None

    k_inst = Kiez(algorithm='SklearnNN', hubness=kiezMethod, hubness_kwargs={'k': kiezK, 'n_candidates': kiezK}, n_candidates = 40, algorithm_kwargs= {'metric' : 'precomputed'})
    k_inst.fit(dist)
    dist, neigh_ind = k_inst.kneighbors()

    # dist = CSLS(dist,kiezK)
    # dist =dist2 + np.abs(np.min(dist, axis=1)[:, np.newaxis])
    # dist,neigh_ind = sklearn.neighbors.NearestNeighbors(n_neighbors=40, metric='precomputed').fit(dist).kneighbors()



    return dist, neigh_ind




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

    # make a random  100 x 100 matrix in npy
    testmat = np.random.rand(50, 50)
    mkdistances(csr_matrix(testmat))


    return
    print(f"Input matrix shape: {csr_mat.shape}")
    print(f"Input matrix content (dense):\n{csr_mat.toarray()}")

    # Test case 1: Default parameters
    print("\nTesting with default parameters...")
    try:
        dist1, neigh_ind1 = mkdistances(csr_mat)
        print(f"Returned dist shape: {dist1.shape}")
        print(f"Returned neigh_ind shape: {neigh_ind1.shape}")
        assert dist1.shape == (csr_mat.shape[0], 5), f"Expected dist shape (3, 5), got {dist1.shape}"
        assert neigh_ind1.shape == (csr_mat.shape[0], 5), f"Expected neigh_ind shape (3, 5), got {neigh_ind1.shape}"
        print("Default parameters test passed.")
    except Exception as e:
        print(f"Error during default parameters test: {e}")

    # Test case 2: With PCA and different metric
    print("\nTesting with PCA (2 components) and cosine metric...")
    try:
        dist2, neigh_ind2 = mkdistances(csr_mat, pca=2, metric='cosine')
        print(f"Returned dist shape: {dist2.shape}")
        print(f"Returned neigh_ind shape: {neigh_ind2.shape}")
        assert dist2.shape == (csr_mat.shape[0], 5), f"Expected dist shape (3, 5), got {dist2.shape}"
        assert neigh_ind2.shape == (csr_mat.shape[0], 5), f"Expected neigh_ind shape (3, 5), got {neigh_ind2.shape}"
        print("PCA and cosine metric test passed.")
    except Exception as e:
        print(f"Error during PCA and cosine metric test: {e}")

    # Test case 3: With Kiez method and different k
    print("\nTesting with kiezMethod='csls' and kiezK=2...")
    try:
        dist3, neigh_ind3 = mkdistances(csr_mat, kiezMethod='csls', kiezK=2)
        print(f"Returned dist shape: {dist3.shape}")
        print(f"Returned neigh_ind shape: {neigh_ind3.shape}")
        assert dist3.shape == (csr_mat.shape[0], 2), f"Expected dist shape (3, 2), got {dist3.shape}"
        assert neigh_ind3.shape == (csr_mat.shape[0], 2), f"Expected neigh_ind shape (3, 2), got {neigh_ind3.shape}"
        print("Kiez method and k test passed.")
    except Exception as e:
        print(f"Error during Kiez method and k test: {e}")

    print("\nTest for mkdistances finished.")

