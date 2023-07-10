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
