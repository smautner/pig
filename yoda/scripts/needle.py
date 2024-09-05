from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from collections import Counter
import numpy as np



from kiez import Kiez

def kiez_neighs(matrix, limit = 100):

    if limit < 1:
        limit = matrix.shape[0]-1

    k_inst = Kiez(algorithm='SklearnNN', hubness='csls', n_candidates = limit,  algorithm_kwargs= {'metric' : 'cosine'})
    k_inst.fit(matrix.toarray())
    dist, neigh_ind = k_inst.kneighbors()
    return dist, neigh_ind

def getranks(matrix,l):
    dist, indices = kiez_neighs(matrix)
    yy= np.array(l)
    neighbor_labels = yy[indices]
    # neighbor_labels = neighbor_labels[:, 1:]

    def searchrank(i,label):
        row_labels = neighbor_labels[i]
        indices = np.where(row_labels == label)[0]
        return indices[0] if len(indices)>0 else 999999

    ranks = np.array([searchrank(i,lab) for i,lab in enumerate(yy) if lab != 0])

    return ranks




def getranks_idx3(matrix,l,oklabels):
    '''
    returns the rank of the third guy
    '''
    dist, indices = kiez_neighs(matrix)
    yy= np.array(l)
    neighbor_labels = yy[indices]
    # neighbor_labels = neighbor_labels[:, 1:]

    def searchrank(i,label):
        row_labels = neighbor_labels[i]
        indices = np.where(row_labels == label)[0]
        return indices[1]-1 if len(indices)>1 else 999999

    ranks = np.array([searchrank(i,lab) for i,lab in enumerate(yy) if lab in oklabels])

    return ranks





def get_same_at_x(matrix,l,oklabels,maxrank= 5):
    '''
    returns the id of the closest same label partner, if its within X
    '''
    dist, indices = kiez_neighs(matrix,limit = maxrank)
    yy= np.array(l)
    neighbor_labels = yy[indices]
    # neighbor_labels = neighbor_labels[:, 1:]

    labelBad = 999999

    def searchId(i,label):
        row_labels = neighbor_labels[i]
        closest = np.where(row_labels == label)[0]
        return indices[i][closest[0]] if len(closest)>0 else labelBad

    r = [searchId(i,lab)
                    if lab != 0 and lab in oklabels else labelBad
        for i,lab in enumerate(yy)]
    # print(r)
    ids = np.array(r)
    return ids

def threeinstances(l):
    c= Counter(l)
    c.pop(0)
    for k,v in list(c.items()):
        if v < 3:
            c.pop(k)
    return c

def sim_label_idx(matrix,labels, oklabels):
    sim = (matrix @ matrix.T).toarray()
    # np.fill_diagonal(sim,0)
    tups =  [[(similarity, startid, endid) for
        endid, similarity in enumerate(sim[startid]) if labels[endid] == label ]# label are the same
        for startid,label in enumerate(labels) if label in oklabels] # for labels with > 2 instances

    [x.sort(key = lambda x:x[0]) for x in tups]

    return tups

def sim_label_idx_limited(oklabels,labels,matrix, maxrank= 5):
    assert maxrank < 100, 'call to kiezneigh will only return 100 currently'

    partners = get_same_at_x(matrix,labels,oklabels,maxrank)

    def makedata(ip,labels=None):
        i,p = ip
        same_label = np.where(labels == labels[i])[0]
        select = {i:1,p:.5}
        return [(select.get(id,0),i,id) for id in same_label ]

    tups = Map(makedata, [(i,p) for i,p in enumerate(partners) if p < 999999], labels = labels)
    [x.sort(key = lambda x:x[0]) for x in tups]

    res = []
    seen = {}
    for tup in tups:
        id = tup[-1][2], tup[-2][2]
        if id in seen:
            continue
        seen[id[::-1]] =1 # if we see this in the future, we skip
        res.append(tup)

    return res



def getrank(sli,matrix):
    # merge the rows
    c,d = sli[-1][2], sli[-2][2]
    needle = np.array([ (a + b) /2   for a,b in zip(matrix[c].toarray()[0],matrix[d].toarray()[0])])

    # search
    ranked_ids = np.argsort(-(matrix @ needle.T).T)

    # return rank of first hit
    okhits = [ sli[-i][2] for i in range(3,len(sli)+1) ]

    # print(f"{okhits=}")
    # print(f"{-(matrix @ needle.T).T[:10]=}")

    for i, id in enumerate(ranked_ids):
        if id in okhits:
            return i - 2

    return 99999999


def getranks(sli, matrix):
    return np.array(Map( getrank,sli, matrix = matrix))
