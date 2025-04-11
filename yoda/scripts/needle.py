from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from collections import Counter
import numpy as np
from kiez import Kiez

def kiez_neighs(matrix, limit = 100):

    if limit < 1:
        limit = matrix.shape[0]-1

    k_inst = Kiez(algorithm='SklearnNN', hubness='CSLS', n_candidates = limit,  algorithm_kwargs= {'metric' : 'cosine'})
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

    return tups

    # res = []
    # seen = {}
    # for tup in tups:
    #     id = tup[-1][2], tup[-2][2]
    #     if id in seen:
    #         continue
    #     seen[id[::-1]] =1 # if we see this in the future, we skip
    #     res.append(tup)
    # return res


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


def getranks_mix(sli, matrix):
    return np.array(Map( getrank,sli, matrix = matrix))



import random
def randomhalf(lst):
    unique_values = list(set(lst))
    num_to_select = len(unique_values) // 2
    selected_values = random.sample(unique_values, num_to_select)
    return selected_values


def clanExtend(csr_matrix, l, alignments, max = 30):
    '''
    the idea is to take all the instance of one clan
    - calculate the average features
    - then search for similar instances...
    '''

    unique_labels = list(set(l))

    clan_averages = []
    labels = []
    for label in unique_labels:
        if label == 0:
            continue

        clan_average = np.mean(csr_matrix[l == label].toarray(),axis =0)
        clan_averages.append(clan_average)
        labels.append(label)

    kz = Kiez(algorithm='SklearnNN', hubness='CSLS', n_candidates = max,  algorithm_kwargs= {'metric' : 'cosine'})

    # kz = Kiez(n_candidates=max,hubness='CSLS')
    kz.fit(np.array(clan_averages), csr_matrix.toarray())
    distances, indices = kz.kneighbors(max)

    for i,label in enumerate(labels):
        # print original members + distances:
        print('NEW CLAN')
        clan = [alignments[j].gf['ID'] for j in np.where(l==label)[0]]
        # print([a.name for a in clan])
        print(clan)
        print('----CLOSEST:----')
        closest = [ (alignments[j].gf['ID'], d) for d,j in zip(distances[i],indices[i])]
        for c in closest:
            print(c)
        print()
        print()





import collections

def pair_rank_average(matrix, labels, oklabels, maxrank=100, rank = True):
    '''
    we want to write another function that returns a rank (similar to other function in this file)
    we are given the instance vectors, and the labels
    previously we did linear combination and straight up looking at the third instances (in this file)
    to find more instances, now we want to look at the pair (both rows in the distance matrix(corrected by kiedz))
    rank both rows and report the average rank. from this rank list we see how far we need to go to find a third instance.
    '''
    oklabels= list(oklabels.keys())
    dist, indices = kiez_neighs(matrix, limit=maxrank)
    yy = np.array(labels)
    neighbor_labels = yy[indices]


    def get_rank(i):
        # we find the first guy of the same class
        same_label_neighbors  = np.where(neighbor_labels[i] == yy[i])[0]
        # find the partner
        if len(same_label_neighbors) == 0:
            return 99999999
        j = indices[i,same_label_neighbors[0]] # apparently 0 is not the self? weird
        same_label_neighbors_j  = np.where(neighbor_labels[j] == yy[i])[0]



        # from the same_label_neighbors we can already calculate the result i think
        d= collections.defaultdict(list)
        for e in same_label_neighbors:
            i_id = indices[i,e]
            d[i_id].append(e if rank else dist[i_id][e])
        for e in same_label_neighbors_j:
            j_id = indices[j,e]
            d[j_id].append(e if rank else dist[j_id][e])
        # now we can calculate the rank
        d.pop(i,None)
        d.pop(j,None)
        # remove all entries where the value-list contains only 1 element
        # else v -> mean
        for k,v in list(d.items()):
            if len(v) < 2:
                d.pop(k)
            else:
                d[k] = np.mean(v)

        if d:
            return min(d.values())
        # failed
        return 99999999


    target_lines = np.where(np.isin(yy,oklabels))[0]
    return np.array(Map(get_rank, target_lines))


