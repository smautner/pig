import lmz
import pandas as pd
import seaborn as sns
from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from collections import Counter
import numpy as np
# from kiez import Kiez
from matplotlib import pyplot as plt
from colormap import gethue
from ubergauss import hubness
from ubergauss import tools as ut
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

def kiez_neighs(matrix, limit: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    # Set limit to maximum possible neighbors if less than 1
    if limit < 1:
        limit = matrix.shape[0] - 1
    dense_matrix = ut.zehidense(matrix)
    dense_matrix = hubness.transform(dense_matrix,dense_matrix, metric = 'cosine')
    nn = NearestNeighbors(n_neighbors=limit, metric='cosine')
    nn.fit(dense_matrix)
    dist, neigh_ind = nn.kneighbors(dense_matrix)
    return dist, neigh_ind


def getranks(matrix,l):
    # calculate dist and indices via sklearn
    dist, indices = NearestNeighbors(n_neighbors=100, metric='cosine').fit(matrix).kneighbors()

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
    c.pop(0,None)
    for k,v in list(c.items()):
        if v < 3:
            c.pop(k)
    return c



# i think this is unused
# def sim_label_idx(matrix,labels, oklabels):
#     sim = (matrix @ matrix.T).toarray()
#     # np.fill_diagonal(sim,0)
#     tups =  [[(similarity, startid, endid) for
#         endid, similarity in enumerate(sim[startid]) if labels[endid] == label ]# label are the same
#         for startid,label in enumerate(labels) if label in oklabels] # for labels with > 2 instances
#     [x.sort(key = lambda x:x[0]) for x in tups]
#     return tups

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
    '''
    clan_averages: mean vectors for the clan
    labels: the associated labels
    '''
    clan_averages = []
    labels = []
    unique_labels = list(set(l))
    for label in unique_labels:
        if label == 0:
            continue

        clan_average = np.mean(csr_matrix[l == label].toarray(),axis =0)
        clan_averages.append(clan_average)
        labels.append(label)


    # '''nearest neighbor search
    # '''
    # kz = Kiez(algorithm='SklearnNN', hubness='CSLS', n_candidates = max,  algorithm_kwargs= {'metric' : 'cosine'})
    # kz.fit(np.array(clan_averages), csr_matrix.toarray())
    # distances, indices = kz.kneighbors(max)


    # 'max' represents the number of neighbors to find
    # anyway this should do what the KIEZthing above did.
    n_neighbors: int = max
    nn: NearestNeighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn.fit(csr_matrix.toarray())
    distances, indices = nn.kneighbors(np.array(clan_averages), n_neighbors=n_neighbors)


    for i,label in enumerate(labels):
        # find the names of the original instances
        print('NEW CLAN')
        clan = [alignments[j].gf['ID'] for j in np.where(l==label)[0]]
        print(clan)

        # the indices tells us which is closest to each clan:
        print('----CLOSEST:----')
        closest = [ (alignments[j].gf['ID'], d) for d,j in zip(distances[i],indices[i])]
        for c in closest:
            print(c)
        print()
        print()







# 1. Incorrect Distance Indexing:
# In the loops, you write dist[i_id][e] and dist[j_id][e]. Since dist has shape (n_samples, limit), indexing with the absolute neighbor ID (i_id/j_id) retrieves the distance from a completely different row. It must be dist[i][e] and dist[j][e], because e is the relative neighbor index of the current source row (i or j).
# 2. Self-Match (Self-Loop) Bug:
# Nearest neighbors include the node itself at index 0. Therefore, same_label_neighbors[0] is always 0. This means j = indices[i, 0], which is just i. You end up comparing the node with itself rather than finding its closest partner. You need to exclude index 0.


import numpy as np
import collections

# def pair_rank_average(matrix, labels, oklabels, maxrank=100, rank = True):
#     '''
#     we want to write another function that returns a rank (similar to other function in this file)
#     we are given the instance vectors, and the labels
#     previously we did linear combination and straight up looking at the third instances (in this file)
#     to find more instances, now we want to look at the pair (both rows in the distance matrix(corrected by kiedz))
#     rank both rows and report the average rank. from this rank list we see how far we need to go to find a third instance.
#     '''
#     # oklabels= list(oklabels.keys())
#     dist, indices = kiez_neighs(matrix, limit=maxrank)
#     yy = np.array(labels)
#     neighbor_labels = yy[indices]
#     def get_rank(i):
#         # we find the first guy of the same class
#         same_label_neighbors  = np.where(neighbor_labels[i] == yy[i])[0]
#         # find the partner
#         if len(same_label_neighbors) <= 2:
#             return 99999999
#         j = indices[i,same_label_neighbors[1]]
#         same_label_neighbors_j  = np.where(neighbor_labels[j] == yy[i])[0]
#         # from the same_label_neighbors we can already calculate the result i think
#         d= collections.defaultdict(list)
#         for e in same_label_neighbors:
#             i_id = indices[i,e]
#             d[i_id].append(e if rank else dist[i_id][e])
#         for e in same_label_neighbors_j:
#             j_id = indices[j,e]
#             d[j_id].append(e if rank else dist[j_id][e])
#         # now we can calculate the rank
#         d.pop(i,None)
#         d.pop(j,None)
#         # remove all entries where the value-list contains only 1 element
#         # else v -> mean
#         for k,v in list(d.items()):
#             if len(v) < 2:
#                 d.pop(k)
#             else:
#                 d[k] = np.mean(v)
#         if d:
#             return min(d.values())
#         # failed
#         return 99999999
#     target_lines = np.where(np.isin(yy,oklabels))[0]
#     return np.array(Map(get_rank, target_lines))

def pair_rank_average(matrix, labels, oklabels, maxrank=300, use_rank=True):
    """
    Finds a nearest-neighbor partner of the same class for each valid instance,
    then finds a third mutual instance of the same class.
    Returns the minimum average rank (or distance) of that third instance.
    """
    # Assuming kiez_neighs is defined elsewhere in your file
    oklabels = list(oklabels.keys())
    dist, indices = kiez_neighs(matrix, limit=maxrank)

    y = np.array(labels)
    neighbor_labels = y[indices]

    def get_rank(i):
        # Find indices (ranks) in the neighbor list where the label matches the instance's label
        same_class_ranks_i = np.where(neighbor_labels[i] == y[i])[0]

        # We need at least 3 instances: the instance itself, a partner, and a 3rd target
        if len(same_class_ranks_i) <= 2:
            return np.inf

        # Find the partner 'j' (assuming index 0 is the instance 'i' itself)
        j = indices[i, same_class_ranks_i[1]]
        same_class_ranks_j = np.where(neighbor_labels[j] == y[i])[0]

        # Dictionary to group ranks/distances for mutual neighbors
        candidate_scores = collections.defaultdict(list)

        # Process the neighbors of instance i
        for rank_idx in same_class_ranks_i:
            neighbor_idx = indices[i, rank_idx]
            score = rank_idx if use_rank else dist[i, rank_idx]
            candidate_scores[neighbor_idx].append(score)

        # Process the neighbors of partner j
        for rank_idx in same_class_ranks_j:
            neighbor_idx = indices[j, rank_idx]
            score = rank_idx if use_rank else dist[j, rank_idx]
            candidate_scores[neighbor_idx].append(score)

        # Remove i and j so they aren't evaluated as the "third" instance
        candidate_scores.pop(i, None)
        candidate_scores.pop(j, None)

        # A mutual neighbor must appear in BOTH lists (len >= 2)
        valid_averages = [
            np.mean(scores)
            for scores in candidate_scores.values()
            if len(scores) >= 2
        ]

        # Return the minimum average rank/dist, or infinity if none are found
        if valid_averages:
            return min(valid_averages)

        return np.inf

    # Find all instances whose labels are in oklabels
    target_lines = np.where(np.isin(y, oklabels))[0]

    # Process each target line and return as a numpy array
    results = [get_rank(i) for i in target_lines]
    return np.array(results)




import yoda.ml.simpleMl as sml

def prep_plotHits(kraidmatrix, edenmatrix,cmcmat,l):

    random_matrix = np.random.rand(*kraidmatrix.shape)
    random_ranks = getranks(random_matrix,l)

    kraid = hubness.transform(kraidmatrix, kraidmatrix)
    nspdk = hubness.transform(edenmatrix, edenmatrix)
    cmc = cmcmat #hubness.justtransform(cmcmat, k=10, algo=2, kstart = 1)

    ranks = getranks(kraid,l)
    ranks2 = getranks(nspdk,l)
    ranks3 = getranks(cmc,l)


    sns.set_theme()
    sns.set_context("talk")

    #plt.plot([sum(ranks < x)/sum(l != 0) for x in range(1,50)])
    #plt.plot([sum(ranks2 < x)/sum(l2 != 0) for x in range(1,50)])
    ranker = lambda ranks:[sum(ranks < x)/sum(l != 0) for x in range(1,50)]

    hue = 'method'
    y_label = 'Label Hit Rate'
    # eden = [sum(ranks2 < x)/sum(l != 0) for x in range(1,50)]
    data = {y_label: ranker(ranks), 'neighbors':lmz.Range(1,50), hue: 'KRAID'}
    data1 = {y_label: ranker(ranks3), 'neighbors':lmz.Range(1,50), hue: 'CMCompare'}
    data2 = {y_label: ranker(ranks2), 'neighbors':lmz.Range(1,50),hue: 'NSPDK'}
    data3 = {y_label: ranker(random_ranks), 'neighbors':lmz.Range(1,50),hue: 'random'}

    # this reveals that all is ok, but why doesn't my cmcomp show up in the plot?? answer briefly!
    print(f"{ranker(ranks3)=}")
    df= pd.concat([pd.DataFrame(data),
                   pd.DataFrame(data2),
                   pd.DataFrame(data1),
                   pd.DataFrame(data3)])

    print(f"{ sml.average_precision(kraid, l) = }")
    print(f"{ sml.average_precision(nspdk, l) = }")
    print(f"{ sml.average_precision(cmc, l) = }")
    return df


def plotHits(df):
    # sns.set_theme('notebook', font_scale=1.5)
    sns.set_theme()
    sns.reset_orig()
    sns.set_theme('notebook', font_scale=1)
    hue = 'method'
    df['method'] = df['method'].replace('Random', 'random')
    ax= sns.lineplot(df, x= 'neighbors', y= 'Label Hit Rate', hue = hue, **gethue(df,hue))
    plt.xlabel('Neighbors')
    plt.ylabel( 'Label Hit Rate')
    ax.legend(title=None, frameon=False)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(-.1, -0.2), ncol=4)
    plt.show()
    return ax


def plotSubsetScores(matrix, l, manyseq, fewseq, rf15Labels):
    def pl(x,l):
        x.append(0)
        mask = [ ll in x for ll in l ]
        ranks = getranks(matrix[mask], l[mask])

        # ranks = calcranks(mat[mask], l[mask])
        data = [sum(ranks < xx)/sum(l[mask] != 0) for xx in range(1,21)]
        # plt.plot(data)
        return data


    many =  pl(manyseq,l) #[sum(ranks < x)/sum(l != 0) for x in range(1,50)]
    few = pl(fewseq,l) # [sum(ranks2 < x)/sum(l2 != 0) for x in range(1,50)]

    y_label = 'Label Hit Rate'
    set_label = 'Alignments'
    repeats = 10
    rand = [pl(randomhalf(l),l) for x in range (repeats)]
    randnei = lmz.Range(1,21)*repeats

    data = {y_label: many, 'neighbors':lmz.Range(1,21), set_label  : '>9 Sequences'}
    data2 = {y_label: few, 'neighbors':lmz.Range(1,21),set_label : '<10 Sequences'}
    data3 = {y_label: lmz.Flatten(rand), 'neighbors':randnei,set_label : f'{repeats} random splits ±σ'} ## !!!!
    data4 = {y_label: pl(list(np.unique(rf15Labels)), rf15Labels), 'neighbors':lmz.Range(1,21),set_label : 'test set (RFam 15)'}

    df= pd.concat([pd.DataFrame(data),pd.DataFrame(data2), pd.DataFrame(data3), pd.DataFrame(data4)])
    ax= sns.lineplot(df, x= 'neighbors', y= y_label, hue = set_label, errorbar='sd', style = set_label)
    plt.xlabel('neighbors')
    return ax

from scipy.special import comb

def calcneedlesum(l):
    needlelabels = threeinstances(l)
    sum = 0
    for e in needlelabels.values():
        sum += comb(e, 2, exact=True)
    return sum




# this might replace getranks_idx3...
def get_same_label_neighbor_ranks(
    matrix: np.ndarray,
    labels: list,
    oklabels: set,
    target_neighbor_index: int = 1
) -> np.ndarray:
    """
    Calculates the 0-indexed rank of the target_neighbor_index-th closest
    neighbor of the same class, excluding the self-match at index 0.

    To find the first partner (excluding self), set target_neighbor_index=1.
    To find the second partner (excluding self), set target_neighbor_index=2.
    """
    # Get nearest neighbors from kiez
    dist, neighbor_indices = kiez_neighs(matrix)
    yy: np.ndarray = np.array(labels)

    ranks: list = []
    for i, label in enumerate(yy):
        if label not in oklabels:
            continue

        # Extract neighbor labels and exclude the self-match at index 0
        neighbor_labs: np.ndarray = yy[neighbor_indices[i]][1:]
        # Locate indices of matching labels among the non-self neighbors
        same_label_indices: np.ndarray = np.where(neighbor_labs == label)[0]
        # Return the rank if we have found enough matching neighbors

        print(same_label_indices)
        if len(same_label_indices) >= target_neighbor_index: # [0, 10, 19]
            ranks.append(same_label_indices[target_neighbor_index - 1])
        else:
            ranks.append(999999)
    return np.array(ranks)



def needle3data(matrix,l):
    # get the labels where we have more than 2 examples
    needlelabels = threeinstances(l)


    # for each row (with the right label) we find the closest other instance according to the kernel
    # sim_label_idx = needle.sim_label_idx(matrix,l,needlelabels)

    sim_label_idx = sim_label_idx_limited(needlelabels, l,matrix,maxrank=50)

    needleranks = getranks_mix(sim_label_idx,matrix)
    rankAvg = pair_rank_average(matrix, l, needlelabels)
    distAvg = pair_rank_average(matrix, l, needlelabels, False)

    # moreranks = getranks_idx3(matrix,l,set([e[-1][2] for e in sim_label_idx]))
    # moreranks = getranks_idx3(matrix, l, set([l[e[-1][2]] for e in sim_label_idx]))
    # the way moreranks  is calculated seems overly complicated and might hide bugs. is there a straightforward way to acchieve the same result? write functions with comments etc..
    moreranks = get_same_label_neighbor_ranks(matrix,l, needlelabels, target_neighbor_index = 2)

    one = [sum(moreranks < x)/len(moreranks) for x in range(1,50)]

    two=  [sum(needleranks < x)/len(needleranks) for x in range(1,50)]
    three=  [sum(rankAvg < x)/len(rankAvg) for x in range(1,50)]
    four=  [sum(distAvg < x)/len(distAvg) for x in range(1,50)]

    method = 'Method'
    data = {'Label Hit Rate': one, 'neighbors':lmz.Range(2,len(one)+2), method : 'next in line'}
    data2 = {'Label Hit Rate': two, 'neighbors':lmz.Range(2,len(one)+2),method : 'average embedding search'}
    data3 = {'Label Hit Rate': three, 'neighbors':lmz.Range(2,len(one)+2),method : 'rank average next in line'}
    data4 = {'Label Hit Rate': four, 'neighbors':lmz.Range(2,len(one)+2),method : 'distance average next in line'}


    df= pd.concat([pd.DataFrame(data),pd.DataFrame(data2), pd.DataFrame(data3), pd.DataFrame(data4)])
    return df


# ok we need to test this needle 3 stuff
from yoda.alignments import load_rfam
import yoda.graphs as gr
def load():
    a, l = load_rfam(full=False)
    matrix = gr.alignment_to_vectors(a)
    return matrix,l


# def plotNeedle3(df):
#     title = 'Finding additional alignments for a clan'
#     ax= sns.lineplot(df, x= 'neighbors', y= 'Label Hit Rate', hue = 'Method')
#     plt.ylabel('Second Position Hit Rate')
#     sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
#     # plt.title(title)
#     return ax

def plotNeedle3(df):


    df['Method'] = df['Method'].replace('next in line', 'Next in Line (NIL)')
    df['Method'] = df['Method'].replace({
        'average embedding search': 'Average Embedding',
        'rank average next in line': 'Rank Average NIL',
        'distance average next in line': 'Distance Average NIL',
    })

    sns.reset_orig()
    sns.set_theme('notebook', font_scale=1)
    # sns.set_theme('notebook', font_scale=1.5)
    ax = sns.lineplot(data=df, x='neighbors', y='Label Hit Rate', hue='Method')
    plt.ylabel('Second Position Hit Rate')
    ax.legend(title=None, frameon=False)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    plt.xlabel('Neighbors')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(-.1, -0.2), ncol=2)
    plt.show()
    return ax






