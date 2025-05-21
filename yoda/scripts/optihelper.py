from yoda.graphs import alignment_to_vectors
from ubergauss import optimization as op
import yoda.ml.simpleMl as sml
import yoda.ml.distances as yodadist
import smallgraph as sg

# the plan is to make a plot  for the hyperparam opti, where i show what each
# part contributes

param = {
    "ct": 0.965,
    "RYthresh": 0,
    "norm": True,
    "d1": 0.25,
    "d2": 0.75,
    "bad_weight": 0.15,
    "fix_edges": True,
    "min_r": 2,
    "maxclust": 10,
    "nest": True,
    "simplegraph": False,
    "clusterSize": 0,
    "pca": 0,
    "metric": "euclidean",
    "kiezMethod": "csls",
    "min_d": 1}



def makedata(n):
    return  [sg.makedata(splits=3)[:2] for i in range(n)]


def mkparams():
    experiments = {}
    experiments['full'] = {}
    experiments['default kernel parameters'] =  {"min_r": 3, "min_d": 3}
    experiments['cosine distance'] = {'metric': 'cosine'}
    experiments['no hubness correction'] = {'kiezMethod': 0}
    experiments['no nesting edges'] = {'nest': False}
    experiments['no conservation on edges'] =  {"fix_edges": False}
    experiments['no conservation smoothing'] = {'d1 ': 0, 'd2': 0}
    experiments['conservation ignored'] =  {"bad_weight": 1}
    experiments['conservation unbinned'] =  {"bad_weight": 9999}

    def fix(k,v):
        v['experiment'] = k
        return v
    return  [ fix(k,v) for k,v in experiments.items()  ]

import pandas as pd
def run(data):
    params = mkparams()
    results = op.gridsearch(eval, tasks =  params, data_list=data, mp=True)
    # concatenate result dataframes
    # results = pd.concat(results)
    return results

def nosamplingdata():
    data  = [sg.makedata(splits=1)]
    res = run(data)
    return fixdata(res)


def fixdata(results):
    res = []
    for row in results.to_dict(orient='records'):
        if row['experiment'] == 'full':
            base_score = row['score']
        else:
            row['score'] = (row['score'] - base_score) / base_score
            res.append(row)
    return pd.DataFrame(res)



import seaborn as sns
import matplotlib.pyplot as plt
def plot(results):
    sns.set_theme()
    sns.set_context("talk")
    # plot a barchart, x is the 'experiment' group, y is the 'score'
    # i want sd error bars stripplot to show all the data

    # Set the style for the plot
    # sns.set_theme(style="whitegrid")

    # Create the bar plot with standard deviation error bars
    #plt.figure(figsize=(12, 6)) # Adjust figure size as needed
    # barplot = sns.barplot(x='experiment', y='score', data=results, ci='sd', capsize=.1, palette='viridis')
    barplot = sns.barplot(x='experiment', y='score', ci = None,  data=results, palette='viridis')

    # Overlay the stripplot to show individual data points
    stripplot = sns.stripplot(x='experiment', y='score', data=results, color='black', size=4,
                           jitter=True, ax=barplot.axes)

    # Rotate x-axis labels if they are long
    plt.xticks(rotation=45, ha='right')

    plt.xlabel('')
    plt.ylabel('relative degredation')
    return barplot



def eval(alis,labels,
         matrix = None,
         ct = .965,
         RYthresh = 0,
         norm = True,
         d1 = .25,
         d2= .75,
         bad_weight = 0.15,
         fix_edges = True,
         min_r = 2,
         maxclust =10,
         nest = True,
         simplegraph = False,
         clusterSize = 0,
         pca = 0,
         metric = 'euclidean',
         kiezMethod = 2,
         kiezPresort = 0,
         kiezK = 27,
         min_d = 1,
         **nothingtosee):

    if matrix is None:
        matrix = alignment_to_vectors(alis, RYthresh=RYthresh,d1=d1,d2=d2,
                                  fix_edges=fix_edges,ct=ct,bad_weight=bad_weight,
                                  min_r=min_r, min_d= min_d,normalization=norm,
                                  inner_normalization=True,clusterSize=clusterSize,
                                  nest=nest,simplegraph=simplegraph,maxclust=maxclust )


    if type(kiezMethod) != str:
        dist = yodadist.mkdistances(matrix, pca, metric, kiezMethod, kiezK, kiezPresort)
        return sml.average_precision(dist, labels)

    dist, neigh_ind = yodadist.mkdistances(matrix, pca, metric, kiezMethod, kiezK)
    return sml.average_precision_limited(dist, neigh_ind, labels)

    '''
    # ret= sml.knn_accuracy(matrix,labels)
    # ret = sml.knn_accuracy(matrix,labels)
    dist = 1- (matrix @ matrix.T).toarray()
    dist[dist<0] =0

    def make_score(method,dist,labels):
        if method == 'norm':
            # here we want to just normalize by cloumn...
            # so we overwrite the distances and tell kiez to not apply itsmethods
            dist = normalize(dist, axis=0)
            method = 'no'
        k_inst = Kiez(algorithm='SklearnNN', hubness=method, n_candidates = 40,  algorithm_kwargs= {'metric' : 'precomputed'})
        k_inst.fit(dist)
        dist, neigh_ind = k_inst.kneighbors()
        return sml.average_precision_limited(dist,neigh_ind,labels)
    # print(dist,neigh_ind,labels)
    # ret = sml.average_precision(dist,labels)
    # ret = sml.average_precision_limited(dist,neigh_ind,labels)
    # ret = {method:make_score(method,dist,labels) for method in hub_methods}
    # ret = {'score': ret,   'score_knn': sml.knn_accuracy(matrix,labels,4),  'score_ari': sml.kmeans_ari(matrix, labels)}
    # return ret
    '''
