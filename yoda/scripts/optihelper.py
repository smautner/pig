from yoda.graphs import alignment_to_vectors
from ubergauss import optimization as op
import yoda.ml.simpleMl as sml
import yoda.ml.distances as yodadist
import smallgraph as sg


def mkparams():
    experiments = {}
    experiments['full'] = {}
    experiments['default kernel parameters'] =  {"min_r": 3, "min_d": 3}
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





def hubness_ranking():

    # use gridsearch to eval hubness methods on all datasets
    # then pick the best config for each dataset for each hubness method and report the average rank per dataset

    myspace  = '''kiezMethod 0 5 1
    kiezK 5 45 3'''
    print(op.string_to_param_dict(myspace))
    data = [sg.makedata(splits=2) for i in range(5)]
    res = op.gridsearch(eval, data_list=[d[:2] for d in data], param_string = myspace, mp=True)
    # saved this as hubnessdata.csv
    return res

def hubness_ranking_analysis(df):
    # ---------------------------------------------------------
    # 1. Data_ID scores for Method 0
    # ---------------------------------------------------------
    print("--- 1. Method 0: Mean and Std Dev per Dataset ---")
    method_0_stats = df[df['kiezMethod'] == 0].groupby('data_id')['score'].agg(['mean', 'std'])
    # Note: std will likely be 0.0 because Method 0 scores are identical across K for the same data_id
    print(method_0_stats)

    # loop over this but the results are the same (0.2)
    stuff = df[df['kiezMethod'] == 0]['score'].agg(['mean', 'std'])
    # Note: std will likely be 0.0 because Method 0 scores are identical across K for the same data_id
    print(stuff)

    # ---------------------------------------------------------
    # 2. Select Best K for each Method (Highest Mean Score across Datasets)
    # ---------------------------------------------------------
    print("\n--- 2. Best K for Each Method (Highest Mean Score across Datasets) ---")
    k_performance = df.groupby(['kiezMethod', 'kiezK'])['score'].mean().reset_index()
    best_k_indices = k_performance.groupby('kiezMethod')['score'].idxmax()
    best_k_per_method = k_performance.loc[best_k_indices]
    best_k_per_method.rename(columns={'kiezK': 'Best_K', 'score': 'Max_Mean_Score'}, inplace=True)
    print(best_k_per_method.to_string(index=False))

    # ---------------------------------------------------------
    # 3. Overall Score Stats per Method (Mean & Std over all Datasets)
    # ---------------------------------------------------------
    print("\n--- 3. Overall Method Performance (Mean & Std over all Datasets & K) ---")
    method_overall_stats = df.groupby('kiezMethod')['score'].agg(['mean', 'std']).reset_index()
    print(method_overall_stats.to_string(index=False))

    # propper SD without ds influence
    sd_per_group = df.groupby(['kiezMethod', 'data_id'])['score'].std().reset_index()

    # 2. Calculate the mean of those SDs for each Method
    final_result = sd_per_group.groupby('kiezMethod')['score'].mean()

    print(final_result)


# In [9]: h.transform_experiments(m, k = 2, algo = 4)
# Out[9]:
# array([[1.1454092 , 3.15353057, 0.73873314, 1.74279892, 0.28335731],
#        [1.07323972, 1.00965858, 2.16785724, 1.73781899, 1.21166016],
#        [0.38281831, 1.0438886 , 1.87808977, 0.25224596, 2.97121601],
#        [1.86906366, 1.2850796 , 2.4713047 , 0.13719651, 1.43173489],
#        [3.01905953, 1.34639998, 1.02353765, 0.67907889, 1.06865922]])

# In [10]: h.transform_experiments(m, k = 2, algo = 3)
# Out[10]:
# array([[0.73070887, 0.99995202, 0.4205794 , 0.95203809, 0.07715258],
#        [0.68394641, 0.63919241, 0.99090113, 0.95119951, 0.76964193],
#        [0.13631733, 0.66368377, 0.97061354, 0.06164602, 0.99985345],
#        [0.96960262, 0.80822444, 0.99777355, 0.01864684, 0.87124769],
#        [0.99988997, 0.8368023 , 0.64923167, 0.36944074, 0.68083036]])

# In [11]: m
# Out[11]:
# array([[0.18161629, 0.87203953, 0.13550407, 0.54023027, 0.06404388],
#        [0.29678084, 0.48692118, 0.69349038, 0.93946621, 0.47760522],
#        [0.07021946, 0.33393652, 0.39852149, 0.09045373, 0.77686956],
#        [0.57936963, 0.69471497, 0.8861935 , 0.08314042, 0.63262063],
#        [0.68236213, 0.53071619, 0.26761946, 0.30005507, 0.34429569]])

# In [12]: np.argsort(m, axis = 1)
# Out[12]:
# array([[4, 2, 0, 3, 1],
#        [0, 4, 1, 2, 3],
#        [0, 3, 1, 2, 4],
#        [3, 0, 4, 1, 2],
#        [2, 3, 4, 1, 0]])

# In [13]: a = h.transform_experiments(m, k = 2, algo = 3).argsort(axis= 1)

# In [14]: b = h.transform_experiments(m, k = 2, algo = 4).argsort(axis= 1)

# In [15]: a == b
# Out[15]:
# array([[ True,  True,  True,  True,  True],
#        [ True,  True,  True,  True,  True],
#        [ True,  True,  True,  True,  True],
#        [ True,  True,  True,  True,  True],
#        [ True,  True,  True,  True,  True]])

# In [16]: a
# Out[16]:
# array([[4, 2, 0, 3, 1],
#        [1, 0, 4, 3, 2],
#        [3, 0, 1, 2, 4],
#        [3, 1, 4, 0, 2],
#        [3, 2, 4, 1, 0]])

# In [17]: b
# Out[17]:
# array([[4, 2, 0, 3, 1],
#        [1, 0, 4, 3, 2],
#        [3, 0, 1, 2, 4],
#        [3, 1, 4, 0, 2],
#        [3, 2, 4, 1, 0]])








space = '''min_r 0 3 1
min_d 0 3 1
fix_edges 0 1 1
ct .88 1
d1 0 1
d2 0 1
nest 0 1 1
kiezK 23 30 1
kiezMethod 0 5 1
simplegraph 0 1 1
bad_weight 0 .3'''


def overfit_plot(numparams=10, numdata = 3):
    '''
    we will make 2 plots:
        - a train/test accuracy plot  ?? maybe later...
        - and the other one :)
    '''

    # get some paramz
    myspace = sg.string_to_space(space)
    parameters = [myspace.sample() for i in range(numparams)]
    for i,p in enumerate(parameters):
        p['ex_id'] = i

    # prepare the data
    data_all = [sg.makedata(splits=2) for i in range(numdata)]
    datasets = [d[:2] for d in data_all]
    test_sets = [d[2] for d in data_all]

    # quick test
    # eval(*datasets[0],**parameters[0])
    # print('ok')

    results =  op.gridsearch(eval, datasets, tasks = parameters, mp=True)
    return results


def oldplot(results):
    # results is a df with a data_id column..
    # for each data_id we repeated the experiment x times, we assume data is in order!

    # pivot to have columns = data_id..
    results_pivot = results.pivot(index='ex_id', columns='data_id', values='score')

    # sort rows by mean
    results_pivot['mean_score'] = results_pivot.mean(axis=1)
    results_pivot = results_pivot.sort_values(by='mean_score', ascending=True)
    results_pivot = results_pivot.drop(columns='mean_score')



    # # we need viridis colors each column (dataset)
    # datatable =  results_pivot.values # to numpy
    # colors = sns.color_palette("viridis", n_colors=datatable.shape[1])
    # # plot each column
    # for i in range(datatable.shape[1]):
    #     plt.scatter( range(datatable.shape[0]), datatable[:,i], color=colors[i], label=f"data {i+1}")
    # plt.show()


    # now the rows are sorted. we can just reindex each row..
    results_pivot['ex_id'] = range(results_pivot.shape[0])
    # .. and melt back
    melted = results_pivot.melt(id_vars=['ex_id'], var_name='data_id', value_name='score')

    # now we are ready to scatter
    sns.set_theme('talk')
    sns.scatterplot(data=melted, x="ex_id", y="score", hue="data_id", palette='viridis')
    plt.xlabel('random parameter set (sorted)')
    plt.ylabel('score')
    plt.title('Performance across different datasets')
    plt.show()


