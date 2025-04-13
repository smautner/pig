import networkx as nx
from matplotlib import pyplot as plt
import lmz
import eden.graph as eg
from yoda.graphs import ali2graph
import numpy as np
import yoda.ml.simpleMl as sml
import eden.display as ed
from yoda.alignments import load_rfam, filter_by_seqcount
import ubergauss.tools as ut
from yoda import graphs as ygraphs
import subprocess
import seaborn as sns
from umap import UMAP
from yoda import draw
import ubergauss.optimization as uo 
import pandas as pd
from yoda.ml import nearneigh

import smallgraph as sg
import networkx as nx
from matplotlib import pyplot as plt
import lmz
import eden.graph as eg
from yoda.graphs import ali2graph
import numpy as np
import yoda.ml.simpleMl as sml
import eden.display as ed
from yoda.alignments import load_rfam, filter_by_seqcount
import ubergauss.tools as ut
from yoda import graphs as ygraphs


from yoda.scripts.colormap import gethue


#########################
# RUNNING CMCOMPARE 
#########################

# GENERATE CM FILES
def dumpcm(family_id1):
    # Extract alignments from Stockholm file based on family IDs
    align1_path = f'{family_id1}.fasta'
    with open('/home/ikea/Rfam.seed.utf8', 'r') as stockholm_file:
        found1 = False
        with open(align1_path, 'w') as align1_file:
            align1_file.write('# STOCKHOLM 1.0\n\n')
            for line in stockholm_file:
                if line.startswith(f'#=GF AC') and family_id1 in line:
                    found1 = True
                elif line.startswith('//'):
                    if found1:
                        align1_file.write('//')
                        break
                    found1 = False
                elif found1:
                    align1_file.write(line)
    # Generate cm files from fasta alignments
    cm1_path = align1_path.replace('.fasta', '.cm')
    infernal_cmd1 = f'cmbuild -F --informat stockholm {cm1_path} {align1_path}'
    subprocess.run(infernal_cmd1, shell=True, check=True)
    return True
    
# RUN CMCOMPARE 
def compare_cm(_,cm1=None, cm2=None):
    cm1 = names[cm1]
    cm2 = names[cm2]
    compare_cmd = f'/home/ikea/Downloads/hsCMCompare-archlinux-x64 {cm1}.cm {cm2}.cm'
    result = subprocess.run(compare_cmd, shell=True, check=True, capture_output=True, text=True)
    #print(result.stdout)
    score = result.stdout.split()[3]  
    score2 = result.stdout.split()[2]  
    #score = float(score_line.split()[2]) 
    return {'score':float(score),'score2': float(score2)}

def run_cmcompare_pairwise():
    num_cm = len(names) # or just use 10 for debugging :) 
    return uo.gridsearch(compare_cm,
                    param_dict = {'cm1':lmz.Range(num_cm), 'cm2':lmz.Range(num_cm)},
                    data = [False], 
                    taskfilter = lambda x: x['cm1'] <= x['cm2']) 










##############
# preprocessing for the evaluation
##############

def pivot_numpy(df):
    '''
    tuns the cmcompare results into a similarity matrix
    '''
    p = df.pivot(index='cm1', columns='cm2', values='score1')
    
    for i in range(p.values.shape[0]):
        for j in range(i+1,p.values.shape[0]):
            p.iloc[j,i] = p.iloc[i,j]
            
    np.fill_diagonal(p.values,0)


    
    p= p.values
    p+=p.min()
    return p

def embed(X,y):
    emb = UMAP().fit_transform(X)
    draw.scatter(emb,y)


def to_dist(X):
    # X is the similarity matrix
    # np.fill_diagonal(X,X.max()+0.05)
    X = -X
    X-=X.min()
    np.fill_diagonal(X,0)
    return X
    
    # print(X[:10,:10])
    # print('AVG PREC:', sml.average_precision(X,l))
def mAP(X,l):
    return sml.average_precision(X,l)
    
def k_clan_discovery(X,l,k):
    return [sml.clan_in_x(X,l,n) for n in range(k)]


def mkHitRateData(data):
    r = []
    ylabel = 'Label Hit Rate'
    for k,dist in data.items():
        for i,val in enumerate( cmcomp.k_clan_discovery(dist,l,50)[1:]):
            r+=[{'Distances':'unmodified','Method':k,'neighbors':i+1,ylabel:val}]

        # dist_norm = normalize(dist, axis=0)
        dist_norm = nearneigh.normalize_csls(dist)
        dist_norm += np.abs(dist_norm.min())

        for i,val in enumerate( cmcomp.k_clan_discovery(dist_norm,l,50)[1:]):
            r+=[{'Distances':'normalized','Method':k,'neighbors':i+1,ylabel:val}]

    df = pd.DataFrame(r)
    return df


def plot_hitrate_plusCSLS(df):
    ylabel= 'Label Hit Rate'
    sns.set_theme()
    sns.set_context("talk")
    ax= sns.lineplot(df, x= 'neighbors', y= ylabel,hue='Method', style='Distances', **gethue(df))
    # ax= sns.lineplot(df, x= 'neighbors', y= ylabel,hue='Method', style='Distances', palette ='bright')
    sns.move_legend( ax,"center left", bbox_to_anchor=(1, 0.5))
    #plt.title('Hit Rate of RNA alignments\nwith respect to their clan')
    plt.show()
    plt.close()


def plot_hitrate_noCSLS(df):
    ylabel= 'Label Hit Rate'
    df = df[df.Distances == 'normalized']
    df = df[df.Method != 'Infernal_global']
    sns.set_theme()
    sns.set_context("talk")
    ax= sns.lineplot(df, x= 'neighbors', y= ylabel,hue='Method', **gethue(df))
    sns.move_legend( ax,"center left", bbox_to_anchor=(1, 0.5))
    #plt.title('Hit Rate of RNA alignments\nwith respect to their clan')
    plt.show()
    plt.close()
