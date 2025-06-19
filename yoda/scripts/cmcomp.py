import networkx as nx
import structout as so
from matplotlib import pyplot as plt
import lmz
import eden.graph as eg
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, rand_score, precision_recall_curve, auc
from yoda.graphs import ali2graph
import numpy as np
import yoda.ml.simpleMl as sml
import eden.display as ed
from yoda.alignments import load_rfam, filter_by_seqcount
import ubergauss.tools as ut
from yoda import graphs as ygraphs, alignments, ml
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
def compare_cm(_, names, cm1=None, cm2=None):
    cm1 = names[cm1]
    cm2 = names[cm2]
    compare_cmd = f'/home/ikea/Downloads/hsCMCompare-archlinux-x64 {cm1}.cm {cm2}.cm'
    result = subprocess.run(compare_cmd, shell=True, check=True, capture_output=True, text=True)
    #print(result.stdout)
    score = result.stdout.split()[3]
    score2 = result.stdout.split()[2]
    #score = float(score_line.split()[2])
    return {'score':float(score),'score2': float(score2)}

def run_cmcompare_pairwise(names):
    num_cm = len(names) # or just use 10 for debugging :)
    return uo.gridsearch(compare_cm,
                    param_dict = {'cm1':lmz.Range(num_cm), 'cm2':lmz.Range(num_cm)},
                    data = [False],
                    taskfilter = lambda x: x['cm1'] <= x['cm2'])



def loadcmcomp(csvpath = 'cmcompare_full_run_2024_06_27'):
    cmcompare_data = pd.read_csv(csvpath)
    cmcompare_data  = cmcompare_data.fillna(500)
    cmcompare_data['score1'] = cmcompare_data[['score', 'score2']].min(axis=1)
    cmcompare_data['score3'] = cmcompare_data[['score', 'score2']].max(axis=1)
    cmcompare_dist = to_dist(pivot_numpy(cmcompare_data))
    return cmcompare_dist




#####################
# stuff from the inference model
##################




def score(mtx,data):
    labels = data[1]
    return silhouette_score(mtx, labels, metric='precomputed')


def data_to_reffile(data):
    alis,_ = data
    reflist = [ ali.gf[f'AC'].split()[1] for ali in alis]
    with open(f'reffile.delme',f'w') as f:
        f.write(f'\n'.join(reflist))
    return reflist


def data_to_fasta(data):
    alis,_ = data
    lines = []

    # sequences = [ ali.graph.graph['sequence'] for ali in alis]
    # for i,s in enumerate(sequences):
    #     lines.append( f'>{i}')
    #     lines.append( s)

    for ali in alis:
        rfamid = ali.gf['AC'].split()[1]
        lines.append( f'>{rfamid}')
        lines.append( ali.graph.graph['sequence'])

    with open(f'fasta.delme',f'w') as f:
        f.write(f'\n'.join(lines))
    # return lines


def readCmscanAndMakeTable(data, path = 'inftools/infernal.tbl'):
    reflist = data_to_reffile(data)
    refdict = {nr:idx for idx,nr in enumerate(reflist)}
    l = len(refdict)
    distmtx= np.ones((l,l))
    distmtx*=0

    for  line in open(path,f'r').readlines():
        if not line.startswith(f"#"):
            line = line.split()
            if line[1] not in refdict or line[2] not in refdict:
                continue
            x = refdict[line[1]]
            #y = int(line[2])
            y = refdict[line[2]]
            evalue = float(line[15])
            distmtx[x,y] = evalue
            distmtx[y,x] = evalue
            # print(f"{ evalue=}")
    np.fill_diagonal(distmtx,0)
    return distmtx


def eval_agglo_ari(dist,labels, linkage = 'single'):
    rand_indices = []
    adjusted_rand_indices = []
    for n in np.unique(dist):
        predict = AgglomerativeClustering(n_clusters = None,
                                          linkage=linkage,
                                          distance_threshold=n,affinity = 'precomputed').fit_predict(dist)
        if 1850 < n < 2100:
            so.lprint(predict)
            so.lprint(labels)
        adjusted_rand_indices.append(adjusted_rand_score(predict, labels))
        rand_indices.append(rand_score(predict, labels))
    x = np.unique(dist)
    plt.scatter(x,rand_indices)
    plt.scatter(x,adjusted_rand_indices)
    plt.show()
    print(f"{max(rand_indices)=}")
    print(f"{max(adjusted_rand_indices)=}")






def infernal_tbl_to_dist(a,l,path):
    SIM_INF =  readCmscanAndMakeTable((a, l), path)
    infernal_dist = to_dist(SIM_INF)
    indices = np.where(infernal_dist == SIM_INF.max())
    noise = np.random.normal(0, 1, size=len(indices[0]))
    infernal_dist[indices] += noise
    return infernal_dist


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

import ubergauss.hubness as uh

def mkHitRateData(data,l):
    r = []
    ylabel = 'Label Hit Rate'
    for k,dist in data.items():
        for i,val in enumerate( k_clan_discovery(dist,l,50)[1:]):
            r+=[{'Distances':'unmodified','Method':k,'neighbors':i+1,ylabel:val}]

        # dist_norm = normalize(dist, axis=0)
        dist_norm = nearneigh.normalize_csls(dist)
        # dist_norm = uh.justtransform(dist.copy(),k=27,algo=2)
        # dist_norm += np.abs(dist_norm.min()) + 1

        for i,val in enumerate(k_clan_discovery(dist_norm,l,50)[1:]):
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
    return ax


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
    return ax

def slobplot(df_hitrate: pd.DataFrame, df_precrec: pd.DataFrame):
    """
    Creates a single plot with two subplots: a hit-rate curve and a
    precision-recall curve, using a shared, horizontal legend below the plots.

    This version explicitly sets font sizes for the main title and legend to
    ensure they are appropriately scaled for a "talk" context.

    Args:
        df_hitrate (pd.DataFrame): DataFrame with hit-rate data.
            Expected columns: 'neighbors', 'Label Hit Rate', 'Method', 'Distances'.
        df_precrec (pd.DataFrame): DataFrame with precision-recall data.
            Expected columns: 'recall', 'precision', 'Method', 'Distances'.
    """
    # --- Local helper functions to draw on specific axes ---
    def _plot_hitrate_on_ax(df, ax):
        """Plots the filtered hit-rate curve on a given Axes object."""
        ylabel = 'Label Hit Rate'
        df_filtered = df[(df.Distances == 'normalized') & (df.Method != 'Infernal_global')]
        sns.lineplot(data=df_filtered, x='neighbors', y=ylabel, hue='Method', ax=ax, **gethue(df_filtered))
        #ax.set_title('Neighbor Hit Rate')

    def _plot_precrec_on_ax(df, ax):
        """Plots the filtered precision-recall curve on a given Axes object."""
        df_filtered = df[(df.Distances == 'Normalized') & (df.Method != 'Infernal_global')]
        sns.lineplot(data=df_filtered, y='precision', x='recall', hue='Method', ax=ax, **gethue(df_filtered, 'Method'))
        #ax.set_title('Precision-Recall Curve')

    # --- Main plotting logic ---
    sns.set_theme()
    sns.set_context("talk")  # Use 'talk' for larger font sizes, 'paper' for smaller
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)) # Increased height for legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) # Increased height for legend

    # Generate the two plots on their respective axes
    _plot_hitrate_on_ax(df_hitrate, ax2)
    _plot_precrec_on_ax(df_precrec, ax1)

    # --- Handle Shared Legend ---
    # Get handles and labels from one plot
    handles, labels = ax2.get_legend_handles_labels()

    # Remove the individual legends automatically added to the subplots
    if ax1.get_legend():
        ax1.get_legend().remove()
    if ax2.get_legend():
        ax2.get_legend().remove()

    fig.legend(
        handles, labels, loc='lower center',             # Center the legend horizontally
        bbox_to_anchor=(0.5, +0.05),    # Position it just below the subplots
        ncol=len(labels),               # Arrange items horizontally
        #title='Method',
        fontsize=17,                    # Explicitly set legend item font size
        title_fontsize=14               # Explicitly set legend title font size
    )
    # fig.suptitle(
    #     'Performance of RNA Alignment Distance Measures (CSLS Normalized)',
    #     fontsize=22
    # )
    # LAYOUT ADJUSTMENT:
    # Adjust layout to make space for the suptitle at the top and the legend at the bottom.
    plt.tight_layout(rect=[0, 0.1, 1, 0.9]) # rect=[left, bottom, right, top]
    plt.show()
    return fig






def get_pairwise_distances(dist_matrix: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Compute pairwise distances and corresponding binary label comparisons.
    Uses the upper triangle indices of the matrix.
    """
    n = dist_matrix.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    pairwise_distance_values = dist_matrix[upper_indices]
    pairwise_same_label = np.array([labels[i] == labels[j] for i, j in zip(*upper_indices)])
    return pairwise_distance_values, pairwise_same_label


def compute_precision_recall(distances: np.ndarray, labels: np.ndarray):
    """
    Compute precision and recall values based on pairwise distance comparisons.
    """
    pairwise_distances, pairwise_labels = get_pairwise_distances(distances, labels)
    precision, recall, _ = precision_recall_curve(pairwise_labels, -pairwise_distances)
    return precision, recall


def collect_results_precrec(data: dict, labels: np.ndarray) -> list:
    """
    Process each distance matrix in the given data using both raw and normalized distances,
    then collect precision-recall values with associated metadata.
    """
    results = []
    for method_name, dist_matrix in data.items():
        # Process raw distances
        precision, recall = compute_precision_recall(dist_matrix, labels)
        for p_val, r_val in zip(precision, recall):
            results.append({'Distances': 'Raw', 'Method': method_name, 'precision': p_val, 'recall': r_val})

        # Process normalized distances using CSLS normalization from nearneigh
        normalized_matrix = nearneigh.normalize_csls(dist_matrix)

        # normalized_matrix = uh.justtransform(dist_matrix.copy(), k=27, algo=2)
        # normalized_matrix += np.abs(normalized_matrix.min()) + 1


        precision, recall = compute_precision_recall(normalized_matrix, labels)
        for p_val, r_val in zip(precision, recall):
            results.append({'Distances': 'Normalized', 'Method': method_name, 'precision': p_val, 'recall': r_val})
    return results


def plot_precision_recall_curve(dataframe, hue_column='Method', style_column=None):
    """Plot the precision-recall curve with given dataframe."""
    sns.set_theme()
    sns.set_context("talk")
    # If a style column is provided, include it in the lineplot
    plot_args = dict(y='precision', x='recall', hue=hue_column, data=dataframe)
    if style_column:
        plot_args['style'] = style_column

    # gethue must be defined elsewhere and is used to unpack additional parameters
    plot_args.update(gethue(dataframe, hue_column))

    ax = sns.lineplot(**plot_args)
    sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
    PLOT_TITLE = 'precision-recall curves of\nRNA alignment distance measures'
    # plt.title(PLOT_TITLE)
    plt.show()
    return ax

def make_results_table(data,l):
    AUC_label = "Precision/Recall AUC"
    AP_label = "Average Precision"
    results = []

    def process_matrix(method, matrix, normalized):
        """
        Calculate and print AUC and mAP values for a given matrix,
        then append the scores to the results list.

        Args:
            method (str): The method name.
            matrix: The similarity/distance matrix.
            normalized (bool): True if matrix is normalized, False otherwise.
        """
        p, r = compute_precision_recall(matrix, l)
        auc_score = auc(r, p)
        map_score =mAP(matrix, l)
        norm_flag = "yes" if normalized else "no"
        norm_text = "normalized" if normalized else "raw"
        print(f"{method} {norm_text} AUC: {auc_score}")
        print(f"{method} {norm_text} mAP: {map_score}")
        results.append({'method': method, 'scoretype': AUC_label, 'normalized': norm_flag, 'score': auc_score})
        results.append({'method': method, 'scoretype': AP_label, 'normalized': norm_flag, 'score': map_score})

    for method, dist in data.items():
        process_matrix(method, dist, normalized=False)
        normalized_matrix = nearneigh.normalize_csls(dist)
        process_matrix(method, normalized_matrix, normalized=True)

    df = pd.DataFrame(results)
    df2 = df.pivot_table(index='method', columns=['scoretype', 'normalized'], values='score', fill_value=0)
    return df2

