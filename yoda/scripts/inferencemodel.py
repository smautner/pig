from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from yoda import alignments
from sklearn.metrics import silhouette_score
import numpy as np

from yoda.alignments import alignment

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

from ubergauss import tools as ut
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


from yoda import ml, draw

import matplotlib
#matplotlib.use('module://matplotlib-backend-sixel')
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
import structout as so
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



def make_the_infernal_table():
    '''
    1. we need to download infernal (with linux binaries) and rfam.cm
    2. make reffile.delme
        python -c 'import inferencemodel as l;data = l.alignments.load_rfam();l.data_to_reffile(data);'
        python -c 'import inferencemodel as l;data = l.alignments.load_rfam();l.data_to_fasta(data);'

    goto infernaltoolfolder
    cmfetch --index Rfam.cm
    ./cmfetch -f Rfam.cm ../reffile.delme > reffile.delme.cm
    ./cmpress reffile.delme.cm
    # ./cmcalibrate --cpu 16 reffile.delme.cm we dont need to calibrate, i tested it
    ./cmscan  -E 10000 --noali --acc --tblout=delme.tbl  reffile.delme.cm fasta.delme
    '''
    pass


if __name__ == f"__main__":
    data = alignments.load_rfam()

    # data_to_fasta(data)
    # data_to_reffile(data)

    '''
    so now we need to get the dist matrix...
    cmfetch --index Rfam.cm
    cmfetch -f Rfam.cm reffile.delme > reffile.delme.cm
    cmpress reffile.delme.cm
    cmcalibrate reffile.delme.cm
    cmscan -g -E 10000 --noali --acc --tblout=delme.tbl  reffile.delme.cm fasta.delme # i should not use -g :)
    cmscan  -E 10000 --noali --acc --tblout=delme.tbl  reffile.delme.cm fasta.delme
    '''



    dist = readCmscanAndMakeTable(data)
    for linkage in 'average complete single'.split():
        print(f"{linkage=}")
        eval_agglo_ari(dist,data[1],linkage=linkage)
    exit()
    import structout as so
    from scipy import sparse
    import seaborn as sns
    sns.clustermap(dist)
    plt.show()
    so.heatmap(sparse.csr_matrix(dist))
    em = ml.embed(dist)
    draw.scatter(em, data[1])


    # draw.scatter(em, Range(data[1]))
    # for e in np.logspace(1,-50):
    #     d2 = dist.copy()
    #     d2[d2==10] = e
    #     print(f"{e=}")
    #     print(f"{score(d2, data)=}")

