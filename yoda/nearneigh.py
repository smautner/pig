from lmz import *
import eden.display as ed
import numpy as np
import structout as so


'''
use the vectors to make a nearest neighbor model...
there is a filter to remove stuff that is too similar
'''


def overlap(a,b):
    score =  len(np.intersect1d(a.indices, b.indices)) / min(len(a.indices), len(b.indices))
    # score = np.intersect1d(a, b)/ min(len(a), len(b))
    return 1-score

def nearneigh(vecz, k = 100):

    def doarow(x):
        distances = np.array([overlap(vectors[x], vectors[y]) for y in Range(vecz)])
        if k < 1:
            return distances, 0
        indices_k = np.argpartition(distances, k )[:k]
        dist_k = [distances[ar] for ar in indices_k]
        return dist_k,indices_k

    d,i  = Transpose ( ut.xmap(doarow, Range(vecz)))
    return d,i






#########
# plotting and related functions
############


def plot_NN(dists, classes=0):
    NN = dists[:,1]
    if isinstance(classes, int):
            so.hist(NN[classes == e],bins= 40)
    else:
        for e in np.unique(classes):
            so.hist(NN[classes == e],bins= 40)

def sortbythresh(distances, indices):
    dst = [(d,s,t) for s,(d,t) in enumerate(zip(distances[:,1],indices[:,1]))] #distance source target
    dst.sort()
    return dst

def plot_alignments(filelist):
    print('todo print the file')
    #!cat {files[b]}
    #!cat {files[a]}
    graphs = [ali.filetograph(f) for f in filelist]
    ed.draw_graph_set(graphs, vertex_border=False, vertex_size=200)
    plt.show()

def plotbythresh(distances, indices, files, thresh = .7, max = 5):
    i = 0
    dst = sortbythresh(distances, indices)
    for d,a,b in dst:
        if d > thresh:
            print(f"{d}")
            if i % 2 == 0: # they al ways show up in pairs, drawing onece is enough
                plot_alignments([a,b])
            i+=1
            if i*2 > max:
                break



##############
# doing some filtering
###############


def filter_thresh(distances, indices, thresh = .07):
    rmlist = set()
    for d, i in zip(distances, indices):
        for zz, (other,dist) in enumerate(zip(i[1:],d[1:])):
            if dist < thresh and (other not in rmlist):
                rmlist.add(i[0])
    return rmlist


def filter_dump(rmlist, files, out = 'okfiles.json'):
    filesF = [f for i,f in enumerate(files) if i not in rmlist]
    ut.jdumpfile(filesF,out)
