from lmz import Zip, Range,Transpose
import numpy as np
from ubergauss import tools as ut
from yoda.ml.simpleMl import overlap_coef
from yoda.alignments import ali2graph
from yoda.draw import _fancyalignment

'''
use the vectors to make a nearest neighbor model...
there is a filter to remove stuff that is too similar
'''


def neighbors(vecz, k = 100):
    def doarow(x):
        distances = np.array([overlap_coef(vecz[x], vecz[y]) for y in Range(vecz)])

        if k < 1:
            return distances, 0

        indices_k = np.argsort(distances)[:k]
        # fix order such that first id is the instance itself
        # note distances dont matter because if we are not first the distance is zero
        if indices_k[0] != x:
            other = np.where(indices_k == x)[0][0]
            indices_k[0], indices_k[other]= indices_k[other], indices_k[0]
        dist_k = [distances[ar] for ar in indices_k]

        return dist_k,indices_k

    d,i  = Transpose ( ut.xmap(doarow, Range(vecz)))
    d = np.array(d)
    i = np.vstack(i)
    return d,i





def _plot_alignments(filelist):
    #!cat {files[b]}
    #!cat {files[a]}
    file2graph = lambda x: ali2graph.mainchainentripy(vv.read_stk_file(x)[0])
    graphs = [file2graph for f in filelist]
    # ed.draw_graph_set(graphs, vertex_border=False, vertex_size=200)
    #plt.show()
    #so.gprint(graphs, size = 40)

    # for f in graphs:
    #     print(f.graph.get('sequence', ''))
    #     print(f.graph.get('structure', ''))

    seq1 = graphs[0].graph.get('sequence', '')
    str1 = graphs[0].graph.get('structure', '')
    seq2 = graphs[1].graph.get('sequence', '')
    str2 = graphs[1].graph.get('structure', '')

    _fancyalignment(seq1,seq2,str1,str2)


def _sortbythresh(distances, indices):
    dst = [(d,s,t) for s,(d,t) in enumerate(zip(distances[:,1],indices[:,1]))] #distance source target
    dst.sort()
    return dst


def plotbythresh(distances, indices, files, thresh = .7, max = 5):
    i = 0
    dst = _sortbythresh(distances, indices)
    for d,a,b in dst:
        if d > thresh:
            if i % 2 == 0: # they al ways show up in pairs, drawing onece is enough
                print(f"{d} {files[a]} {files[b]}")
                _plot_alignments([files[z] for z in [a,b]])
            i+=1
            if i > max*2:
                break


##############
# doing some filtering
###############


def filter_thresh(distances, indices, thresh = .7, checkmax = 400):
    rmlist = set()
    for d, i in zip(distances, indices):
        for zz, (other,dist) in enumerate(Zip(i[1:],d[1:])[:checkmax]):
            if dist > thresh:
                break
            if dist < thresh and (other not in rmlist):
                rmlist.add(i[0])
                break
    return rmlist



def filter_dump(rmlist, files, out = 'okfiles.json'):
    filesF = [f for i,f in enumerate(files) if i not in rmlist]
    ut.jdumpfile(filesF,out)


