from lmz import *
import eden.display as ed
import numpy as np
import structout as so
from ubergauss import tools as ut
import yoda.vectorizer as vv
import yoda.needle as needle

'''
use the vectors to make a nearest neighbor model...
there is a filter to remove stuff that is too similar
'''


def _overlap(a,b):
    score =  len(np.intersect1d(a.indices, b.indices)) / min(len(a.indices), len(b.indices))
    # score = np.intersect1d(a, b)/ min(len(a), len(b))
    return 1-score



def neighbors(vecz, k = 100):
    def doarow(x):
        distances = np.array([_overlap(vecz[x], vecz[y]) for y in Range(vecz)])

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






#########
# plotting and related functions
############


def plot_NN(dists, classes=0):
    NN = dists[:,1]
    if isinstance(classes, int):
            so.hist(NN,bins= 40)
    else:
        for e in np.unique(classes):
            so.hist(NN[classes == e],bins= 40)
    print(f"closest {min(NN)}")

def _sortbythresh(distances, indices):
    dst = [(d,s,t) for s,(d,t) in enumerate(zip(distances[:,1],indices[:,1]))] #distance source target
    dst.sort()
    return dst

def _plot_alignments(filelist):
    #!cat {files[b]}
    #!cat {files[a]}
    graphs = [vv.filetograph(f) for f in filelist]
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


def _lsearch(seq):
    # returns index of first char
    for i,e in enumerate(seq):
        if e != '-':
            return i

def _rsearch(seq):
    numfucks = _lsearch(seq[::-1])
    return len(seq) - numfucks

def _fancyalignment(seq1,seq2,str1,str2):

    al1, al2 = needle.needle(seq1, seq2)

    def adjuststruct(al1,str1):
        # 1. insert dashes into the str
        str1=list(str1)
        re = ''
        for e in al1[::-1]:
            if e == '-':
                re+=e
            else:
                re+=str1.pop()
        return re[::-1]

    str1 =  adjuststruct(al1,str1)
    str2 =  adjuststruct(al2,str2)


    '''
    here we have st1+2 and al1+2 that all have the same lengh
    ... next we want to format it...
    everything is aligned, we just cut it into 3 pieces
    '''

    # cutoff
    left = max(Map(_lsearch, (al1,al2)))
    right = min(Map(_rsearch, (al1,al2)))


    def cutprint(item):
        nust = item[left:right]
        left1  = _countfmt(item[:left])
        right1  = _countfmt(item[right:])
        nust = f'{left1} {nust} {right1}'
        print(nust)

    Map(cutprint,(al1,str1,al2,str2))
    print()
    print()

def _countfmt(item):
    # count '-' and gives the number a pading so the string has length 4
    cnt = str(len([a for a in item if a !='-']))
    return cnt+' '*(4-len(cnt))



def test_fancy_alignment():
    ab = ('----asdasda asd as da sd---', '---asdasdasd asd a sd---', )
    _fancyalignment(*ab,*ab)
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
