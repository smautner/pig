import numpy as np
import glob
import networkx as nx
from lmz import *
import eden.graph as eg # eden-kernel in pip
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
import math
import ubergauss.tools as ut
from yoda import encodealignment



'''
#####################
alignmentfile to vector
##################
now we do the grap but use vec to annotate the graph

'''

def readfile(fname="data/asd.sto"):
    alignment = []
    r = []
    for line in open(fname, 'r').readlines():
        if line.startswith( "#=GC SS_cons"):
            r.append( line.split()[-1])
        elif line.startswith("#=GC cov_SS_cons"):
            r.append( line.split()[-1])
        elif line.startswith("#=GC col_entropy_0"):
            r.append( line.split()[-1])
        elif line.startswith("#=GC col_entropy_2"):
            r.append( line.split()[-1])
        elif line.startswith("#=GC col_entropy_3"):
            r.append( line.split()[-1])
        elif line.startswith("#"):
            # ignore all other lines
            continue
        elif line.startswith("/"):
            # end of file somehow there is a marker for that
            continue
        else:
            alignment.append(line.split()[-1])
    alignment = np.array([list(a.upper()) for a in alignment])
    return fname, alignment, r


def mkgraph(filename, alignment, annot, return_graph = False, method = '2020', discrete = False):
    if method == '2020':
        graph = encodealignment.nested_frag_encoder(filename, alignment, annot)
    if method == 'mainchainentropy':
        graph = encodealignment.mainchainentropy(filename,alignment, annot)
    if return_graph:
        return graph
    return eg.vectorize([graph], min_r = 1, min_d = 1, discrete = discrete) # discrete allows us to use the vec attribute, which contains the covariance info










import scipy.sparse as sparse
def vectorizedump(file):
    inn,ou = file
    f,a,r = readfile(inn)
    graph = mkgraph(f,a,r)
    ut.dumpfile(graph, f'res2/{ou}')

def filetovec(file, method = 'mainchainentropy', discrete = True):
    f,a,r = readfile(file)
    vector = mkgraph(f,a,r,return_graph = False, method = method, discrete = discrete)
    return vector

def filetograph(file, method = 'mainchainentropy'):
    f,a,r = readfile(file)
    graph = mkgraph(f,a,r,return_graph = True, method=method)
    return graph


def issmall(f):
        return readfile(f)[1].shape[0] < 3
def filtersmall(files):
     toosmall = ut.xmap(issmall,files)
     return [f for f,bad in zip(files,toosmall) if not bad]



def getfiles(path = '',removesmall=True, limit = 0):
    files = []
    for asd in 'neg pos pos2'.split():
        currentfiles  = glob.glob(f'{path}/{asd}/*')
        if not currentfiles:
            print(f'path is wrong: {path}')
        if removesmall:
            currentfiles = filtersmall(currentfiles)
        if limit:
            currentfiles = currentfiles[:limit]
        files.append(currentfiles)
    return files





def getXYFiles(path = '', limit = 0, encode = 'mainchainentropy', discrete = False):
    allfiles = getfiles(path = path,removesmall = True, limit = limit)
    flatfiles = [ a for files in allfiles for a in files]

    vectors = ut.xmap(lambda x: filetovec(x,method = 'mainchainentropy', discrete = discrete), flatfiles, processes = 88)
    #vectors = Map(lambda x: filetovec(x,method = 'mainchainentropy'), flatfiles)
    # r = sparse.vstack(vectors)
    values = [ i  for i,e in enumerate(allfiles) for z in Range(e) ]
    return vectors, values, flatfiles



