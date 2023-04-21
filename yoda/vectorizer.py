import numpy as np
import glob
import networkx as nx 
from lmz import * 
import eden.graph as eg # eden-kernel in pip
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
import math
import ubergauss.tools as ut



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


def mkgraph(alignment, annot, return_graph = False):
    graph = nx.Graph()
    lifo = defaultdict(list)
    open_brace_string={")":"(",
                "]":"[",
                ">":"<","}":"{"}

    for i, ssc in enumerate(annot[0]):
        # FIND NODE LABEL 
        ct = Counter( alignment[:,i].tolist())
        for k,v in ct.most_common():
            if k in "ACGU":
                nodelabel = k 
                break
        # ADD NODE 
        #myv  = np.array([ ord(rr[i]) for rr in annot ])
        myv  = [ ord(rr[i]) for rr in annot ]
        #print(myv)
        graph.add_node(i, label=k, vec=myv)

        # ADD PAIRED BASES
        if ssc in ['(','[','<']:
            lifo['x'].append(i)
        if ssc in [')',']','>']:
            j = lifo['x'].pop()
            graph.add_edge(i, j, label='=', type='basepair', len=1)


    # ADD BACKBONE 
    lastgoodnode =  0
    for i in range(len(annot[0])-1):
        a,b = annot[0][i]=='.', annot[0][i+1]=='.'
        if a == b: # if a and b are the same we can just insert a normal edge
            graph.add_edge(i,i+1, label='-', type='backbone', len=1)
        elif a and not b: #  .-
            graph.add_edge(i,i+1, label='zz', nesting=True) #nesting are dark edges in eden 
            if lastgoodnode:
                graph.add_edge(lastgoodnode, i+1, label='-', type='backbone', len=1)
        elif b and not a: #  -.
            lastgoodnode = i 
            graph.add_edge(i,i+1, label='zz', nesting=True) #nesting are dark edges in eden 
    if return_graph:
        return graph
    return eg.vectorize([graph], min_r = 1, min_d = 1)


import scipy.sparse as sparse

def vectorizedump(file):
    inn,ou = file
    _,a,r = readfile(inn) 
    graph = mkgraph(a,r)
    ut.dumpfile(graph, f'res2/{ou}')

def filetovec(file):
    _,a,r = readfile(file) 
    vector = mkgraph(a,r,return_graph = False)
    return vector

def filetograph(file):
    _,a,r = readfile(file) 
    graph = mkgraph(a,r,return_graph = True)
    return graph


def issmall(f):
        return ali.readfile(f)[1].shape[0] < 3
def filtersmall(files):
     toosmall = ut.xmap(issmall,files)
     return [f for f,bad in zip(files,toosmall) if not bad]
     


def getfiles(removesmall=True):
    files = []
    for asd in 'neg pos pos2'.split():
        currentfiles  = glob.glob(f'{asd}/*')
        if removesmall:
            currentfiles = filtersmall(currentfiles)
        files.append(currentfiles)
    return files





def getXYFiles():
    allfiles = getfiles(removesmall = True)
    flatfiles = [ a for files in allfiles for a in files]
    vectors = ut.xmap(filetovec, flatfiles, processes = -1)
    # r = sparse.vstack(vectors)
    values = [ i  for i,e in enumerate(allfiles) for z in Range(e) ]
    return vectors, values, allfiles



