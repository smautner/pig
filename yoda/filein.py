import numpy as np
import glob
import networkx as nx
from lmz import *
import eden.graph as eg # eden-kernel in pip
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
import math
import ubergauss.tools as ut
from yoda import ali2graph



'''
#####################
alignmentfile to vector
##################
now we do the grap but use vec to annotate the graph

'''


import re

def split_on_empty_lines(s):
    blank_line_regex = r'\r?\n\s*\n'
    return re.split(blank_line_regex, s.strip())


def readfile(fname):
    text =  open(fname, 'r').read()
    alignments = split_on_empty_lines(text)
    return Map(ali.Alignment(alignments), fname = fname)




def mkgraph(alignment, return_graph = False, method = '2020', discrete = True):

    if method == '2020':
        graph = ali2graph.nested_frag_encoder(alignment)
    if method == 'mainchainentropy':
        graph = ali2graph.mainchainentropy(alignment)
    if return_graph:
        return graph
    return eg.vectorize([graph], min_r = 1, min_d = 1, discrete = discrete) # discrete allows us to use the vec attribute, which contains the covariance info




def ali2vec(alignment, method = 'mainchainentropy', discrete = True):
    vector = mkgraph(alignment,return_graph = False, method = method, discrete = discrete)
    return vector

def filetograph(file, method = 'mainchainentropy'):
    ali = readfile(file)
    graph = mkgraph(ali[0],return_graph = True, method=method)
    return graph



# TODO, theese are not actually bad they just contain columns without nucleotides
bad = '''/home/stefan/WEINBERG//neg/550-1278861-0-0.sto
/home/stefan/WEINBERG//neg/550-1454352-0-0.sto
/home/stefan/WEINBERG//neg/550-1382269-0-0.sto
/home/stefan/WEINBERG//neg/769-1136-0-0.sto
/home/stefan/WEINBERG//neg/769-1136-0-0.sto
/home/stefan/WEINBERG//neg/550-1261710-0-0.sto
/home/stefan/WEINBERG//neg/550-1261710-0-0.sto
/home/stefan/WEINBERG//neg/550-211660-0-0.sto
/home/stefan/WEINBERG//neg/592-880-0-0.sto
/home/stefan/WEINBERG//neg/592-880-0-0.sto
/home/stefan/WEINBERG//neg/592-880-0-0.sto
/home/stefan/WEINBERG//neg/592-880-0-0.sto
/home/stefan/WEINBERG//neg/592-880-0-0.sto
/home/stefan/WEINBERG//neg/550-2011241-0-0.sto
/home/stefan/WEINBERG//neg/550-2011241-0-0.sto
/home/stefan/WEINBERG//neg/550-2025985-0-0.sto
/home/stefan/WEINBERG//neg/550-2025985-0-0.sto
/home/stefan/WEINBERG//neg/550-2025985-0-0.sto
/home/stefan/WEINBERG//neg/550-2025985-0-0.sto
/home/stefan/WEINBERG//neg/550-1260425-0-0.sto
/home/stefan/WEINBERG//neg/550-1927583-0-0.sto
/home/stefan/WEINBERG//neg/550-1927583-0-0.sto
/home/stefan/WEINBERG//neg/550-2308630-0-0.sto
/home/stefan/WEINBERG//neg/550-2308630-0-0.sto
/home/stefan/WEINBERG//neg/550-2308630-0-0.sto
/home/stefan/WEINBERG//neg/550-1708760-0-0.sto'''.split('\n')


def _getfiles_70k(path = '',removesmall=True, limit = 0):
    files = []
    for asd in 'neg pos pos2'.split():
        currentfiles  = glob.glob(f'{path}/{asd}/*')
        if not currentfiles:
            print(f'path is wrong: {path}')
        if removesmall:
            currentfiles = filtersmall(currentfiles)
        if limit:
            currentfiles = currentfiles[:limit]
        currentfiles = [c for c in currentfiles if c not in bad]
        files.append(currentfiles)
    return files

def issmall(f):
    return readfile(f)[0].alignment.shape[0] < 3

def filtersmall(files):
    toosmall = ut.xmap(issmall, files)
    return [f for f, bad in zip(files, toosmall) if not bad]


def getXYFiles_70k(path = '', limit = 0, encode = 'mainchainentropy'):
    allfiles = _getfiles_70k(path = path,removesmall = True, limit = limit)
    flatfiles = flatten(allfiles)
    alis = ut.xmap(readfile, flatfiles)
    alis = flatten(alis)
    vectors = ut.xmap(lambda x: ali2vec(x,method = 'mainchainentropy'),
                      alis, processes = 88)
    values = [ i  for i,e in enumerate(allfiles) for z in Range(e) ]
    return vectors, values, alis



