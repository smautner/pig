from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import numpy as np
import glob
import networkx as nx
import eden.graph as eg # eden-kernel in pip
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize
import math
import ubergauss.tools as ut
from yoda.graphs import ali2graph
import yoda.alignments.alignment as ali
import re


'''
i think we should just load alignments here...
'''

def _split_on_empty_lines(s):
    blank_line_regex = r'\r?\n\s*\n'
    return re.split(blank_line_regex, s.strip())

def read_stk_file(fname):
    text = open(fname, 'r').read()
    alignments = _split_on_empty_lines(text)
    r = Map(ali.stk_to_alignment, alignments, fname = fname)
    return [a for a in r if a]



def readseedfile(fname):
    text = open(fname, 'r').read()
    alignments = _split_on_slashstockholm(text)
    r = Map(ali.stk_to_alignment, alignments, fname = False)
    return [a for a in r if a]

def _split_on_slashstockholm(s):
    blank_line_regex = r'//\n# STOCKHOLM 1.0'
    return re.split(blank_line_regex, s.strip())



def split_on_newseq(s):
    blank_line_regex = r'>.*'
    return re.split(blank_line_regex, s.strip())


def read_fasta(fname):
    text =  open(fname, 'r').read()
    sequences = split_on_newseq(text)
    sequences = [s.strip() for s in sequences if s]
    sequences = np.array([list(a.upper()) for a in sequences])
    return ali.Alignment(sequences, {}, {}, fname)

#############################
# nanaman
#############################


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


from collections import Counter



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
    return read_stk_file(f)[0].alignment.shape[0] < 3

def filtersmall(files):
    toosmall = ut.xmap(issmall, files)
    return [f for f, bad in zip(files, toosmall) if not bad]


def getXYFiles_70k(path = '', limit = 0, encode = 'mainchainentropy'):
    allfiles = _getfiles_70k(path = path,removesmall = True, limit = limit)
    flatfiles = Flatten(allfiles)
    alis = ut.xmap(read_stk_file, flatfiles)
    alis = Flatten(alis)
    mkvec = lambda x: eg.vectorize([ali2graph.mainchainentropy(x)])
    vectors = ut.xmap(mkvec, alis, processes = 88)
    values = [ i  for i,e in enumerate(allfiles) for z in Range(e) ]
    return vectors, values, alis

#########################################
# RFAMOME
####################################
def _mktuple(x):
    x = x.split()
    return (int(x[1])-1,int( x[2])-1, float(x[4])) #  ['~', '46', '124', '4.14138', '0.0468148', '6', '0.03']


def loadrfamome(path, verbose = True):
    currentfiles  = glob.glob(f'{path}/*.fasta')
    alis = ut.xmap(read_fasta, currentfiles)
    print(f"{Counter([ali.label for ali in alis])}")
    return alis


def addcov(alis):
    for a in alis:
        covname = a.fname.replace(f'.fasta',f'_1.cov')
        try:
            text = open(covname, 'r').read()
        except:
            text = f''
            print(f" no cov file ; { covname=}")
        allcov = re.findall(r'~.*', text)
        a.rscape = [_mktuple(f) for f in allcov]
    return alis

def addstructure(alis):
    for a in alis:
        strname = a.fname+f'.lina'
        text = open(strname, 'r').readlines()[1].strip()
        a.struct = text
    return alis



def process_cov(alis, debug = False):
    for ali in alis:
        try:
            s= ali.struct
            cov = ali.rscape
        except:
            print(f'structure and cov are missing... abort')

        stack = []
        pairs = []
        for i,e in enumerate(s):
            if e == f'(':
                stack.append(i)
            if e == f')':
                pairs.append((stack.pop(),i))
        annotation = [0]*len(s)
        for start,end,value in cov:
            if (start,end) in pairs:
                annotation[start] = value
                annotation[end] = value
        ali.covariance = annotation
        ali.pairs = pairs
        if debug:
            print(f"{ annotation}")
    return alis



################
# ijust do this on the rfam dataset....
####################
def addcov_rfam(alis, path ):
    for ali in alis:
        cname = ut.fixpath( f'{path}/{ali.label}_{ali.gf["ID"][3:]}.sorted.cov' )
        try:
            text = open(cname, 'r').read()
            allcov = re.findall(r'~.*', text)
            ali.rscape = [_mktuple(f) for f in allcov]
        except:
            print(f"no file: { cname=}, no problem there is just nothing covariing")
            ali.rscape = []
    return alis
