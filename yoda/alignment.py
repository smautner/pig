import numpy as np
import uuid
import re
class Alignment:
    def __init__(self, ali,gc,gf, fname = ''):
        self.alignment = ali
        self.gc = gc
        self.gf = gf
        self.fname = fname
        self.label  = grepfamily(fname)

    def __repr__(self):
        return f'\n'.join([f''.join(row) for row in self.alignment])+f'\n'


def grepfamily(name):
    return re.findall(r'RF\d\d\d\d\d',name)[0]



import re
def split_on_newseq(s):
    blank_line_regex = r'>.*'
    return re.split(blank_line_regex, s.strip())

def read_fasta(fname):
    text =  open(fname, 'r').read()
    sequences = split_on_newseq(text)
    sequences = [s.strip() for s in sequences if s]
    sequences = np.array([list(a.upper()) for a in sequences])
    return Alignment(sequences, {}, {}, fname)



def read_single_alignment(text,fname):
    alignment = []
    gc = {}
    gf = {}
    for line in text.split(f'\n'):
        if line.startswith( "# STOCK") or not len(line) or line.startswith("/"):
            continue
        elif line.startswith("#=GC"):
            _, name, value = line.split()
            gc[name] = value
        elif line.startswith("#=GF"):
            split = line.split()
            gf[split[1]] = ' '.join(split[1:])
        else:
            alignment.append(line.split()[-1])
    if not alignment:
        return None
    alignment = np.array([list(a.upper()) for a in alignment])
    return Alignment(alignment, gc, gf, fname)

def fasta(ali):
    out = f''
    text= []
    filename = str(uuid.uuid4())
    for i, line in enumerate(ali.alignment):
        text.append(f'>{i}')
        text.append(f''.join(line)) # this could work...
    text = f'\n'.join(text)
    return filename

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
