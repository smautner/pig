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


