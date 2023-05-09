import numpy as np
class Alignment:
    def __init__(self, text, fname = ''):
        ali, gc,gf = read_single_alignment(text)
        self.alignment = ali
        self.gc = gc
        self.gf = gf
        self.fname = fname


def read_single_alignment(text):
    alignment = []
    gc = {}
    gf = {}
    for line in text.readlines():
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

    alignment = np.array([list(a.upper()) for a in alignment])
    return alignment, gc, gf
