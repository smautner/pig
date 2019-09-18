import feat 
import numpy as np
def start(st,text):
    return st[:len(text)]==text

def readfile(fname="asd.sto"):
    asd = open("asd.sto").readlines()

    alignment=[]
    for line in asd:
        if start(line,"#=GC SS_cons"):
            stru = line.split()[-1]
        elif start(line,"#=GC cov_SS_cons"):
            cov = line.split()[-1]
        elif line.startswith("#"):
            # ignore all other lines
            continue
        elif line.startswith("/"):
            # end of file somehow there is a marker for that 
            continue
        else:
            alignment.append(line.split()[-1])
    alignment = np.array([list(a) for a in alignment])
    return fname, alignment, stru, cov 

class bidir:
    def __init__(self):
        self.f ={}
        self.b ={}
    def lol(self,a,b):
        self.f[a]=b
        self.b[b]=a
    def fin(self):
        self.both = dict(self.f)
        self.both.update(self.b)

def getblocks(stru):
    # shoud return a list of (start,stop)
    
    # get start/end 
    stack=[]
    bdir = bidir()
    for i,l in enumerate(stru):
        if l == '<':
            stack.append(i)
        if l == '>':
            bdir.lol(stack.pop(),i)
    bdir.fin()
    unpaired= '._:-'
    mode = 'def'
    blocks = []
    for i,l in enumerate(stru):
        if mode == 'def':
            if l == '<' or l == '>':
                start = i 
                mode = l

        elif mode == '<':
            if l != '<':
                blocks.append((start,i-1))
                if l in unpaired:
                    mode = 'def'
                if l == '>':
                    start = i 
                    mode  = l

        elif mode == '>':
            if l != '>':
                blocks.append((start,i-1))
                if l in unpaired:
                    mode = 'def'
                if l == '<':
                    start = i 
                    mode  = l
    
    realblocks = [(a-5,bdir.f[e]+5) if stru[a]=='<' else (bdir.b[a]-5,e+5)  for a,e in blocks]
    return realblocks, blocks, bdir
        

def makevec(ali, stru, cov):
    _, blocks,con = getblocks(stru)
    return [a  for b in [feat.conservation(cov,ali),
                        feat.cov_sloppycov_disturbance_instem(stru,cov,ali,blocks, con),
                        feat.stemconservation(blocks, ali), 
                        feat.stemlength(blocks)] for a in b  ]
    


name, ali, stru, cov = readfile()
print(makevec(ali, stru, cov))
