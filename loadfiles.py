import random
from collections import defaultdict
import os 
import feat 
import numpy as np

def start(st,text):
    return st[:len(text)]==text

def readfile(fname="asd.sto"):
    alignment=[]
    for line in open(fname,'r').readlines():
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
    stack=defaultdict(list)
    bdir = bidir()
    op='<([{'
    cl='>)]}'
    d={z:i for i,z in enumerate(op)} 
    d.update({z:i for i,z in enumerate(cl)})
    for i,l in enumerate(stru):
        if l in op:
            stack[d[l]].append(i)
        if l in cl:
            bdir.lol(stack[d[l]].pop(),i)
            
    bdir.fin()
    unpaired= '._:-,'
    #allowed = '._:-,()<>'
    mode = 'def'
    blocktypes="()<>{}[]"
    blocks = []
    
    
    for i,l in enumerate(stru):
        
        # i am in default mode, 
        # i encounter a bracket, mode=current  symbol
        # else continue
        if mode == 'def':
            if l in blocktypes:
                start = i 
                mode = l

        # i am in reading block mode, 
        # the new element is not the block i was reading so far...
        elif mode in blocktypes:
            if l != mode: 
                blocks.append((start,i-1)) # end block  and decide if a new block starts or we are in unpaired territory
                if l in unpaired: 
                    mode = 'def'
                if l in blocktypes:
                    start = i 
                    mode  = l
    
    realblocks = [x for x in makerealblock(stru, blocks,bdir)]
    
    return realblocks, blocks, bdir

def makerealblock(stru,blocks,bdir):
    lstru = len(stru)
    op='<([{'
    cl= '>)]}'
    for a,e in blocks: 
        blockset = set()
        surroundset = set()
        for x in range(a,e+1):
            blockset.add(x)
            
        if stru[a] in op:
            other_a = bdir.f[a]
            other_e = bdir.f[e]
            
        elif stru[a] in cl:
            other_a = bdir.b[a]
            other_e = bdir.b[e]
        else:
            print ("make real block failed horribly:",stru,blocks,bdir,a,e)

        for x in range(a-5,a):
            if x >= 0:
                surroundset.add(x)
        for x in range(e+1,e+6):
            if x <lstru:
                surroundset.add(x)
        for x in range(other_a-5,other_a):
            if x >= 0:
                surroundset.add(x)
        for x in range(other_e+1,other_e+6):
            if x <lstru:
                surroundset.add(x)
        yield  blockset,surroundset

class alignment:
    def __init__(self,ali,stru,cov,con,blocks,name, stems):
        self.ali=ali
        self.stru = stru
        self.cov=cov
        self.con = con 
        self.blocks=blocks
        self.name= name
        self.stems=stems
    
def makevec(name, ali, stru, cov):
    stems, blocks,con = getblocks(stru) # stems are all the indices, and blocks are start-end tupples
    if len(blocks) == 0:
        print ("no blocks:", name)
    ali = alignment(ali,stru,cov,con,blocks,name,stems)
    
    #return [a  for b in [feat.conservation(ali), feat.cov_sloppycov_disturbance_instem(ali), feat.stemconservation(ali),  feat.percstem(ali),  feat.stemlength(ali)] for a in b  ]
    block =  [feat.conservation(ali),
                        feat.cov_sloppycov_disturbance_instem(ali),
                        feat.stemconservation(ali), 
                        feat.percstem(ali), 
                        feat.stemlength(ali), feat.blocktype(ali)]
   
    #print (block)
    r = {a:b for c in block for a,b in c.items()}
    r['name'] = name
    return r 

def fnames_to_vec(fnames, getall=False):
    for f in fnames:
        _,a,b,c = readfile(f)
        if a.shape[0]>2 or getall:
            yield  makevec(_,a,b,c)

def loaddata(path, numneg = 10000, pos='both',getall=False, seed=None):
    pos1 = [ "%s/pos/%s" %(path,f) for f in  os.listdir("%s/pos" % path )]  
    pos2 = [ "%s/pos2/%s" %(path,f) for f in  os.listdir("%s/pos2" % path )] 
    if pos == 'both':
        pos = pos1+pos2
    elif pos == '1':
        pos= pos1
    else:
        pos= pos2
        
    # NEGATIVES:
    if seed:
        random.seed()
    negfnames =   list(os.listdir("%s/neg" % path ))
    random.shuffle(negfnames)
    print(negfnames[:5])
    neg = [ "%s/neg/%s" %(path,f) for f in  negfnames[:numneg]] 

    pos = list(fnames_to_vec(pos,getall))
    neg = list(fnames_to_vec(neg,getall))
    
    return pos, neg
    #data = np.array(pos+neg)
    #return data ,[1]*len(pos)+[0]*len(neg)

   
