import random
from collections import defaultdict
import os 
import other.feat as feat
import numpy as np
import json


  
#############################
# relevant classes for operation,... read on :) 
#############################

class bidir:
    def __init__(self, stru):
        self.f ={}
        self.b ={}
        stack=defaultdict(list)
        op='<([{'
        cl='>)]}'
        d={z:i for i,z in enumerate(op)} 
        d.update({z:i for i,z in enumerate(cl)})
        for i,l in enumerate(stru):
            if l in op:
                stack[d[l]].append(i)
            if l in cl:
                self.lol(stack[d[l]].pop(),i)
        self.fin()   
        
    def lol(self,a,b):
        self.f[a]=b
        self.b[b]=a
        
    def fin(self):
        self.both = dict(self.f)
        self.both.update(self.b)
        

class Alignment:
    def __init__(self,fname, ali,stru,name):
        self.ali=ali
        self.structure = stru
        self.name = name
        self.fname = fname
        self.getblocks(stru)
        self.covariance=self.covariance()
        
        def __str__(self):
            return f"Alignment: {ali.name} {ali.fname}"

    def getblocks(self,stru):
        # shoud return a list of (start,stop)
        # get start/end 
        self.basepairs=bidir(stru)
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
        
        
        sets = self.makerealblock(stru, blocks,self.basepairs)
        block,stem = list(zip(*sets))
        def setify(x):
            s = set()
            for li in x:
                for e in li:
                    s.add(e)
            return s

        self.blockmask=list(setify(block))
        self.flankmask = list(setify(stem))
        self.blockstartend=blocks

    def makerealblock( self, stru,blocks,bdir):
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
        if not blocks:
            yield set(),set()
            


    def covariance(self):
        # return covariance string 
        
        cov = ["0"]*len(self.structure)
        
        for Open,Close in self.basepairs.f.items():
                A=-1
                B=-1 
                try:
                    for a,b in zip(self.ali[:,Open], self.ali[:,Close]):
                        if a != '.' and b!='.':
                            if A==-1:
                                A=a
                                B=b
                            elif a!=A and b!=B: # when we find a single alternative, all is good
                                cov[Open]='2'
                                cov[Close]='2'
                                break
                    else:
                        pass  # all are the sam
                except Exception as ex: 
                    print ("covcheckproblem:", self.ali[:,Open],self.ali[:,Close],ex)

        return ''.join(cov)
    

############
# parse a file
#############
        
def readfile(fname="asd.sto"):
    def start(st,text):
        return st[:len(text)]==text

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


################
# vary the parsed alignment files
################

def weirdo_detection(ali):
    height, length = ali.shape
    points = [0]*height
    for a in range(length): # for every possition 
        items = ali[:,a]
        dots = (items == '.').sum()
        notdot = height - dots
        if dots > notdot: # dots are the norm 
            for i,e in enumerate(items):
                if e != '.':
                    points[i]+=1
        if notdot > dots: # not dots are the norm 
            for i,e in enumerate(items):
                if e == '.':  # if you have a dot you get a minus point
                    points[i]+=1
        else: # same number
            pass
    return np.argsort(points)


def structurecheck(ali,stru,basepairs):
    '''return 
        1. astructure that is possible when considering the altered aligment
        2. alignments
    '''
    stru2=[]
    stru = list(stru)
    z= list(range(len(stru)))
    z.reverse()
    aliindices=[]
    for i in z:
        a = ali[:,i]
        s= stru[i]
        if all(a=='.'): # no nucleotides 
            aliindices.append(False) # mark column for deletion 
            if i in basepairs.b: stru[basepairs.b[i]] = '.' # if there is a corresponding bracket, delete that also
            continue
        aliindices.append(True)
        if s not in  ")>]}": #  no h-bonds,  keep letter
            stru2.append(s)
        else:  # ok so there is a closing bracket, so we need to decide what to do 
            other = basepairs.b[i]
            b = ali[:,other]
            for x,z in zip(a,b):
                x,z  = (x,z) if x<z else (z,x)
                if (x=='A' and z =='U') or (x=="C" and z =='G'):
                    # all is good
                    stru2.append(s)
                    break
            else: # structure broken
                stru2.append('.')
                stru[basepairs.b[i]] = '.'
    aliindices.reverse()
    stru2.reverse()        
    return ali[:,aliindices], ''.join(stru2)
            
def rm_small_stems(stru):
    stru = list(stru)
    basepairs = bidir(stru)
    last_char = ''
    state = 0 
    bracket = "{([<>])}"
    for i in range(len(stru)): 
        e = stru[i]
        if e == last_char:
            state += 1
        else:
            if last_char in bracket and state < 3: 
                if stru[i-1]==last_char:
                    stru[i-1]='.'
                    stru[basepairs.both[i-1]]='.'
                if stru[i-2]==last_char:
                    stru[i-2]='.'
                    stru[basepairs.both[i-2]]='.'
            state = 0
        last_char = e 
    return ''.join(stru) 
    
def vary_alignment(fname,ali,stru,cov):
    weird = weirdo_detection(ali)
    
    alignments = []
    alignments.append( ali[[ a for a  in range(ali.shape[0]) if a not in weird[-max(1,int(ali.shape[0]/3)):]]] )
    alignments.append( ali[[ a for a  in range(ali.shape[0]) if a not in weird[-1:]]])
    alignments.append( ali[[ a for a  in range(ali.shape[0]) if a not in weird[-2:]]])

    # for each make a new stru-line 
    structures = [stru]
    alis = [ali]
    basepairs = bidir(stru)
    
    for ali in alignments:
        nuali,nustru = structurecheck(ali,stru,basepairs)
        alis.append(nuali)
        structures.append(nustru)
        
    # then make variations where i am ignoring small stacks
    alternative_str=[rm_small_stems(s) for s in structures]
    
    
    return [ (ali,st,text) for ali,st,text in zip(alis+alis,
                                                  structures+alternative_str,
                                                  [a+b for a in ['allblock','delblock'] for b in ['ali','Xali','1ali','2ali']])]

    
    
def ali_to_dict(name, alignments, yao_scores, rnaz):
    #block =  [feat.conservation(ali),
    #                    feat.cov_sloppycov_disturbance_instem(ali),
    #                    feat.stemconservation(ali), 
    #                    feat.percstem(ali), 
    #                    feat.stemlength(ali), feat.blocktype(ali)]

    #print (block)
    list_of_dict = [feat.getfeatures(A) for A in alignments]
    master = list_of_dict[0]
    ld2= [master]
    for d,ali in zip(list_of_dict[1:],alignments[1:]):
        #print (ali.name)
        ld2.append({  ali.name+k:v  for k,v in d.items()}) 
        #ld2.append({  "diff %s %s" % (ali.name, k): v-d[k]   for k,v in master.items()}) 
    
    
    ######
    # add the differences to the original for every variation 
    ######
    r = {a:b for c in ld2 for a,b in c.items()}
    
    ####
    # add a file name
    ####
    r['name'] = name
    r['yao_score'] = yao_scores[name]
    if rnaz: # Only add RNAz as a feature if "use_rnaz in loaddata() is True
        r['rnaz_score'] = rnaz[name]
    return r 

def fnames_to_dict(fnames, yao_scores, rnaz):
    for f in fnames:
        parsed = readfile(f)
        if parsed[1].shape[0] < 3:
            continue
        alignments = vary_alignment(*parsed)
        #print ('lena',len(alignments)) # 8
        alignments2 = [Alignment(f, *a) for a in alignments]
        #print ('lenb',len(alignments2)) # 8 ok
        z=  ali_to_dict(f,alignments2, yao_scores, rnaz)
        #print ('lenc',len(z)) # 64 to few
        yield z

        
def loaddata(path, numneg = 10000, pos='both', seed=None, use_rnaz=True):
    import other.help_functions as h
    random.seed(seed)
    if os.path.isfile("tmp/blacklist.json"):
        blacklist = set(h.loadfile("tmp/blacklist.json"))
    else:
        blacklist = set()
    ##############
    # positives
    ##############
    pos1 = [ "%s/pos/%s" %(path,f) for f in  os.listdir("%s/pos" % path) if f not in blacklist]  
    pos2 = [ "%s/pos2/%s" %(path,f) for f in  os.listdir("%s/pos2" % path) if f not in blacklist] 
    if pos == 'both':
        pos = pos1+pos2
    elif pos == '1':
        pos= pos1
    else:
        pos= pos2
        
        
    ###########
    # negatives
    ##############
    negfnames =   list(os.listdir("%s/neg" % path ))
    random.shuffle(negfnames)
    print(negfnames[:5])
    neg = [ "%s/neg/%s" %(path,f) for f in  negfnames[:numneg] if f not in blacklist] 

    
    if len(pos) > numneg:
        random.shuffle(pos)
        pos=pos[:numneg]
        print ("loadfiles.py: removing some positives")

    with open(f"{path}/yaoscores.json") as f:
        yao = json.load(f)
    if use_rnaz:
        with open(f"{path}/rnaz_scores.json") as f:
            rnaz = json.load(f)
    else:
        rnaz = False

    pos = list(fnames_to_dict(pos, yao, rnaz))
    neg = list(fnames_to_dict(neg, yao, rnaz))
    
    return pos, neg

   

    
