import feat 
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

    return fname, alignment, stru, cov 

class bidir:
    def __init__(self):
        self.f ={}
        self.b ={}
    def lol(self,a,b):
        self.f[a]=b
        self.b[b]=a

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
    return realblocks, blocks
        

def makevec(ali, stru, cov):
    _, blocks = getblocks(stru)

    return [a for a in b for b in [feat.conservation(cov,ali)] ]
    


name, ali, stru, cov = readfile()
blocks = getblocks(stru)
print(getblocks(stru))
