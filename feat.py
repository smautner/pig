import numpy as np


def cons_targets(pos,ali):
    r=[]
    nu,le = ali.shape
    for i in pos: 
        a = ali[:,i]
        R = (a=='G').sum()+(a=='A').sum()
        Y = (a=='C').sum()+(a=='U').sum()
        d = (a=='.').sum()

        if R < Y:
            R = Y
        if d > R:
            r.append(0)
        else:
            if R/nu <= .5:
                r.append (0)
            else:
                r.append(R/nu)
    return r 

def conservation(ali):
    # account for cov or not
    covcnt =  ali.cov.count("2")
    nu,le = ali.ali.shape
    r = cons_targets(range(le),ali.ali)
    #print (ali.cov)
    #print([ 1 if b == '2' else a  for a,b in zip(r,ali.cov) ])
    return  np.mean(r), np.mean([ 1 if b == '2' else a  for a,b in zip(r,ali.cov)  ])
        
def percstem(stru,stacks):
    a = len(stru)
    allstacks = sum([ b-a+1 for a,b in stacks ])
    bigstacks = sum([ b-a+1 for a,b in stacks if b-a+1 > 2])
    return  allstacks/a,bigstacks/a 

def get_sloppy(pos, ali):
    sloppy=0
    ok = 0
    try:
        for p in pos: 
            for a,b in zip( ali.ali[:,p], ali.ali[:,ali.con.both[p]] ):
                z = [a,b] if b > a else [b,a]
                if z == ["G","U"]: # TODO  ok ich siollte ALLE strnagen paarings detectemn 
                    sloppy+=1
                else:
                    ok +=1 
        return  sloppy/(sloppy+ok) 
    except: 
        print ("getsloppy: p con.both ali.name, ali.blocks", p, ali.con.both, ali.name,ali.blocks )

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
    return [np.argmax(points)]
            

def checov(ali, pos, con, cov):
    # chhecking covariance manually, this is necessary after we remove the worst hit line 
    covn = cov.count("2")*2
    if covn == 0: return 0 
    loss = 0
    for p in pos:
        if cov[p]=='2':
            A=-1
            B=-1 
            try:
                for a,b in zip(ali[:,p], ali[:,con.both[p]]):
                    if a != '.' and b!='.':
                        if A==-1:
                            A=a
                            B=b
                        if a!=A and b!=B: # when we find a single alternative, all is good
                            break
                else:
                    loss+=1  # all are the sam
            except: 
                print ("cov check: ali p con.both", ali, p, con.both )
                
    return loss/covn

        

def cov_sloppycov_disturbance_instem(ali):
    # % covariance in stems in stems   # with or without shorts removed
    # sloppy
    # # optionally remove fail line 

    # get indices of stems (all and long)
    hbonds = [ range(a,b+1) for a,b in ali.blocks if b-a+1>2]
    hbonds = [a for b in hbonds for a in b]
    hbonds2 = [ range(a,b+1) for a,b in ali.blocks]
    hbonds2 = [a for b in hbonds2 for a in b]

    pcov_large = sum([1 for a in hbonds if ali.cov[a]=='2'])/len(hbonds)
    pvoc_all = sum([1 for a in hbonds2 if ali.cov[a]=='2'])/len(hbonds2)
    
    # now we get the sloppy covariance 
    sl_large = get_sloppy(hbonds, ali)
    sl_all = get_sloppy(hbonds2, ali)

    # now we filter the most horrible line and do the stuff up there again. 
    weird = weirdo_detection(ali.ali)
    ali2 = ali.ali[[ a for a  in range(ali.ali.shape[0]) if a not in weird]]
    
    return pcov_large,pvoc_all, sl_large, sl_all, checov(ali2,hbonds,ali.con,ali.cov), checov(ali2,hbonds2,ali.con,ali.cov)

def stemlength(ali):
    s = [ b-a+1 for a,b in ali.blocks  ]
    s.sort()
    if len(s) ==2:
        return 0,s[0],s[1]
    elif len(s) ==1:
        return 0,0,s[0]
    elif len(s) == 0:
        return 0,0,0
    return s[0],s[-2],s[-1]


def stemconservation(ali):
    stacks2 = [ (a,b) for a,b in ali.blocks if b-a+1 > 2  ]
    return cons_stem(ali.blocks,ali.ali), cons_stem(stacks2,ali.ali)

def cons_stem(stacks,ali):

    targets = set()
    for a,b in stacks:
        for z in range(a-5,a):
            if z > 0:
                targets.add(z)
        for z in range(b+1,b+6):
            if z < ali.shape[1]:
                targets.add(z)

    return np.mean(cons_targets(targets,ali))


def beststem(ali):
    pass